import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO

# --- Page Config ---
st.set_page_config(
    page_title="PCB Vision | Offline YOLO Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .status-card-ok {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 6px solid #2e7d32;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        color: #1b5e20;
        margin-bottom: 1rem;
    }
    .status-card-fail {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 6px solid #c62828;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        color: #b71c1c;
        margin-bottom: 1rem;
    }
    .part-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.15rem 0.25rem 0.15rem 0;
    }
    .pill-missing { background-color: #fdecea; color: #b71c1c; border: 1px solid #f5c6cb; }
    .pill-ok { background-color: #e8f5e9; color: #1b5e20; border: 1px solid #c3e6cb; }
    </style>
    """,
    unsafe_allow_html=True,
)

BOX_COLORS = [
    "#00C853", "#2979FF", "#FF9100", "#D500F9",
    "#00B8D4", "#FFD600", "#FF4081", "#FF3D00",
]

# --- Session State Initialization ---
defaults = {
    "gt_image": None,
    "gt_filename": None,
    "parts": [],  # [{"name": str, "box": (x, y, w, h), "color": str}]
    "trained_model_path": None,
    "trained_model_name": None,
    "last_box_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --- Drawing Helpers ---
def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def draw_label_pill(img: np.ndarray, text: str, x: int, y: int, bg_color: tuple):
    """Draws a crisp, readable label pill directly above (or inside) a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, img.shape[1] / 1800.0)
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    top = y - th - baseline - 8
    if top < 0:
        top = y + 4  # Place inside top-left if near upper edge

    cv2.rectangle(img, (x, top), (x + tw + 10, top + th + baseline + 8), bg_color, -1)
    cv2.putText(
        img,
        text,
        (x + 5, top + th + 3),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def compose_labeled_canvas_bg(gt_img: np.ndarray, parts: list) -> Image.Image:
    """Renders current box numbers and names directly onto the editable canvas background."""
    overlay = gt_img.copy()
    for idx, part in enumerate(parts):
        x, y, w, h = part["box"]
        rgb = hex_to_rgb(part["color"])
        draw_label_pill(overlay, f"#{idx + 1}: {part['name']}", x, y, rgb)
    return Image.fromarray(overlay)


# --- Image Alignment ---
def align_image(test_img: np.ndarray, gt_img: np.ndarray) -> np.ndarray:
    gray_gt = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=2500)
    kp1, des1 = orb.detectAndCompute(gray_gt, None)
    kp2, des2 = orb.detectAndCompute(gray_test, None)

    if des1 is None or des2 is None:
        return cv2.resize(test_img, (gt_img.shape[1], gt_img.shape[0]))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    good_matches = matches[: int(len(matches) * 0.25)]

    if len(good_matches) < 10:
        return cv2.resize(test_img, (gt_img.shape[1], gt_img.shape[0]))

    pts_gt = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_test = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(pts_test, pts_gt, cv2.RANSAC, 5.0)
    if matrix is None:
        return cv2.resize(test_img, (gt_img.shape[1], gt_img.shape[0]))

    h, w = gt_img.shape[:2]
    return cv2.warpPerspective(test_img, matrix, (w, h))


# --- Improved One-Shot Dataset Generator (with Synthetic Missing-Part Dropout) ---
def build_augmented_dataset(
    gt_img: np.ndarray, parts: list, work_dir: Path, num_samples: int = 36, progress_bar=None
):
    """Generates an augmented dataset with brightness/shift jitter AND synthetic component dropout."""
    img_dir = work_dir / "images" / "train"
    lbl_dir = work_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    h_img, w_img = gt_img.shape[:2]
    gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)

    # Estimate median board color (for realistic synthetic missing-part masking)
    median_pcb_color = np.median(gt_bgr.reshape(-1, 3), axis=0).astype(np.uint8)
    class_names = {idx: part["name"] for idx, part in enumerate(parts)}

    rng = np.random.default_rng(42)
    for i in range(num_samples):
        aug = gt_bgr.copy()
        dropped_indices = set()

        # On 40% of samples (after sample 0), randomly mask 1-2 components so YOLO learns
        # visual appearance instead of memorizing fixed spatial neighbors.
        if i > 2 and len(parts) > 2 and rng.random() < 0.40:
            drop_count = int(rng.integers(1, min(3, len(parts))))
            dropped_indices = set(rng.choice(len(parts), size=drop_count, replace=False).tolist())
            for d_idx in dropped_indices:
                x, y, w, h = parts[d_idx]["box"]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_img, x + w), min(h_img, y + h)
                # Fill with median PCB color + subtle noise to mimic bare board
                patch_noise = rng.normal(0, 6, (y2 - y1, x2 - x1, 3))
                aug[y1:y2, x1:x2] = np.clip(median_pcb_color + patch_noise, 0, 255).astype(np.uint8)

        aug = aug.astype(np.float32)
        if i > 0:
            alpha = rng.uniform(0.82, 1.18)
            beta = rng.uniform(-20, 20)
            aug = np.clip(aug * alpha + beta, 0, 255)
            aug = np.clip(aug + rng.normal(0, 3.5, aug.shape), 0, 255)
        aug = aug.astype(np.uint8)

        # Micro-shift translation
        dx = int(rng.integers(-14, 15)) if i > 0 else 0
        dy = int(rng.integers(-14, 15)) if i > 0 else 0
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aug = cv2.warpAffine(aug, M, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)

        cv2.imwrite(str(img_dir / f"gt_{i:03d}.jpg"), aug)

        label_lines = []
        for cls_id, part in enumerate(parts):
            if cls_id in dropped_indices:
                continue  # Omitted intentionally because this part was masked out
            x, y, w, h = part["box"]
            nx1 = np.clip(x + dx, 0, w_img - 1)
            ny1 = np.clip(y + dy, 0, h_img - 1)
            nx2 = np.clip(x + w + dx, 0, w_img - 1)
            ny2 = np.clip(y + h + dy, 0, h_img - 1)

            bw = (nx2 - nx1) / w_img
            bh = (ny2 - ny1) / h_img
            cx = (nx1 + nx2) / (2.0 * w_img)
            cy = (ny1 + ny2) / (2.0 * h_img)

            if bw > 0.005 and bh > 0.005:
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (lbl_dir / f"gt_{i:03d}.txt").write_text("\n".join(label_lines))

        if progress_bar is not None:
            progress_bar.progress(int(((i + 1) / num_samples) * 100))

    data_yaml = {
        "path": str(work_dir.resolve()),
        "train": "images/train",
        "val": "images/train",
        "names": class_names,
    }
    yaml_path = work_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml))
    return yaml_path


def train_offline_yolo(
    gt_img: np.ndarray, parts: list, base_weights: str, epochs: int, imgsz: int, epoch_bar, epoch_text
):
    """Fine-tunes YOLO26/YOLO11 locally with live epoch progress updates."""
    work_dir = Path(tempfile.mkdtemp(prefix="pcb_yolo_"))
    yaml_path = build_augmented_dataset(gt_img, parts, work_dir)

    model = YOLO(base_weights)

    def on_epoch_end(trainer):
        current = trainer.epoch + 1
        pct = min(100, int((current / epochs) * 100))
        epoch_bar.progress(pct)
        epoch_text.caption(f"Training {base_weights} — Epoch {current} / {epochs} ({pct}%)")

    model.add_callback("on_fit_epoch_end", on_epoch_end)

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=4 if imgsz >= 1024 else 8,
        workers=0,       # Windows-safe: prevents multiprocessing spawn crash
        cache=True,
        mosaic=0.0,      # Disabled: preserves fixed PCB spatial layout
        fliplr=0.0,      # Disabled: PCB components are orientation-specific
        flipud=0.0,
        project=str(work_dir / "runs"),
        name="pcb_detector",
        exist_ok=True,
        verbose=False,
        plots=False,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    return str(best_weights)


# --- Offline YOLO Inspection ---
def inspect_board_yolo(
    model: YOLO,
    gt_img: np.ndarray,
    test_img: np.ndarray,
    parts: list,
    conf_threshold: float,
    imgsz: int,
    use_alignment: bool,
):
    processed_test = (
        align_image(test_img, gt_img)
        if use_alignment
        else cv2.resize(test_img, (gt_img.shape[1], gt_img.shape[0]))
    )

    preds = model.predict(processed_test, conf=conf_threshold, imgsz=imgsz, verbose=False)[0]

    detected = {}  # cls_id -> (conf, (x1, y1, x2, y2))
    if preds.boxes is not None:
        for box in preds.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if cls_id not in detected or conf > detected[cls_id][0]:
                detected[cls_id] = (conf, (x1, y1, x2, y2))

    annotated = processed_test.copy()
    missing_parts = []
    present_parts = []

    for cls_id, part in enumerate(parts):
        if cls_id in detected:
            conf, (x1, y1, x2, y2) = detected[cls_id]
            present_parts.append((part["name"], conf))
            color_ok = (40, 167, 69)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_ok, 2)
            draw_label_pill(annotated, f"{part['name']} ({conf:.2f})", x1, y1, color_ok)
        else:
            missing_parts.append(part["name"])
            x, y, w, h = part["box"]
            color_missing = (220, 53, 69)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color_missing, 3)
            draw_label_pill(annotated, f"MISSING: {part['name']}", x, y, color_missing)

    return missing_parts, present_parts, annotated


# --- Sidebar: Global Settings & Status ---
with st.sidebar:
    st.title("🔬 PCB Vision")
    st.caption("100% Offline YOLO Inspection Suite")
    st.divider()

    st.subheader("Model Configuration")
    base_weights = st.selectbox(
        "Local Base Weights",
        ["yolo26m.pt", "yolo26n.pt", "yolo11m.pt", "yolo11n.pt"],
        index=0,
    )
    imgsz = st.select_slider(
        "Training & Inference Resolution",
        options=[640, 800, 1024, 1280],
        value=1024,
        help="1024+ significantly improves small component recall (resistors, LEDs, capacitors).",
    )
    epochs = st.slider("Training Epochs", min_value=20, max_value=80, value=50, step=5)
    conf_threshold = st.slider(
        "Detection Confidence",
        min_value=0.15,
        max_value=0.85,
        value=0.35,
        step=0.05,
        help="Parts detected below this score are flagged as missing.",
    )
    use_alignment = st.toggle("Auto-align boards (ORB)", value=True)

    st.divider()
    st.subheader("System Status")
    st.write(f"**Ground Truth:** {'Loaded ✅' if st.session_state.gt_image is not None else 'Not set ⚪'}")
    st.write(f"**Labeled Parts:** `{len(st.session_state.parts)}`")
    st.write(
        f"**Trained Model:** `{st.session_state.trained_model_name}` ✅"
        if st.session_state.trained_model_path
        else "**Trained Model:** Not trained ⚪"
    )

# --- Main Header ---
st.title("PCB Missing Part Inspector")
st.caption("Define components once on a reference board, train locally, and inspect batches completely offline.")

tab_setup, tab_inspect = st.tabs(["🛠️ 1. Ground Truth & Training", "🔍 2. Inspect Boards"])

# === TAB 1: GROUND TRUTH, ON-DIAGRAM LABELING & TRAINING ===
with tab_setup:
    gt_file = st.file_uploader(
        "Upload Reference Board (Ground Truth)",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear, well-lit photo of a complete, defect-free PCB.",
    )

    if gt_file:
        # Show upload/processing progress bar when a new file is uploaded
        if st.session_state.gt_filename != gt_file.name:
            upload_bar = st.progress(0, text="Uploading and decoding reference image...")
            gt_pil = Image.open(gt_file).convert("RGB")
            upload_bar.progress(50, text="Preparing image buffer...")
            st.session_state.gt_image = np.array(gt_pil)
            st.session_state.gt_filename = gt_file.name
            st.session_state.parts = []
            st.session_state.last_box_count = 0
            st.session_state.trained_model_path = None
            upload_bar.progress(100, text="Reference image ready!")
            time.sleep(0.25)
            upload_bar.empty()

        gt_np = st.session_state.gt_image
        orig_h, orig_w = gt_np.shape[:2]

        canvas_w = 720
        scale = canvas_w / orig_w
        canvas_h = int(orig_h * scale)
        next_color = BOX_COLORS[len(st.session_state.parts) % len(BOX_COLORS)]

        # Render live #Number: Name labels directly onto the editable canvas background
        labeled_bg_pil = compose_labeled_canvas_bg(gt_np, st.session_state.parts)

        col_canvas, col_panel = st.columns([2.2, 1], gap="large")

        with col_canvas:
            st.subheader("Interactive Board Diagram")
            st.caption("Draw a box around each component. Labels update directly on this diagram.")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.06)",
                stroke_width=3,
                stroke_color=next_color,
                background_image=labeled_bg_pil,
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="rect",
                key="gt_canvas",
            )

        with col_panel:
            st.subheader("Component Labels")
            objects = (
                canvas_result.json_data["objects"]
                if canvas_result.json_data and "objects" in canvas_result.json_data
                else []
            )

            # Show a quick loading bar when a new bounding box field is added/synced
            if len(objects) != st.session_state.last_box_count:
                sync_bar = st.progress(0, text="Syncing bounding box fields...")
                sync_bar.progress(60, text="Updating diagram labels...")
                st.session_state.last_box_count = len(objects)
                sync_bar.progress(100, text="Done!")
                time.sleep(0.15)
                sync_bar.empty()

            if not objects:
                st.info("Draw rectangles on the board to register components.")
            else:
                updated_parts = []
                with st.container(height=min(520, max(180, len(objects) * 85))):
                    for idx, obj in enumerate(objects):
                        x = int(obj["left"] / scale)
                        y = int(obj["top"] / scale)
                        w = int((obj["width"] * obj["scaleX"]) / scale)
                        h = int((obj["height"] * obj["scaleY"]) / scale)

                        box_color = obj.get("stroke", BOX_COLORS[idx % len(BOX_COLORS)])
                        default_name = (
                            st.session_state.parts[idx]["name"]
                            if idx < len(st.session_state.parts)
                            else f"Part_{idx + 1}"
                        )

                        c_swatch, c_input = st.columns([0.15, 0.85])
                        with c_swatch:
                            st.markdown(
                                f"<div style='margin-top:8px; width:16px; height:16px; "
                                f"border-radius:4px; background:{box_color};'></div>",
                                unsafe_allow_html=True,
                            )
                        with c_input:
                            part_name = st.text_input(
                                f"Box #{idx + 1}",
                                value=default_name,
                                key=f"name_input_{idx}",
                                label_visibility="collapsed",
                                placeholder=f"Name for Box #{idx + 1}",
                            )
                        updated_parts.append(
                            {"name": part_name, "box": (x, y, w, h), "color": box_color}
                        )

                st.session_state.parts = updated_parts

                st.divider()
                if st.button("🚀 Train Offline YOLO Model", type="primary", use_container_width=True):
                    if not os.path.exists(base_weights):
                        st.error(
                            f"Local weights `{base_weights}` not found. "
                            "Place the `.pt` file in the same folder as `app.py`."
                        )
                    else:
                        with st.status(f"Training {base_weights} locally...", expanded=True) as status:
                            st.write("1. Generating augmented dataset with synthetic missing-part dropout...")
                            epoch_text = st.empty()
                            epoch_bar = st.progress(0)

                            best_pt = train_offline_yolo(
                                st.session_state.gt_image,
                                st.session_state.parts,
                                base_weights,
                                epochs,
                                imgsz,
                                epoch_bar,
                                epoch_text,
                            )
                            st.session_state.trained_model_path = best_pt
                            st.session_state.trained_model_name = base_weights
                            status.update(label="Training complete! Ready to inspect.", state="complete")
                        st.toast("Model trained successfully! Switch to Tab 2.", icon="✅")

# === TAB 2: BATCH INSPECTION ===
with tab_inspect:
    if not st.session_state.trained_model_path:
        st.info("Complete Step 1: Upload a Ground Truth board, label components, and train the YOLO model.")
    else:
        test_files = st.file_uploader(
            "Upload Board Photos to Inspect",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if test_files:
            model = YOLO(st.session_state.trained_model_path)
            batch_bar = st.progress(0, text="Starting batch inspection...")

            for idx, test_file in enumerate(test_files):
                batch_bar.progress(
                    int((idx / len(test_files)) * 100),
                    text=f"Inspecting {test_file.name} ({idx + 1}/{len(test_files)})...",
                )

                test_pil = Image.open(test_file).convert("RGB")
                test_np = np.array(test_pil)

                missing, present, annotated_img = inspect_board_yolo(
                    model,
                    st.session_state.gt_image,
                    test_np,
                    st.session_state.parts,
                    conf_threshold,
                    imgsz,
                    use_alignment,
                )

                with st.container(border=True):
                    st.subheader(f"📄 {test_file.name}")

                    # KPI Metrics Row
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Monitored", len(st.session_state.parts))
                    m2.metric("Identified", len(present))
                    m3.metric("Missing", len(missing))
                    m4.metric("Verdict", "PASS" if not missing else "FAIL")

                    col_res, col_img = st.columns([1, 2.2], gap="large")

                    with col_res:
                        if not missing:
                            st.markdown(
                                "<div class='status-card-ok'>"
                                "<h2 style='margin:0;'>OK</h2>"
                                "<p style='margin:0.25rem 0 0 0;'>All labeled components are present.</p>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div class='status-card-fail'>"
                                f"<h3 style='margin:0;'>Missing {len(missing)} Part(s)</h3>"
                                f"<p style='margin:0.25rem 0 0 0;'>Highlighted in red on the board.</p>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            pills_html = "".join(
                                f"<span class='part-pill pill-missing'>✕ {name}</span>"
                                for name in missing
                            )
                            st.markdown(pills_html, unsafe_allow_html=True)

                        if present:
                            with st.expander(f"Identified Parts ({len(present)})", expanded=False):
                                ok_html = "".join(
                                    f"<span class='part-pill pill-ok'>✓ {name} ({conf:.2f})</span>"
                                    for name, conf in present
                                )
                                st.markdown(ok_html, unsafe_allow_html=True)

                    with col_img:
                        st.image(annotated_img, use_container_width=True)
                        st.caption("🟢 Green = Identified by YOLO  |  🔴 Red = Missing")

            batch_bar.progress(100, text=f"Completed inspection of {len(test_files)} board(s).")
            time.sleep(0.4)
            batch_bar.empty()