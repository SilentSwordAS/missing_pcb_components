import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import yaml
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO

# --- Page Config ---
st.set_page_config(page_title="Offline YOLO PCB Inspector", layout="wide")
st.title("PCB Missing Part Detector (Offline YOLO)")

BOX_COLORS = [
    "#00C853", "#2979FF", "#FF9100", "#D500F9",
    "#00B8D4", "#FFD600", "#FF4081", "#FF3D00",
]

# --- Session State Initialization ---
if "gt_image" not in st.session_state:
    st.session_state.gt_image = None
if "parts" not in st.session_state:
    st.session_state.parts = []  # [{"name": str, "box": (x, y, w, h), "color": str}]
if "trained_model_path" not in st.session_state:
    st.session_state.trained_model_path = None
if "class_names" not in st.session_state:
    st.session_state.class_names = {}


# --- Drawing Helpers ---
def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def draw_label_box(img: np.ndarray, text: str, x: int, y: int, bg_color: tuple):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    top = max(0, y - th - baseline - 6)
    cv2.rectangle(img, (x, top), (x + tw + 8, top + th + baseline + 6), bg_color, -1)
    cv2.putText(
        img, text, (x + 4, top + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def render_numbered_gt_preview(gt_img: np.ndarray, parts: list) -> np.ndarray:
    preview = gt_img.copy()
    for idx, part in enumerate(parts):
        x, y, w, h = part["box"]
        rgb = hex_to_rgb(part["color"])
        cv2.rectangle(preview, (x, y), (x + w, y + h), rgb, 3)
        draw_label_box(preview, f"#{idx + 1}: {part['name']}", x, y, rgb)
    return preview


# --- Image Alignment (used to project missing GT boxes accurately) ---
def align_image(test_img: np.ndarray, gt_img: np.ndarray) -> np.ndarray:
    gray_gt = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
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


# --- Offline One-Shot YOLO Training ---
def build_augmented_dataset(gt_img: np.ndarray, parts: list, work_dir: Path, num_samples: int = 24):
    """Generates a local augmented YOLO dataset from the single Ground Truth board."""
    img_dir = work_dir / "images" / "train"
    lbl_dir = work_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    h_img, w_img = gt_img.shape[:2]
    gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)

    # Map each labeled part to its own class ID
    class_names = {idx: part["name"] for idx, part in enumerate(parts)}

    rng = np.random.default_rng(42)
    for i in range(num_samples):
        aug = gt_bgr.copy().astype(np.float32)

        if i > 0:
            # Brightness & contrast jitter
            alpha = rng.uniform(0.85, 1.15)
            beta = rng.uniform(-18, 18)
            aug = np.clip(aug * alpha + beta, 0, 255)

            # Slight Gaussian noise
            noise = rng.normal(0, 4.0, aug.shape)
            aug = np.clip(aug + noise, 0, 255)

        aug = aug.astype(np.uint8)

        # Slight translation to make YOLO robust to minor board shifts
        dx = int(rng.integers(-12, 13)) if i > 0 else 0
        dy = int(rng.integers(-12, 13)) if i > 0 else 0
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aug = cv2.warpAffine(aug, M, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)

        cv2.imwrite(str(img_dir / f"gt_{i:03d}.jpg"), aug)

        # Write YOLO normalized labels: <class_id> <x_center> <y_center> <w> <h>
        label_lines = []
        for cls_id, part in enumerate(parts):
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

    data_yaml = {
        "path": str(work_dir.resolve()),
        "train": "images/train",
        "val": "images/train",
        "names": class_names,
    }
    yaml_path = work_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml))
    return yaml_path, class_names


def train_offline_yolo(gt_img: np.ndarray, parts: list, base_weights: str, epochs: int):
    """Fine-tunes YOLO26n or YOLO11n locally on the augmented Ground Truth board."""
    work_dir = Path(tempfile.mkdtemp(prefix="pcb_yolo_"))
    yaml_path, class_names = build_augmented_dataset(gt_img, parts, work_dir)

    model = YOLO(base_weights)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=8,
        workers=0,
        cache=True,
        project=str(work_dir / "runs"),
        name="pcb_detector",
        exist_ok=True,
        verbose=False,
        plots=False,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    return str(best_weights), class_names


# --- Offline YOLO Inspection ---
def inspect_board_yolo(
    model: YOLO,
    gt_img: np.ndarray,
    test_img: np.ndarray,
    parts: list,
    conf_threshold: float,
    use_alignment: bool,
):
    """Runs YOLO inference on the test board and flags any undetected GT parts as missing."""
    processed_test = (
        align_image(test_img, gt_img)
        if use_alignment
        else cv2.resize(test_img, (gt_img.shape[1], gt_img.shape[0]))
    )

    # Run offline YOLO inference
    preds = model.predict(processed_test, conf=conf_threshold, verbose=False)[0]

    # Track highest-confidence detection per class ID
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
            draw_label_box(annotated, f"{part['name']} ({conf:.2f})", x1, y1, color_ok)
        else:
            missing_parts.append(part["name"])
            x, y, w, h = part["box"]
            color_missing = (220, 53, 69)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color_missing, 3)
            draw_label_box(annotated, f"MISSING: {part['name']}", x, y, color_missing)

    return missing_parts, present_parts, annotated


# --- Minimalist UI Tabs ---
tab_setup, tab_inspect = st.tabs(["1. Ground Truth & Train", "2. Inspect Boards"])

# === TAB 1: GROUND TRUTH, LABELING & LOCAL TRAINING ===
with tab_setup:
    gt_file = st.file_uploader("Upload Good Board (Ground Truth)", type=["png", "jpg", "jpeg"])

    if gt_file:
        gt_pil = Image.open(gt_file).convert("RGB")
        st.session_state.gt_image = np.array(gt_pil)
        orig_w, orig_h = gt_pil.size

        canvas_w = 700
        scale = canvas_w / orig_w
        canvas_h = int(orig_h * scale)
        next_color = BOX_COLORS[len(st.session_state.parts) % len(BOX_COLORS)]

        col_canvas, col_names = st.columns([2, 1])

        with col_canvas:
            st.caption("Draw a rectangle around each component. Each new box gets a distinct color.")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.08)",
                stroke_width=3,
                stroke_color=next_color,
                background_image=gt_pil,
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="rect",
                key="gt_canvas",
            )

        with col_names:
            st.subheader("Named Parts")
            objects = (
                canvas_result.json_data["objects"]
                if canvas_result.json_data and "objects" in canvas_result.json_data
                else []
            )

            if not objects:
                st.info("Draw boxes on the board to name parts.")
            else:
                updated_parts = []
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

                    st.markdown(
                        f"<span style='color:{box_color}; font-size:1.1em;'>&#9632;</span> "
                        f"**Box #{idx + 1}**",
                        unsafe_allow_html=True,
                    )
                    part_name = st.text_input(
                        f"Name for Box #{idx + 1}",
                        value=default_name,
                        key=f"name_input_{idx}",
                        label_visibility="collapsed",
                    )
                    updated_parts.append(
                        {"name": part_name, "box": (x, y, w, h), "color": box_color}
                    )

                st.session_state.parts = updated_parts

        if st.session_state.parts:
            with st.expander("View Numbered Reference Map", expanded=False):
                st.image(
                    render_numbered_gt_preview(st.session_state.gt_image, st.session_state.parts),
                    use_container_width=True,
                )

            st.divider()
            col_m, col_e, col_btn = st.columns([1, 1, 1])
            with col_m:
                base_weights = st.selectbox("Local Base Model", ["yolo26m.pt"])
            with col_e:
                epochs = st.slider("Training Epochs", min_value=15, max_value=60, value=30, step=5)
            with col_btn:
                st.write("")
                st.write("")
                train_clicked = st.button("Train Offline YOLO Model", type="primary", use_container_width=True)

            if train_clicked:
                if not os.path.exists(base_weights):
                    st.error(
                        f"Local weights `{base_weights}` not found in the current directory. "
                        "Place the `.pt` file alongside `app.py` for offline use."
                    )
                else:
                    with st.spinner(f"Training {base_weights} locally on {len(st.session_state.parts)} parts..."):
                        best_pt, cls_names = train_offline_yolo(
                            st.session_state.gt_image, st.session_state.parts, base_weights, epochs
                        )
                        st.session_state.trained_model_path = best_pt
                        st.session_state.class_names = cls_names
                    st.success("Model trained offline! Switch to Tab 2 to inspect boards.")

# === TAB 2: BATCH INSPECTION ===
with tab_inspect:
    if not st.session_state.trained_model_path:
        st.warning("Upload a Ground Truth board, label parts, and train the YOLO model in Tab 1 first.")
    else:
        with st.sidebar:
            st.header("YOLO Settings")
            conf_threshold = st.slider(
                "Detection Confidence",
                min_value=0.15,
                max_value=0.90,
                value=0.45,
                step=0.05,
                help="Parts detected below this confidence are flagged as missing.",
            )
            use_alignment = st.checkbox("Auto-align boards (ORB)", value=True)

        model = YOLO(st.session_state.trained_model_path)

        test_files = st.file_uploader(
            "Upload Board Photos to Inspect",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        for test_file in test_files:
            st.divider()
            st.subheader(test_file.name)

            test_pil = Image.open(test_file).convert("RGB")
            test_np = np.array(test_pil)

            missing, present, annotated_img = inspect_board_yolo(
                model,
                st.session_state.gt_image,
                test_np,
                st.session_state.parts,
                conf_threshold,
                use_alignment,
            )

            col_res, col_img = st.columns([1, 2])

            with col_res:
                if not missing:
                    st.markdown(
                        "<h2 style='color: #28a745; margin-top: 0;'>OK</h2>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"All {len(present)} labeled parts detected.")
                else:
                    st.markdown(
                        f"<h3 style='color: #dc3545; margin-top: 0;'>Missing ({len(missing)})</h3>",
                        unsafe_allow_html=True,
                    )
                    for name in missing:
                        st.write(f"- **{name}**")

                if present:
                    with st.expander(f"Identified Parts ({len(present)})", expanded=False):
                        for name, conf in present:
                            st.write(f"- {name} (`{conf:.2f}`)")

            with col_img:
                st.image(annotated_img, use_container_width=True)
                st.caption("Green = Identified by YOLO  |  Red = Missing")