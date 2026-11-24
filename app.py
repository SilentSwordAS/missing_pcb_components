import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from streamlit_drawable_canvas import st_canvas

# --- Page Config ---
st.set_page_config(page_title="Offline PCB Inspector", layout="wide")
st.title("PCB Missing Part Detector")

# Distinct high-contrast palette for bounding boxes
BOX_COLORS = [
    "#00C853",  # Green
    "#2979FF",  # Blue
    "#FF9100",  # Orange
    "#D500F9",  # Purple
    "#00B8D4",  # Cyan
    "#FFD600",  # Yellow
    "#FF4081",  # Pink
    "#FF3D00",  # Deep Orange
]

# --- Session State Initialization ---
if "gt_image" not in st.session_state:
    st.session_state.gt_image = None
if "parts" not in st.session_state:
    st.session_state.parts = []  # List of dicts: {"name": str, "box": (x, y, w, h), "color": str}


# --- Helper Functions ---
def hex_to_rgb(hex_color: str) -> tuple:
    """Converts #RRGGBB hex string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def draw_label_box(img: np.ndarray, text: str, x: int, y: int, bg_color: tuple):
    """Draws readable text with a solid background pill above a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    top = max(0, y - th - baseline - 6)
    cv2.rectangle(img, (x, top), (x + tw + 8, top + th + baseline + 6), bg_color, -1)
    cv2.putText(
        img,
        text,
        (x + 4, top + th + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def render_numbered_gt_preview(gt_img: np.ndarray, parts: list) -> np.ndarray:
    """Draws numbered, color-coded boxes on a copy of the Ground Truth image."""
    preview = gt_img.copy()
    for idx, part in enumerate(parts):
        x, y, w, h = part["box"]
        rgb = hex_to_rgb(part["color"])
        cv2.rectangle(preview, (x, y), (x + w, y + h), rgb, 3)
        draw_label_box(preview, f"#{idx + 1}: {part['name']}", x, y, rgb)
    return preview


# --- Core Offline Vision Functions ---
def align_image(test_img: np.ndarray, gt_img: np.ndarray) -> np.ndarray:
    """Aligns the test PCB image to the Ground Truth using ORB feature matching."""
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


def inspect_board(
    gt_img: np.ndarray, test_img: np.ndarray, parts: list, threshold: float, use_alignment: bool
):
    """Compares each named ROI using SSIM and highlights both present and missing parts."""
    processed_test = (
        align_image(test_img, gt_img)
        if use_alignment
        else cv2.resize(test_img, (gt_img.shape[1], gt_img.shape[0]))
    )

    annotated = processed_test.copy()
    missing_parts = []
    present_parts = []

    for part in parts:
        x, y, w, h = part["box"]
        x, y = max(0, x), max(0, y)
        roi_gt = gt_img[y : y + h, x : x + w]
        roi_test = processed_test[y : y + h, x : x + w]

        if roi_gt.size == 0 or roi_test.size == 0:
            continue

        gray_gt = cv2.cvtColor(roi_gt, cv2.COLOR_RGB2GRAY)
        gray_test = cv2.cvtColor(roi_test, cv2.COLOR_RGB2GRAY)

        min_dim = min(gray_gt.shape[0], gray_gt.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        if win_size < 3:
            continue

        score = ssim(gray_gt, gray_test, win_size=win_size)

        if score < threshold:
            missing_parts.append((part["name"], score))
            # Red highlight for missing part
            color_missing = (220, 53, 69)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color_missing, 3)
            draw_label_box(annotated, f"MISSING: {part['name']}", x, y, color_missing)
        else:
            present_parts.append((part["name"], score))
            # Green highlight for identified/present part
            color_ok = (40, 167, 69)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color_ok, 2)
            draw_label_box(annotated, f"{part['name']} ({score:.2f})", x, y, color_ok)

    return missing_parts, present_parts, annotated


# --- Minimalist UI Tabs ---
tab_setup, tab_inspect = st.tabs(["1. Ground Truth Setup", "2. Inspect Boards"])

# === TAB 1: GROUND TRUTH & LABELING ===
with tab_setup:
    gt_file = st.file_uploader("Upload Good Board (Ground Truth)", type=["png", "jpg", "jpeg"])

    if gt_file:
        gt_pil = Image.open(gt_file).convert("RGB")
        st.session_state.gt_image = np.array(gt_pil)
        orig_w, orig_h = gt_pil.size

        canvas_w = 700
        scale = canvas_w / orig_w
        canvas_h = int(orig_h * scale)

        # Cycle stroke color automatically for each new box drawn
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

                    # Preserve the box's drawn stroke color
                    box_color = obj.get("stroke", BOX_COLORS[idx % len(BOX_COLORS)])
                    default_name = (
                        st.session_state.parts[idx]["name"]
                        if idx < len(st.session_state.parts)
                        else f"Part_{idx + 1}"
                    )

                    # Color-coded header matching the box on the diagram
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

        # Live numbered reference diagram so users can verify box numbers directly on the board
        if st.session_state.parts:
            with st.expander("View Numbered Reference Map", expanded=False):
                numbered_preview = render_numbered_gt_preview(
                    st.session_state.gt_image, st.session_state.parts
                )
                st.image(numbered_preview, use_container_width=True)

# === TAB 2: BATCH INSPECTION ===
with tab_inspect:
    if st.session_state.gt_image is None or not st.session_state.parts:
        st.warning("Upload a Ground Truth board and label at least one part in Tab 1 first.")
    else:
        with st.sidebar:
            st.header("Detection Settings")
            threshold = st.slider(
                "Similarity Threshold (SSIM)",
                min_value=0.30,
                max_value=0.95,
                value=0.70,
                step=0.05,
                help="Lower = more tolerant to lighting changes. Higher = stricter detection.",
            )
            use_alignment = st.checkbox("Auto-align boards (ORB)", value=True)

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

            missing, present, annotated_img = inspect_board(
                st.session_state.gt_image,
                test_np,
                st.session_state.parts,
                threshold,
                use_alignment,
            )

            col_res, col_img = st.columns([1, 2])

            with col_res:
                if not missing:
                    st.markdown(
                        "<h2 style='color: #28a745; margin-top: 0;'>OK</h2>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"All {len(present)} labeled parts identified.")
                else:
                    st.markdown(
                        f"<h3 style='color: #dc3545; margin-top: 0;'>Missing ({len(missing)})</h3>",
                        unsafe_allow_html=True,
                    )
                    for name, score in missing:
                        st.write(f"- **{name}** (similarity: `{score:.2f}`)")

                if present:
                    with st.expander(f"Identified Parts ({len(present)})", expanded=False):
                        for name, score in present:
                            st.write(f"- {name} (`{score:.2f}`)")

            with col_img:
                st.image(annotated_img, use_container_width=True)
                st.caption("Green = Identified / Present  |  Red = Missing")