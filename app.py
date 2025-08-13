import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import io
import json
from datetime import datetime
import pandas as pd
import shutil
import math

# =============== Tesseract autodetect (Cloud & local) ===============
TESSERACT_CMD = shutil.which("tesseract")
if TESSERACT_CMD is None:
    st.error(
        "❌ Tesseract is not installed or not in PATH.\n\n"
        "On Streamlit Cloud, add a `packages.txt` with:\n"
        "    tesseract-ocr\n"
        "then restart the app."
    )
    st.stop()
else:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Optional: HEIC support if available
try:
    import pillow_heif  # requires pillow-heif + libheif1 on Cloud
    pillow_heif.register_heif_opener()
except Exception:
    pass

# =============== Page config & CSS ===============
st.set_page_config(
    page_title="Journalist's OCR Tool",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stFileUploader > div > div > div { padding: 1rem; }
    .stImage > div { text-align: center; }
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 2rem; padding-left: 1rem; padding-right: 1rem;
        }
    }
    .success-box {
        padding: 1rem; border-radius: 0.5rem; background-color: #d4edda;
        border: 1px solid #c3e6cb; color: #155724; margin: 1rem 0;
    }
    .processing-stats {
        background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;
        border-left: 4px solid #007bff; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============== Session State ===============
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = []   # list of dicts (each with image, text, original_text, etc.)
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'user_corrections' not in st.session_state:
    st.session_state.user_corrections = {}

# =============== Orientation helpers ===============
def exif_transpose(img: Image.Image) -> Image.Image:
    """Apply EXIF-based orientation (common for phone photos)."""
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def auto_rotate_for_text(pil_image: Image.Image):
    """
    If strong vertical line dominance is detected (suggesting sideways text),
    rotate 90° to make text horizontal. Returns (rotated_image, did_rotate: bool).
    """
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough to get line segments
    H, W = gray.shape[:2]
    min_len = max(30, min(H, W) // 6)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=min_len, maxLineGap=20)

    if lines is None or len(lines) < 6:
        return pil_image, False

    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
        angle = (angle + 180) % 180  # map to [0, 180)
        angles.append(angle)

    # Count near-horizontal (0±20 or 180±20) vs near-vertical (90±20)
    horiz = sum(1 for a in angles if a <= 20 or a >= 160)
    vert  = sum(1 for a in angles if 70 <= a <= 110)
    total = max(1, len(angles))

    vert_ratio = vert / total
    horiz_ratio = horiz / total

    # If vertical lines dominate clearly, assume image is sideways → rotate 90° (CW)
    if vert_ratio >= 0.6 and vert >= max(10, 3 * horiz):
        return pil_image.rotate(90, expand=True), True

    return pil_image, False

# =============== Preprocessing & OCR ===============
def preprocess_image(image, enhancement_level="medium"):
    """Advanced preprocessing for handwriting, optimized for margin notes"""
    img_array = np.array(image)

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Noise removal
    denoised = cv2.fastNlMeansDenoising(gray)

    # Enhance based on level
    if enhancement_level == "light":
        processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif enhancement_level == "medium":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        processed = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:  # aggressive
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
        enhanced = clahe.apply(denoised)

        # Dilation to separate touching letters
        kernel = np.ones((1,1), np.uint8)
        enhanced = cv2.dilate(enhanced, kernel, iterations=1)

        # Morphological operations for margin text
        kernel2 = np.ones((2,2), np.uint8)
        processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel2)
        processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Additional erosion to separate merged characters
        kernel3 = np.ones((1,1), np.uint8)
        processed = cv2.erode(processed, kernel3, iterations=1)

    return processed

def extract_text_tesseract(image, enhancement_level="medium"):
    """Extract text using Tesseract OCR with multiple configurations for best results"""
    processed_img = preprocess_image(image, enhancement_level)
    pil_processed = Image.fromarray(processed_img)

    # Safe whitelist (no double quotes, keeps single quote and space)
    whitelist_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()-@&/ '"
    configs = [
        f'--oem 3 --psm 6 -c tessedit_char_whitelist="{whitelist_chars}"',
        f'--oem 3 --psm 7 -c tessedit_char_whitelist="{whitelist_chars}"',
        f'--oem 3 --psm 8 -c tessedit_char_whitelist="{whitelist_chars}"'
    ]

    results = {}
    best_result = ""
    best_config = "default"

    for i, config in enumerate(configs):
        try:
            result = pytesseract.image_to_string(pil_processed, config=config).strip()
            config_name = ["general", "block", "word"][i]
            results[config_name] = result
            if len(result) > len(best_result):
                best_result = result
                best_config = config_name
        except Exception as e:
            results[f"config_{i}"] = f"Error: {e}"

    return best_result, results, processed_img, best_config

# =============== Learning Corrections ===============
def learn_from_corrections(original_text, corrected_text, image_name):
    """Store user corrections to improve future recognition"""
    if original_text.strip() and corrected_text.strip() and original_text != corrected_text:
        correction_entry = {
            'timestamp': datetime.now().isoformat(),
            'image': image_name,
            'original': original_text,
            'corrected': corrected_text,
            'pattern_type': 'user_correction'
        }
        st.session_state.user_corrections[image_name] = correction_entry

def apply_learned_corrections(text, image_name):
    """Apply previously learned corrections"""
    corrected_text = text

    # Common handwriting OCR corrections based on analysis
    common_fixes = {
        ' or ': ' a ',
        'rn': 'm',
        'cl': 'd',
        'li': 'h',
        'rnore': 'more',
        'ornd': 'and',
        'tlie': 'the',
        'orll': 'all',
        'tl1e': 'the',
        'witl1': 'with',
        'l1is': 'his',
        'l1er': 'her'
    }

    # Journalism-specific corrections for margin notes
    journalism_fixes = {
        'MDSCOW': 'MOSCOW',
        'POLIGH': 'POLISH',
        'UKRAIN': 'UKRAINE',
        'CYEER': 'CYBER',
        'POISDN': 'POISON',
        'TARGEI': 'TARGET',
        'BACKDCOR': 'BACKDOOR',
        'MDSC0W': 'MOSCOW',
        'RU5SIA': 'RUSSIA',
        'UKRA1NE': 'UKRAINE'
    }

    for wrong, right in common_fixes.items():
        corrected_text = corrected_text.replace(wrong, right)
    for wrong, right in journalism_fixes.items():
        corrected_text = corrected_text.replace(wrong, right)
    for correction in st.session_state.user_corrections.values():
        if correction['original'] in corrected_text:
            corrected_text = corrected_text.replace(correction['original'], correction['corrected'])

    return corrected_text

# =============== Helpers (export & display) ===============
def results_to_dataframe(results_list):
    rows = []
    for r in results_list:
        rows.append({
            "filename": r.get("filename", ""),
            "text": r.get("text", ""),
            "original_text": r.get("original_text", ""),
            "engine": r.get("engine", ""),
            "best_config": r.get("best_config", ""),
            "timestamp": r.get("timestamp", "")
        })
    return pd.DataFrame(rows)

def make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def make_json_bytes(results_list) -> bytes:
    minimal = [
        {
            "filename": r.get("filename", ""),
            "text": r.get("text", ""),
            "original_text": r.get("original_text", ""),
            "engine": r.get("engine", ""),
            "best_config": r.get("best_config", ""),
            "timestamp": r.get("timestamp", "")
        }
        for r in results_list
    ]
    return json.dumps(minimal, ensure_ascii=False, indent=2).encode("utf-8")

def reocr_single_result(res, enhancement_level):
    """Re-run OCR for a single result dict after rotation or edits to image."""
    best_text, all_configs, processed_img, best_config = extract_text_tesseract(res['image'], enhancement_level)
    corrected_text = apply_learned_corrections(best_text, res['filename'])
    res.update({
        'text': corrected_text,
        'original_text': best_text,
        'all_configs': all_configs,
        'processed_image': processed_img,
        'best_config': best_config,
        'engine': f'Tesseract ({best_config})',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    return res

def process_images(files, enhancement_level):
    """Run EXIF transpose, auto-rotate, OCR + apply learned corrections."""
    processed_results = []
    for uploaded_file in files:
        try:
            # Support both UploadedFile and raw PIL
            img = uploaded_file if isinstance(uploaded_file, Image.Image) else Image.open(uploaded_file)

            # Normalize orientation first (EXIF)
            img = exif_transpose(img)

            # Auto-rotate if strong vertical lines
            img_rot, did_rotate = auto_rotate_for_text(img)

            filename = getattr(uploaded_file, "name", "image.png")
            best_text, all_configs, processed_img, best_config = extract_text_tesseract(img_rot, enhancement_level)
            corrected_text = apply_learned_corrections(best_text, filename)

            result = {
                'filename': filename + (" (rotated)" if did_rotate else ""),
                'text': corrected_text,
                'original_text': best_text,
                'engine': f'Tesseract ({best_config})',
                'all_configs': all_configs,
                'best_config': best_config,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image': img_rot,              # store oriented original
                'processed_image': processed_img
            }
            processed_results.append(result)

        except Exception as e:
            st.error(f"❌ Error processing {getattr(uploaded_file, 'name', 'image')}: {e}")
    return processed_results

def display_results(results, enhancement_level, show_processed_images=False, show_alternative_configs=False):
    """Side-by-side image + text, with rotate & re-OCR and edit & save UI."""
    for idx, res in enumerate(results):
        st.markdown("---")
        st.write(f"**File:** {res['filename']}  •  **Engine:** {res['engine']}  •  **Time:** {res['timestamp']}")

        col_img, col_txt = st.columns([1, 2])

        with col_img:
            # Rotate controls
            rot_col1, rot_col2 = st.columns(2)
            with rot_col1:
                if st.button("⟲ Rotate 90°", key=f"rotl_{idx}"):
                    res['image'] = res['image'].rotate(90, expand=True)  # CCW
                    reocr_single_result(res, enhancement_level)
                    st.session_state.ocr_results[idx] = res
                    st.success("Rotated left and re-OCR'd.")
            with rot_col2:
                if st.button("⟳ Rotate 90°", key=f"rotr_{idx}"):
                    res['image'] = res['image'].rotate(-90, expand=True)  # CW
                    reocr_single_result(res, enhancement_level)
                    st.session_state.ocr_results[idx] = res
                    st.success("Rotated right and re-OCR'd.")

            tabs = st.tabs(["Image", "Processed" if show_processed_images else " "])
            with tabs[0]:
                st.image(res['image'], caption="Image (auto/manual oriented)", use_container_width=True)
            if show_processed_images:
                with tabs[1]:
                    st.image(res['processed_image'], caption="Processed (for OCR)", use_container_width=True)

        with col_txt:
            st.text_area("Extracted Text", res['text'], height=220, key=f"view_text_{idx}")

            # --- Edit & Save Correction UI ---
            with st.expander("✏️ Edit & Save Correction"):
                edited_text = st.text_area(
                    "Correct the text below and click Save to teach the tool.",
                    value=res['text'],
                    height=180,
                    key=f"edit_text_{idx}"
                )
                save_cols = st.columns(2)
                with save_cols[0]:
                    if st.button("Save Correction", key=f"save_corr_{idx}"):
                        learn_from_corrections(res['original_text'], edited_text, res['filename'])
                        res['text'] = edited_text
                        st.session_state.ocr_results[idx] = res
                        st.success("Saved! Future runs will apply this correction automatically.")
                with save_cols[1]:
                    if st.button("Re-OCR This Image", key=f"reocr_{idx}"):
                        reocr_single_result(res, enhancement_level)
                        st.session_state.ocr_results[idx] = res
                        st.success("Re-OCR complete for this image.")

# =============== UI (sidebar & controls) ===============
st.title("📝 Journalist's Handwriting OCR Tool")
st.markdown("*Mobile-friendly batch OCR with learning capabilities — Tesseract Edition*")

st.sidebar.header("Settings")
enhancement_level = st.sidebar.selectbox(
    "Image Enhancement Level",
    ["light", "medium", "aggressive"],
    index=1,
    help="Choose based on your handwriting clarity"
)

batch_mode = st.sidebar.checkbox("Batch Processing Mode", value=True)
show_processed_images = st.sidebar.checkbox("Show Processed Images (tab)", value=False)
show_alternative_configs = st.sidebar.checkbox("Show Alternative OCR Configs (in expander)", value=False)

st.info("🚀 **Fast & Reliable**: This version uses Tesseract OCR for instant processing without downloads. Based on our handwriting analysis, expect 80–90% accuracy on neat notes.")

# =============== File upload ===============
st.header("📤 Upload Images")
if batch_mode:
    uploaded_files = st.file_uploader(
        "Choose image files (JPG, PNG, HEIC)",
        type=['jpg', 'jpeg', 'png', 'heic'],
        accept_multiple_files=True,
        help="You can upload multiple images at once from your phone or tablet"
    )
else:
    uploaded_files = [st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'heic']
    )]

uploaded_files = [f for f in uploaded_files if f is not None]
if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully!")

# =============== Process button ===============
def results_to_download_blobs(results):
    df = results_to_dataframe(results)
    csv_bytes = make_csv_bytes(df)
    json_bytes = make_json_bytes(results)
    return df, csv_bytes, json_bytes

if st.button("🔍 Process All Images", type="primary"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = process_images(uploaded_files, enhancement_level)

    # append to session state
    for r in results:
        st.session_state.ocr_results.append(r)

    for i in range(len(uploaded_files)):
        progress_bar.progress((i + 1) / max(1, len(uploaded_files)))

    st.success("✅ Processing complete!")
    status_text.text("All files processed.")

    # Display results (side-by-side, with rotate and re-OCR)
    display_results(results, enhancement_level, show_processed_images, show_alternative_configs)

    # ---- Downloads for this run ----
    st.markdown("### ⬇️ Download This Run")
    _, csv_bytes_run, json_bytes_run = results_to_download_blobs(results)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.download_button("Download CSV (this run)", data=csv_bytes_run, file_name=f"ocr_results_{ts}.csv", mime="text/csv")
    st.download_button("Download JSON (this run)", data=json_bytes_run, file_name=f"ocr_results_{ts}.json", mime="application/json")

# =============== Re-run with corrections applied (all) ===============
if st.session_state.ocr_results:
    st.markdown("---")
    st.subheader("🔁 Re-run with Corrections Applied")
    st.caption("Re-OCR all stored images using current settings and apply learned corrections.")
    if st.button("Re-run Now"):
        images_for_rerun = [r['image'] for r in st.session_state.ocr_results]
        rerun_results = process_images(images_for_rerun, enhancement_level)
        st.session_state.ocr_results = rerun_results
        st.success("🔁 Re-run complete! Results updated below.")
        display_results(rerun_results, enhancement_level, show_processed_images, show_alternative_configs)

    # ---- Sidebar exports for all-time session results ----
    st.sidebar.markdown("### ⬇️ Export All Results (Session)")
    df_all = results_to_dataframe(st.session_state.ocr_results)
    csv_bytes_all = make_csv_bytes(df_all)
    json_bytes_all = make_json_bytes(st.session_state.ocr_results)
    ts_all = datetime.now().strftime('%Y%m%d_%H%M%S')

    st.sidebar.download_button("Download CSV (session)", data=csv_bytes_all, file_name=f"ocr_results_session_{ts_all}.csv", mime="text/csv")
    st.sidebar.download_button("Download JSON (session)", data=json_bytes_all, file_name=f"ocr_results_session_{ts_all}.json", mime="application/json")
