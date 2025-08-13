import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import pytesseract
import io
import json
from datetime import datetime
import pandas as pd
import shutil
import math
import platform
import hashlib
import uuid

# ===================== HARD INIT (first thing) =====================
if 'session_uid' not in st.session_state:
    st.session_state['session_uid'] = str(uuid.uuid4())
if 'user_corrections' not in st.session_state:
    st.session_state['user_corrections'] = {}
if 'ocr_results' not in st.session_state:
    st.session_state['ocr_results'] = []
if 'processing_history' not in st.session_state:
    st.session_state['processing_history'] = []

# ===================== Tesseract autodetect =====================
st.set_page_config(page_title="Journalist's OCR Tool", page_icon="📝", layout="wide", initial_sidebar_state="expanded")
TESSERACT_CMD = shutil.which("tesseract")
if TESSERACT_CMD is None:
    st.error("❌ Tesseract is not installed or not in PATH.\n\nOn Streamlit Cloud, add `packages.txt` with:\n    tesseract-ocr")
    st.stop()
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Optional HEIC
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ===================== Sidebar: settings & debug =====================
st.sidebar.header("Settings")
enhancement_level = st.sidebar.selectbox("Image Enhancement Level", ["light", "medium", "aggressive"], index=1)
ocr_lang = st.sidebar.selectbox("OCR Language", ["eng", "deu", "fra", "spa", "ita", "por", "nld"], index=0)
batch_mode = st.sidebar.checkbox("Batch Processing Mode", value=True)
show_processed_images = st.sidebar.checkbox("Show Processed Images (tab)", value=False)
show_alternative_configs = st.sidebar.checkbox("Show Alternative OCR Configs (in expander)", value=False)

st.sidebar.markdown("### 🐞 Debug")
debug_mode = st.sidebar.toggle("Enable Debug Mode", value=False)
if st.sidebar.button("🔄 Refresh now"):
    st.rerun()
st.sidebar.caption(f"Session: `{st.session_state['session_uid']}`")

if debug_mode:
    st.sidebar.write("Python:", platform.python_version())
    st.sidebar.write("OS:", platform.platform())
    st.sidebar.write("Tesseract path:", pytesseract.pytesseract.tesseract_cmd)
    try:
        st.sidebar.write("Tesseract version:", str(pytesseract.get_tesseract_version()))
    except Exception as e:
        st.sidebar.error(f"Tesseract not callable: {e}")
    try:
        test_img = Image.new("RGB", (600, 120), "white")
        d = ImageDraw.Draw(test_img)
        d.text((10, 40), "Hello OCR 123!", fill="black")
        test_text = pytesseract.image_to_string(test_img, config="--oem 3 --psm 6 -l eng")
        st.sidebar.write("Self-test OCR:", test_text.strip())
        st.sidebar.image(test_img, caption="Self-test image", use_container_width=True)
    except Exception as e:
        st.sidebar.error(f"OCR self-test failed: {e}")

if st.sidebar.checkbox("Show raw corrections (debug)", value=False, key="show_raw_corrs"):
    st.sidebar.json(st.session_state.get('user_corrections', {}))

def _override_keys_only():
    return [k for k, v in st.session_state['user_corrections'].items()
            if isinstance(v, dict) and 'corrected' in v]
st.sidebar.caption(f"✅ Saved corrections: {len(_override_keys_only())}")

# ===================== Utils: keys, orientation, OCR =====================
def image_content_hash(pil_image: Image.Image) -> str:
    with io.BytesIO() as buf:
        pil_image.save(buf, format="PNG")
        data = buf.getvalue()
    return hashlib.sha1(data).hexdigest()[:10]

def make_storage_key(filename: str, pil_image: Image.Image) -> str:
    base = filename if (filename and filename.strip()) else "UNNAMED"
    h = image_content_hash(pil_image)
    return f"{base}__{h}"

def exif_transpose(img: Image.Image) -> Image.Image:
    try: return ImageOps.exif_transpose(img)
    except Exception: return img

def auto_rotate_for_text(pil_image: Image.Image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    H, W = gray.shape[:2]
    min_len = max(30, min(H, W) // 6)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min_len, maxLineGap=20)
    if lines is None or len(lines) < 6:
        return pil_image, False
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
        angle = (angle + 180) % 180
        angles.append(angle)
    horiz = sum(1 for a in angles if a <= 20 or a >= 160)
    vert  = sum(1 for a in angles if 70 <= a <= 110)
    total = max(1, len(angles))
    if (vert / total) >= 0.6 and vert >= max(10, 3 * horiz):
        return pil_image.rotate(90, expand=True), True
    return pil_image, False

def preprocess_image(image, enhancement_level="medium"):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    denoised = cv2.fastNlMeansDenoising(gray)
    if enhancement_level == "light":
        processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif enhancement_level == "medium":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        processed = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
        enhanced = clahe.apply(denoised)
        kernel = np.ones((1,1), np.uint8)
        enhanced = cv2.dilate(enhanced, kernel, iterations=1)
        kernel2 = np.ones((2,2), np.uint8)
        processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel2)
        processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel3 = np.ones((1,1), np.uint8)
        processed = cv2.erode(processed, kernel3, iterations=1)
    return processed

def extract_text_tesseract(image, enhancement_level="medium", lang="eng"):
    processed_img = preprocess_image(image, enhancement_level)
    pil_processed = Image.fromarray(processed_img)
    whitelist_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()-@&/ '"
    configs = [
        f'--oem 3 --psm 6 -l {lang} -c tessedit_char_whitelist="{whitelist_chars}"',
        f'--oem 3 --psm 7 -l {lang} -c tessedit_char_whitelist="{whitelist_chars}"',
        f'--oem 3 --psm 8 -l {lang} -c tessedit_char_whitelist="{whitelist_chars}"'
    ]
    results = {}
    best_result, best_config = "", "default"
    for i, config in enumerate(configs):
        try:
            result = pytesseract.image_to_string(pil_processed, config=config).strip()
            cfg_name = ["general", "block", "word"][i]
            results[cfg_name] = result if result else ""
            if len(result) > len(best_result):
                best_result, best_config = result, cfg_name
        except Exception as e:
            results[f"config_{i}"] = f"Error: {e}"
    return best_result, results, processed_img, best_config

# ===================== Corrections =====================
def learn_from_corrections(original_text, corrected_text, image_key):
    if 'user_corrections' not in st.session_state:
        st.session_state['user_corrections'] = {}
    image_key = image_key or "UNNAMED__" + datetime.now().strftime("%H%M%S")
    entry = {
        'timestamp': datetime.now().isoformat(),
        'image': image_key,
        'original': original_text or "",
        'corrected': corrected_text or "",
        'pattern_type': 'user_correction'
    }
    hist_key = f"{image_key}::history"
    st.session_state['user_corrections'].setdefault(hist_key, []).append(entry)
    if corrected_text and corrected_text.strip():
        st.session_state['user_corrections'][image_key] = {
            'original': original_text or "",
            'corrected': corrected_text,
            'timestamp': entry['timestamp']
        }

def apply_learned_corrections(text, image_key):
    if image_key in st.session_state['user_corrections']:
        override = st.session_state['user_corrections'][image_key]
        if isinstance(override, dict) and 'corrected' in override:
            return override.get('corrected', text)
    corrected_text = text
    common_fixes = {
        ' or ': ' a ', 'rn': 'm', 'cl': 'd', 'li': 'h', 'rnore': 'more',
        'ornd': 'and', 'tlie': 'the', 'orll': 'all', 'tl1e': 'the',
        'witl1': 'with', 'l1is': 'his', 'l1er': 'her'
    }
    journalism_fixes = {
        'MDSCOW': 'MOSCOW','POLIGH': 'POLISH','UKRAIN': 'UKRAINE',
        'CYEER': 'CYBER','POISDN': 'POISON','TARGEI': 'TARGET',
        'BACKDCOR': 'BACKDOOR','MDSC0W': 'MOSCOW','RU5SIA': 'RUSSIA','UKRA1NE': 'UKRAINE'
    }
    for w, r in common_fixes.items(): corrected_text = corrected_text.replace(w, r)
    for w, r in journalism_fixes.items(): corrected_text = corrected_text.replace(w, r)
    for k, v in st.session_state['user_corrections'].items():
        if isinstance(v, dict) and 'corrected' in v:
            orig, corr = v.get('original', ''), v.get('corrected', '')
            if orig and corr and orig in corrected_text:
                corrected_text = corrected_text.replace(orig, corr)
    return corrected_text

# ===================== Results helpers =====================
def results_to_dataframe(results_list):
    return pd.DataFrame([{
        "filename": r.get("filename",""),
        "text": r.get("text",""),
        "original_text": r.get("original_text",""),
        "engine": r.get("engine",""),
        "best_config": r.get("best_config",""),
        "timestamp": r.get("timestamp","")
    } for r in results_list])

def make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode("utf-8")

def make_json_bytes(results_list) -> bytes:
    minimal = [{
        "filename": r.get("filename",""),
        "text": r.get("text",""),
        "original_text": r.get("original_text",""),
        "engine": r.get("engine",""),
        "best_config": r.get("best_config",""),
        "timestamp": r.get("timestamp","")
    } for r in results_list]
    return json.dumps(minimal, ensure_ascii=False, indent=2).encode("utf-8")

def reocr_single_result(res, enhancement_level, lang):
    best_text, all_configs, processed_img, best_config = extract_text_tesseract(res['image'], enhancement_level, lang)
    corrected_text = apply_learned_corrections(best_text, res['storage_key'])
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

def process_images(files, enhancement_level, lang):
    processed_results = []
    for uploaded_file in files:
        try:
            img = uploaded_file if isinstance(uploaded_file, Image.Image) else Image.open(uploaded_file)
            img = exif_transpose(img)
            img_rot, did_rotate = auto_rotate_for_text(img)
            base_name = getattr(uploaded_file, "name", None)
            storage_key = make_storage_key(base_name, img_rot)
            display_name = (base_name or "UNNAMED") + (" (rotated)" if did_rotate else "")
            best_text, all_configs, processed_img, best_config = extract_text_tesseract(img_rot, enhancement_level, lang)
            corrected_text = apply_learned_corrections(best_text, storage_key)
            processed_results.append({
                'filename': display_name,
                'storage_key': storage_key,
                'text': corrected_text,
                'original_text': best_text,
                'engine': f'Tesseract ({best_config})',
                'all_configs': all_configs,
                'best_config': best_config,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image': img_rot,
                'processed_image': processed_img
            })
        except Exception as e:
            st.error(f"❌ Error processing {getattr(uploaded_file, 'name', 'image')}: {e}")
    return processed_results

def reprocess_existing_results(enhancement_level, lang):
    new_results = []
    for res in st.session_state.ocr_results:
        try:
            img = res['image']
            storage_key = res.get('storage_key')
            display_name = res.get('filename', storage_key or "image.png")
            best_text, all_configs, processed_img, best_config = extract_text_tesseract(img, enhancement_level, lang)
            corrected_text = apply_learned_corrections(best_text, storage_key)
            new_results.append({
                'filename': display_name,
                'storage_key': storage_key,
                'text': corrected_text,
                'original_text': best_text,
                'engine': f'Tesseract ({best_config})',
                'all_configs': all_configs,
                'best_config': best_config,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image': img,
                'processed_image': processed_img
            })
        except Exception as e:
            st.error(f"❌ Error reprocessing {res.get('filename','image')}: {e}")
    return new_results

# ===================== UI: per-image display (uses FORMS) =====================
def display_results(results, enhancement_level, lang, show_processed=False, show_alts=False, debug=False):
    for idx, res in enumerate(results):
        st.markdown("---")
        st.write(f"**File:** {res['filename']}  •  **Engine:** {res['engine']}  •  **Time:** {res['timestamp']}")
        st.caption(f"Storage key: `{res['storage_key']}`")

        col_img, col_txt = st.columns([1, 2])

        with col_img:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("⟲ Rotate 90°", key=f"rotl_{res['storage_key']}"):
                    res['image'] = res['image'].rotate(90, expand=True)
                    reocr_single_result(res, enhancement_level, lang)
                    st.session_state.ocr_results[idx] = res
                    st.rerun()
            with c2:
                if st.button("⟳ Rotate 90°", key=f"rotr_{res['storage_key']}"):
                    res['image'] = res['image'].rotate(-90, expand=True)
                    reocr_single_result(res, enhancement_level, lang)
                    st.session_state.ocr_results[idx] = res
                    st.rerun()

            tabs = st.tabs(["Image", "Processed" if show_processed else " "])
            with tabs[0]:
                st.image(res['image'], caption="Image (auto/manual oriented)", use_container_width=True)
            if show_processed:
                with tabs[1]:
                    st.image(res['processed_image'], caption="Processed (for OCR)", use_container_width=True)

        with col_txt:
            st.text_area("Extracted Text", res['text'], height=220, key=f"view_text_{res['storage_key']}")

            if show_alts or debug:
                with st.expander("Alternative OCR Configurations (raw output)"):
                    for cfg, txt in res['all_configs'].items():
                        st.write(f"[{cfg}]"); st.code(txt if txt else "(no output)")

            # --- FORM: edit + save correction ---
            with st.form(key=f"corr_form_{res['storage_key']}"):
                edited_text = st.text_area("Correct the text below and press Submit.", value=res['text'], height=180, key=f"edit_text_{res['storage_key']}")
                submitted = st.form_submit_button("💾 Submit Correction")
                if submitted:
                    learn_from_corrections(res['original_text'], edited_text, res['storage_key'])
                    res['text'] = edited_text
                    st.session_state.ocr_results[idx] = res
                    st.session_state[f"view_text_{res['storage_key']}"] = edited_text
                    st.success("Saved correction. It will be applied on future runs.")
                    st.rerun()

# ===================== Header & Info =====================
st.title("📝 Journalist's Handwriting OCR Tool")
st.markdown("*Mobile-friendly batch OCR with learning capabilities — Tesseract Edition*")
st.info("Tip: For notebook photos, try **Enhancement = aggressive**. Use rotate buttons if orientation looks wrong.")

# ===================== Upload =====================
st.header("📤 Upload Images")
if batch_mode:
    uploaded_files = st.file_uploader("Choose image files (JPG, PNG, HEIC)", type=['jpg','jpeg','png','heic'], accept_multiple_files=True)
else:
    uploaded_files = [st.file_uploader("Choose an image file", type=['jpg','jpeg','png','heic'])]
uploaded_files = [f for f in uploaded_files if f is not None]
if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully!")

# ===================== Process button =====================
def results_to_download_blobs(results):
    df = results_to_dataframe(results)
    return df, make_csv_bytes(df), make_json_bytes(results)

if st.button("🔍 Process All Images", type="primary"):
    results = process_images(uploaded_files, enhancement_level, ocr_lang)
    for r in results:
        st.session_state.ocr_results.append(r)
    st.success("✅ Processing complete!")
    display_results(results, enhancement_level, ocr_lang, show_processed_images, show_alternative_configs, debug_mode)
    st.markdown("### ⬇️ Download This Run")
    _, csv_bytes_run, json_bytes_run = results_to_download_blobs(results)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.download_button("Download CSV (this run)", data=csv_bytes_run, file_name=f"ocr_results_{ts}.csv", mime="text/csv")
    st.download_button("Download JSON (this run)", data=json_bytes_run, file_name=f"ocr_results_{ts}.json", mime="application/json")

# ===================== Re-run with corrections applied (all) =====================
if st.session_state.ocr_results:
    st.markdown("---")
    st.subheader("🔁 Re-run with Corrections Applied")
    st.caption("Re-OCR all stored images using current settings and apply learned corrections.")
    if st.button("Re-run Now"):
        st.session_state.ocr_results = reprocess_existing_results(enhancement_level, ocr_lang)
        st.success("🔁 Re-run complete! Results updated below.")
        st.rerun()

    st.sidebar.markdown("### ⬇️ Export All Results (Session)")
    df_all = results_to_dataframe(st.session_state.ocr_results)
    st.sidebar.download_button("Download CSV (session)", data=make_csv_bytes(df_all),
                               file_name=f"ocr_results_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    st.sidebar.download_button("Download JSON (session)", data=make_json_bytes(st.session_state.ocr_results),
                               file_name=f"ocr_results_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

# ===================== Corrections Manager =====================
st.markdown("---")
st.header("🛠 Corrections Manager")

def list_override_keys():
    return sorted([k for k, v in st.session_state.user_corrections.items()
                   if isinstance(v, dict) and 'corrected' in v and not str(k).endswith("::history")])

def get_history(key): return st.session_state.user_corrections.get(f"{key}::history", [])

def delete_override(key, delete_history=False):
    if key in st.session_state.user_corrections: del st.session_state.user_corrections[key]
    if delete_history:
        hk = f"{key}::history"
        if hk in st.session_state.user_corrections: del st.session_state.user_corrections[hk]

def export_corrections_json():
    return json.dumps(st.session_state.user_corrections, ensure_ascii=False, indent=2).encode("utf-8")

ov_keys = list_override_keys()
if not ov_keys:
    st.info("No saved corrections yet. Save a correction above to see it here.")
else:
    col_sel, col_btns = st.columns([2, 1])
    with col_sel:
        selected_key = st.selectbox("Select image (storage key):", ov_keys, key="corr_sel")
    with col_btns:
        st.download_button("Export all corrections (JSON)", data=export_corrections_json(),
                           file_name=f"corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

    if selected_key:
        override = st.session_state.user_corrections.get(selected_key, {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original (last saved)**"); st.code(override.get('original','') or "(unknown)")
        with c2:
            st.markdown("**Corrected (applied)**")
            new_corr = st.text_area("Edit corrected text & Update", value=override.get('corrected',''), height=140, key="mgr_edit_corr")

        m1, m2, m3 = st.columns(3)
        with m1:
            if st.button("Update Override", key="mgr_update"):
                learn_from_corrections(override.get('original',''), new_corr, selected_key)
                for i, r in enumerate(st.session_state.ocr_results):
                    if r.get('storage_key') == selected_key:
                        r['text'] = new_corr
                        st.session_state.ocr_results[i] = r
                        st.session_state[f"view_text_{selected_key}"] = new_corr
                st.success("Override updated."); st.rerun()
        with m2:
            if st.button("Delete Override", key="mgr_delete"):
                delete_override(selected_key, delete_history=False)
                st.success("Override deleted (history kept)."); st.rerun()
        with m3:
            if st.button("Delete Override + History", key="mgr_delete_all"):
                delete_override(selected_key, delete_history=True)
                st.success("Override and history deleted."); st.rerun()

        st.markdown("**History**")
        hist = get_history(selected_key)
        if not hist: st.caption("(No history yet.)")
        else:
            for i, h in enumerate(reversed(hist), start=1):
                st.write(f"**{i}.** {h.get('timestamp','')} — corrected →")
                st.code(h.get('corrected','') or "(empty)")
