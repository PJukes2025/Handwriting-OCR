# app.py
import os, io, json, time, tempfile, hashlib
from datetime import datetime

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import pytesseract

# ========== Paths ==========
DATA_DIR = "ocr_data"
CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")
PATTERNS_PATH    = os.path.join(DATA_DIR, "learned_patterns.json")
USER_WORDS_PATH  = os.path.join(DATA_DIR, "user_words.txt")

# ========== JSON layer (robust, atomic, versioned) ==========
SCHEMA_VERSION = 1  # bump if you change structure

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _wrap(data):
    return {"_meta": {"schema": SCHEMA_VERSION, "saved_at": _now_iso()}, "data": data}

def _unwrap(payload, default):
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data", default), payload.get("_meta", {})
    return payload, {"schema": 0, "saved_at": None}

def _migrate(payload):
    """Migrate old payloads to current schema (extend as needed)."""
    if not isinstance(payload, dict):
        return {"_meta": {"schema": SCHEMA_VERSION, "saved_at": _now_iso()}, "data": payload}
    meta = payload.get("_meta", {})
    ver = meta.get("schema", 0)
    data = payload.get("data", payload)
    # if ver < 1:  # example future migration
    #     pass
    return {"_meta": {"schema": SCHEMA_VERSION, "saved_at": _now_iso()}, "data": data}

def _atomic_write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dir_)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass

def load_json_versioned(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return default, {"schema": 0, "saved_at": None}
    try:
        if isinstance(payload, dict) and payload.get("_meta", {}).get("schema", 0) != SCHEMA_VERSION:
            payload = _migrate(payload)
            _atomic_write(path, payload)  # write back migrated
        return _unwrap(payload, default)
    except Exception:
        return default, {"schema": 0, "saved_at": None}

def save_json_versioned(path, data):
    _atomic_write(path, _wrap(data))

# ========== Helpers ==========
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USER_WORDS_PATH):
        with open(USER_WORDS_PATH, "w", encoding="utf-8") as f:
            pass

def cleanup_token(w: str) -> str:
    return w.strip().strip('.,;:!?()[]{}"\'“”’‘')

def rebuild_user_words():
    """Build Tesseract user-words file from all corrected outputs."""
    words = set()
    for corr in st.session_state.corrections.values():
        for w in corr["corrected"].split():
            w2 = cleanup_token(w)
            if w2 and any(ch.isalpha() for ch in w2):
                words.add(w2)
    with open(USER_WORDS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(words)))

def make_image_key(filename, image):
    """Unique key per image (name + hash)."""
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    digest = hashlib.md5(bio.getvalue()).hexdigest()[:8]
    return f"{filename}_{digest}"

def learn_from_correction(original, corrected):
    """Very simple word/char learning map."""
    if not original or not corrected or original == corrected:
        return
    orig_words, corr_words = original.split(), corrected.split()
    if len(orig_words) == len(corr_words):
        for ow, cw in zip(orig_words, corr_words):
            if ow != cw and len(ow) > 1:
                entry = st.session_state.learned_patterns.get(ow)
                if entry is None:
                    st.session_state.learned_patterns[ow] = {
                        "replacement": cw,
                        "count": 1,
                        "examples": [{"original": original, "corrected": corrected}],
                    }
                else:
                    entry["count"] += 1
                    entry["replacement"] = cw

def save_correction(image_key, original_text, corrected_text):
    if original_text == corrected_text:
        return False
    st.session_state.corrections[image_key] = {
        "original": original_text,
        "corrected": corrected_text,
        "timestamp": datetime.now().isoformat(),
    }
    learn_from_correction(original_text, corrected_text)
    persist_all()
    return True

def apply_learned_corrections(text, image_key):
    if image_key in st.session_state.corrections:
        return st.session_state.corrections[image_key]["corrected"]
    corrected_text = text
    built_in_fixes = {
        " or ": " a ",
        "rn": "m",
        "cl": "d",
        "li": "h",
        "tlie": "the",
        "ornd": "and",
        "MDSCOW": "MOSCOW",
        "RUSSI": "RUSSIA",
        "CYEER": "CYBER",
    }
    for a, b in built_in_fixes.items():
        corrected_text = corrected_text.replace(a, b)
    patterns = sorted(st.session_state.learned_patterns.items(), key=lambda x: -x[1]["count"])
    for pat, info in patterns:
        if pat in corrected_text:
            corrected_text = corrected_text.replace(pat, info["replacement"])
    return corrected_text

def preprocess_image(image, enhancement_level="medium"):
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if len(arr.shape) == 3 else arr
    den = cv2.fastNlMeansDenoising(gray)
    if enhancement_level == "light":
        processed = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif enhancement_level == "medium":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enh = clahe.apply(den)
        processed = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        enh = clahe.apply(den)
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(enh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return processed

def extract_text(image, enhancement_level="medium"):
    processed = preprocess_image(image, enhancement_level)
    pil_proc = Image.fromarray(processed)

    extra = ""
    try:
        if os.path.getsize(USER_WORDS_PATH) > 0:
            extra = f' --user-words "{USER_WORDS_PATH}"'
    except Exception:
        pass

    configs = [
        f"--oem 3 --psm 6{extra}",
        f"--oem 3 --psm 7{extra}",
        f"--oem 3 --psm 8{extra}",
    ]
    best, best_cfg = "", "default"
    for i, cfg in enumerate(configs):
        try:
            out = pytesseract.image_to_string(pil_proc, config=cfg).strip()
            if len(out) > len(best):
                best, best_cfg = out, ["general", "block", "word"][i]
        except Exception:
            continue
    return best, processed, best_cfg

def persist_all():
    save_json_versioned(CORRECTIONS_PATH, st.session_state.corrections)
    save_json_versioned(PATTERNS_PATH,    st.session_state.learned_patterns)
    rebuild_user_words()

# ========== UI ==========
st.set_page_config(page_title="Journalist's OCR Tool", page_icon="📝", layout="wide")
ensure_data_dir()

# Session state
if "ocr_results" not in st.session_state: st.session_state.ocr_results = []
if "corrections" not in st.session_state:
    st.session_state.corrections, _ = load_json_versioned(CORRECTIONS_PATH, {})
if "learned_patterns" not in st.session_state:
    st.session_state.learned_patterns, _ = load_json_versioned(PATTERNS_PATH, {})
rebuild_user_words()

# Sidebar
st.sidebar.header("Settings")
enhancement_level = st.sidebar.selectbox("Enhancement Level", ["light","medium","aggressive"], index=1)

st.sidebar.markdown("### 🧠 Learning Status")
st.sidebar.metric("Saved Corrections", len(st.session_state.corrections))
st.sidebar.metric("Learned Patterns", len(st.session_state.learned_patterns))

if st.sidebar.button("💾 Save All Now"):
    persist_all()
    st.sidebar.success("Saved to ocr_data/")

st.sidebar.markdown("### Export / Import")
st.sidebar.download_button(
    "⬇️ corrections.json",
    data=json.dumps(_wrap(st.session_state.corrections), ensure_ascii=False, indent=2),
    file_name="corrections.json",
    mime="application/json",
)
st.sidebar.download_button(
    "⬇️ learned_patterns.json",
    data=json.dumps(_wrap(st.session_state.learned_patterns), ensure_ascii=False, indent=2),
    file_name="learned_patterns.json",
    mime="application/json",
)
imp_corr = st.sidebar.file_uploader("Restore corrections.json", type=["json"], key="restore_corr")
if imp_corr:
    try:
        payload = json.load(imp_corr)
        data, _ = _unwrap(payload, {})
        st.session_state.corrections = data
        persist_all()
        st.sidebar.success("Corrections restored.")
    except Exception as e:
        st.sidebar.error(f"Restore failed: {e}")

imp_pat = st.sidebar.file_uploader("Restore learned_patterns.json", type=["json"], key="restore_pat")
if imp_pat:
    try:
        payload = json.load(imp_pat)
        data, _ = _unwrap(payload, {})
        st.session_state.learned_patterns = data
        persist_all()
        st.sidebar.success("Patterns restored.")
    except Exception as e:
        st.sidebar.error(f"Restore failed: {e}")

# Main
st.title("📝 Journalist's OCR Tool")
st.markdown("*Simple learning OCR — gets smarter with each correction!*")

uploaded_files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} files uploaded!")
    if st.button("🔍 Process Images", type="primary"):
        progress = st.progress(0)
        for i, uf in enumerate(uploaded_files):
            try:
                image = Image.open(uf)
                image = ImageOps.exif_transpose(image)
                key = make_image_key(uf.name, image)
                raw_text, proc_img, cfg = extract_text(image, enhancement_level)
                fixed = apply_learned_corrections(raw_text, key)
                st.session_state.ocr_results.append({
                    "filename": uf.name,
                    "image_key": key,
                    "original_text": raw_text,
                    "text": fixed,
                    "config": cfg,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image": image,
                    "processed_image": proc_img,
                })
            except Exception as e:
                st.error(f"Error processing {uf.name}: {e}")
            progress.progress((i+1)/len(uploaded_files))
        st.success("✅ Processing complete!")

# Results & learning
if st.session_state.ocr_results:
    st.header("📋 Results & Learning")
    for i, r in enumerate(st.session_state.ocr_results):
        st.markdown("---")
        st.subheader(f"📄 {r['filename']}")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(r["image"], caption="Original", use_container_width=True)
        with col2:
            st.text_area("Current OCR Result:", value=r["text"], height=150, disabled=True, key=f"cur_{i}")
        with st.form(f"learn_{i}"):
            st.write("**Correct any mistakes and submit to improve future OCR:**")
            corrected = st.text_area("Corrected Text:", value=r["text"], height=300, key=f"corr_{i}")
            submitted = st.form_submit_button("💾 Save Correction & Learn")
            if submitted:
                if save_correction(r["image_key"], r["original_text"], corrected):
                    r["text"] = corrected
                    st.session_state.ocr_results[i] = r
                    st.success("✅ Saved & learned.")
                    st.rerun()
                else:
                    st.info("No changes detected.")

# Export combined text
if st.session_state.ocr_results:
    st.header("📥 Export")
    combined = "\n\n".join([f"FILE: {x['filename']}\nTIME: {x['timestamp']}\n\n{x['text']}" for x in st.session_state.ocr_results])
    st.download_button("📄 Download Text File", data=combined,
                       file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                       mime="text/plain")

# Corrections manager
if st.session_state.corrections:
    st.header("🛠️ Saved Corrections")
    for k, c in st.session_state.corrections.items():
        with st.expander(f"Correction: {k}"):
            col1, col2 = st.columns(2)
            with col1: st.write("**Original:**");  st.code(c["original"])
            with col2: st.write("**Corrected:**"); st.code(c["corrected"])
            st.caption(f"Saved: {c['timestamp']}")
            if st.button("🗑️ Delete", key=f"del_{k}"):
                del st.session_state.corrections[k]
                persist_all()
                st.rerun()

# Debug
if st.sidebar.checkbox("Show Debug Info"):
    st.header("🐞 Debug Info")
    st.write("**Corrections:**", st.session_state.corrections)
    st.write("**Learned Patterns:**", st.session_state.learned_patterns)
    st.write("**Number of Results:**", len(st.session_state.ocr_results))

st.markdown("---")
st.markdown("*💡 Tip: The more corrections you make, the smarter the OCR becomes for your handwriting!*")
