import os
import json
import io
import hashlib
from datetime import datetime

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import pytesseract
import pandas as pd  # kept since you had it; safe to remove if unused

# ---------- Persistence paths ----------
DATA_DIR = "ocr_data"
CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")
PATTERNS_PATH = os.path.join(DATA_DIR, "learned_patterns.json")
USER_WORDS_PATH = os.path.join(DATA_DIR, "user_words.txt")


# ---------- Utilities for persistence ----------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USER_WORDS_PATH):
        with open(USER_WORDS_PATH, "w", encoding="utf-8") as f:
            pass


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def persist_all():
    save_json(CORRECTIONS_PATH, st.session_state.corrections)
    save_json(PATTERNS_PATH, st.session_state.learned_patterns)
    rebuild_user_words()


def cleanup_token(w: str) -> str:
    # strip simple punctuation that often rides along words
    return w.strip().strip('.,;:!?()[]{}"\'“”’‘')


def rebuild_user_words():
    """Build a Tesseract user-words file from corrected outputs."""
    words = set()
    for corr in st.session_state.corrections.values():
        for w in corr["corrected"].split():
            w2 = cleanup_token(w)
            if w2 and any(ch.isalpha() for ch in w2):
                words.add(w2)
    ensure_data_dir()
    with open(USER_WORDS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(words)))


# ---------- Streamlit page config ----------
st.set_page_config(
    page_title="Journalist's OCR Tool",
    page_icon="📝",
    layout="wide"
)

ensure_data_dir()

# ---------- Session state (load saved learning) ----------
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = []

if "corrections" not in st.session_state:
    st.session_state.corrections = load_json(CORRECTIONS_PATH, {})

if "learned_patterns" not in st.session_state:
    st.session_state.learned_patterns = load_json(PATTERNS_PATH, {})

# Rebuild user words at startup so OCR benefits immediately
rebuild_user_words()


# ---------- Helper functions ----------
def make_image_key(filename, image):
    """Create a unique key for this image"""
    img_data = io.BytesIO()
    image.save(img_data, format="PNG")
    hash_val = hashlib.md5(img_data.getvalue()).hexdigest()[:8]
    return f"{filename}_{hash_val}"


def learn_from_correction(original, corrected):
    """Extract learning patterns from a correction"""
    if not original or not corrected or original == corrected:
        return

    # Word-level learning (most reliable)
    orig_words = original.split()
    corr_words = corrected.split()

    if len(orig_words) == len(corr_words):
        for orig_word, corr_word in zip(orig_words, corr_words):
            if orig_word != corr_word and len(orig_word) > 1:
                entry = st.session_state.learned_patterns.get(orig_word)
                if entry is None:
                    st.session_state.learned_patterns[orig_word] = {
                        "replacement": corr_word,
                        "count": 1,
                        "examples": [{"original": original, "corrected": corrected}],
                    }
                else:
                    entry["count"] += 1
                    entry["replacement"] = corr_word


def save_correction(image_key, original_text, corrected_text):
    """Save a correction and learn from it (and persist to disk)"""
    if original_text == corrected_text:
        return False

    st.session_state.corrections[image_key] = {
        "original": original_text,
        "corrected": corrected_text,
        "timestamp": datetime.now().isoformat(),
    }

    learn_from_correction(original_text, corrected_text)
    persist_all()  # <-- write to disk + rebuild user words
    return True


def apply_learned_corrections(text, image_key):
    """Apply learned corrections to text"""
    # Direct override
    if image_key in st.session_state.corrections:
        return st.session_state.corrections[image_key]["corrected"]

    corrected_text = text

    # Built-in quick fixes (tune as you like)
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
    for wrong, right in built_in_fixes.items():
        corrected_text = corrected_text.replace(wrong, right)

    # Learned patterns
    sorted_patterns = sorted(
        st.session_state.learned_patterns.items(), key=lambda x: -x[1]["count"]
    )
    for pattern, info in sorted_patterns:
        replacement = info["replacement"]
        if pattern in corrected_text:
            corrected_text = corrected_text.replace(pattern, replacement)

    return corrected_text


def preprocess_image(image, enhancement_level="medium"):
    """Preprocess image for OCR"""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    denoised = cv2.fastNlMeansDenoising(gray)

    if enhancement_level == "light":
        processed = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
    elif enhancement_level == "medium":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        processed = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
    else:  # aggressive
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        enhanced = clahe.apply(denoised)
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        processed = cv2.threshold(
            processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

    return processed


def extract_text(image, enhancement_level="medium"):
    """Extract text using Tesseract with multiple configurations"""
    processed_img = preprocess_image(image, enhancement_level)
    pil_processed = Image.fromarray(processed_img)

    # If we have user-words, pass them to Tesseract
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

    best_result = ""
    best_config = "default"

    for i, config in enumerate(configs):
        try:
            result = pytesseract.image_to_string(pil_processed, config=config).strip()
            config_name = ["general", "block", "word"][i]
            if len(result) > len(best_result):
                best_result = result
                best_config = config_name
        except Exception:
            continue

    return best_result, processed_img, best_config


# ---------- Sidebar ----------
st.sidebar.header("Settings")
enhancement_level = st.sidebar.selectbox(
    "Enhancement Level", ["light", "medium", "aggressive"], index=1
)

st.sidebar.markdown("### 🧠 Learning Status")
num_corrections = len(st.session_state.corrections)
num_patterns = len(st.session_state.learned_patterns)
st.sidebar.metric("Saved Corrections", num_corrections)
st.sidebar.metric("Learned Patterns", num_patterns)

if st.session_state.learned_patterns:
    st.sidebar.markdown("**Top Learned Patterns:**")
    sorted_patterns = sorted(
        st.session_state.learned_patterns.items(), key=lambda x: -x[1]["count"]
    )[:5]
    for pattern, info in sorted_patterns:
        st.sidebar.caption(f'"{pattern}" → "{info["replacement"]}" ({info["count"]}x)')

st.sidebar.markdown("### 💾 Persistence")
if st.sidebar.button("Save All Now"):
    persist_all()
    st.sidebar.success("Saved to ocr_data/")

# Export / Import your learning
st.sidebar.download_button(
    "⬇️ Export corrections.json",
    data=json.dumps(st.session_state.corrections, ensure_ascii=False, indent=2),
    file_name="corrections.json",
    mime="application/json",
)
st.sidebar.download_button(
    "⬇️ Export learned_patterns.json",
    data=json.dumps(st.session_state.learned_patterns, ensure_ascii=False, indent=2),
    file_name="learned_patterns.json",
    mime="application/json",
)

uploaded_corr = st.sidebar.file_uploader(
    "Restore corrections.json", type=["json"], key="restore_corr"
)
if uploaded_corr is not None:
    try:
        st.session_state.corrections = json.load(uploaded_corr)
        save_json(CORRECTIONS_PATH, st.session_state.corrections)
        rebuild_user_words()
        st.sidebar.success("Corrections restored.")
    except Exception as e:
        st.sidebar.error(f"Restore failed: {e}")

uploaded_pat = st.sidebar.file_uploader(
    "Restore learned_patterns.json", type=["json"], key="restore_pat"
)
if uploaded_pat is not None:
    try:
        st.session_state.learned_patterns = json.load(uploaded_pat)
        save_json(PATTERNS_PATH, st.session_state.learned_patterns)
        st.sidebar.success("Patterns restored.")
    except Exception as e:
        st.sidebar.error(f"Restore failed: {e}")


# ---------- Main UI ----------
st.title("📝 Journalist's OCR Tool")
st.markdown("*Simple learning OCR — gets smarter with each correction!*")

uploaded_files = st.file_uploader(
    "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} files uploaded!")

    if st.button("🔍 Process Images", type="primary"):
        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image)  # Fix orientation
                image_key = make_image_key(uploaded_file.name, image)

                original_text, processed_img, config = extract_text(
                    image, enhancement_level
                )
                corrected_text = apply_learned_corrections(original_text, image_key)

                result = {
                    "filename": uploaded_file.name,
                    "image_key": image_key,
                    "original_text": original_text,
                    "text": corrected_text,
                    "config": config,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image": image,
                    "processed_image": processed_img,
                }
                st.session_state.ocr_results.append(result)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

            progress.progress((i + 1) / len(uploaded_files))

        st.success("✅ Processing complete!")


# ---------- Results & Learning ----------
if st.session_state.ocr_results:
    st.header("📋 Results & Learning")

    for i, result in enumerate(st.session_state.ocr_results):
        st.markdown("---")
        st.subheader(f"📄 {result['filename']}")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(result["image"], caption="Original", use_container_width=True)

        with col2:
            st.text_area(
                "Current OCR Result:",
                value=result["text"],
                height=150,   # taller preview
                disabled=True,
                key=f"current_{i}",
            )

        # Full-width correction area
        with st.form(f"learn_{i}"):
            st.write("**Correct any mistakes and submit to improve future OCR:**")

            corrected_text = st.text_area(
                "Corrected Text:",
                value=result["text"],
                height=300,   # larger editing area
                key=f"correct_{i}",
            )

            submitted = st.form_submit_button("💾 Save Correction & Learn")

            if submitted:
                if save_correction(
                    result["image_key"], result["original_text"], corrected_text
                ):
                    result["text"] = corrected_text
                    st.session_state.ocr_results[i] = result
                    st.success("✅ Correction saved. Model hints updated.")
                    st.rerun()
                else:
                    st.info("No changes detected.")


# ---------- Export results ----------
if st.session_state.ocr_results:
    st.header("📥 Export")
    combined_text = "\n\n".join(
        [
            f"FILE: {r['filename']}\nTIME: {r['timestamp']}\n\n{r['text']}"
            for r in st.session_state.ocr_results
        ]
    )
    st.download_button(
        "📄 Download Text File",
        data=combined_text,
        file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )


# ---------- Corrections manager ----------
if st.session_state.corrections:
    st.header("🛠️ Saved Corrections")
    for image_key, correction in st.session_state.corrections.items():
        with st.expander(f"Correction: {image_key}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original:**")
                st.code(correction["original"])
            with col2:
                st.write("**Corrected:**")
                st.code(correction["corrected"])

            st.caption(f"Saved: {correction['timestamp']}")
            if st.button("🗑️ Delete", key=f"del_{image_key}"):
                del st.session_state.corrections[image_key]
                persist_all()
                st.rerun()


# ---------- Debug ----------
if st.sidebar.checkbox("Show Debug Info"):
    st.header("🐞 Debug Info")
    st.write("**Corrections:**", st.session_state.corrections)
    st.write("**Learned Patterns:**", st.session_state.learned_patterns)
    st.write("**Number of Results:**", len(st.session_state.ocr_results))

st.markdown("---")
st.markdown("*💡 Tip: The more corrections you make, the smarter the OCR becomes for your handwriting!*")
