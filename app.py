import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import pytesseract
import json
from datetime import datetime
import pandas as pd
import hashlib
import io

# Page config
st.set_page_config(
    page_title="Journalist's OCR Tool",
    page_icon="📝",
    layout="wide"
)

# Initialize session state - SIMPLIFIED
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = []

if "corrections" not in st.session_state:
    st.session_state.corrections = {}

if "learned_patterns" not in st.session_state:
    st.session_state.learned_patterns = {}

# Helper functions
def make_image_key(filename, image):
    """Create a unique key for this image"""
    img_data = io.BytesIO()
    image.save(img_data, format='PNG')
    hash_val = hashlib.md5(img_data.getvalue()).hexdigest()[:8]
    return f"{filename}_{hash_val}"

def save_correction(image_key, original_text, corrected_text):
    """Save a correction and learn from it"""
    if original_text == corrected_text:
        return False

    # Save the correction
    st.session_state.corrections[image_key] = {
        "original": original_text,
        "corrected": corrected_text,
        "timestamp": datetime.now().isoformat()
    }

    # Learn patterns from the correction
    learn_from_correction(original_text, corrected_text)
    return True

def learn_from_correction(original, corrected):
    """Extract learning patterns from a correction"""
    if not original or not corrected or original == corrected:
        return

    # Word-level learning (most reliable)
    orig_words = original.split()
    corr_words = corrected.split()

    # Simple word-by-word mapping
    if len(orig_words) == len(corr_words):
        for orig_word, corr_word in zip(orig_words, corr_words):
            if orig_word != corr_word and len(orig_word) > 1:
                if orig_word not in st.session_state.learned_patterns:
                    st.session_state.learned_patterns[orig_word] = {
                        "replacement": corr_word,
                        "count": 1,
                        "examples": [{"original": original, "corrected": corrected}]
                    }
                else:
                    # Update existing pattern
                    st.session_state.learned_patterns[orig_word]["count"] += 1
                    st.session_state.learned_patterns[orig_word]["replacement"] = corr_word

    # Character-level patterns for single substitutions
    if abs(len(original) - len(corrected)) <= 1:
        # Example: common OCR error patterns
        if "or" in original and "a" in corrected:
            pattern = "or"
            if pattern not in st.session_state.learned_patterns:
                st.session_state.learned_patterns[pattern] = {
                    "replacement": "a",
                    "count": 1,
                    "examples": [{"original": original, "corrected": corrected}]
                }
            else:
                st.session_state.learned_patterns[pattern]["count"] += 1

def apply_learned_corrections(text, image_key):
    """Apply learned corrections to text"""
    # Check for direct override first
    if image_key in st.session_state.corrections:
        return st.session_state.corrections[image_key]["corrected"]

    corrected_text = text

    # Apply built-in fixes
    built_in_fixes = {
        ' or ': ' a ',
        'rn': 'm',
        'cl': 'd',
        'li': 'h',
        'tlie': 'the',
        'ornd': 'and',
        'MDSCOW': 'MOSCOW',
        'RUSSI': 'RUSSIA',
        'CYEER': 'CYBER'
    }

    for wrong, right in built_in_fixes.items():
        corrected_text = corrected_text.replace(wrong, right)

    # Apply learned patterns (sort by count - most frequent first)
    sorted_patterns = sorted(
        st.session_state.learned_patterns.items(),
        key=lambda x: -x[1]["count"]
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

    configs = [
        '--oem 3 --psm 6',
        '--oem 3 --psm 7',
        '--oem 3 --psm 8'
    ]

    best_result = ""
    best_config = "default"

    for i, config in enumerate(configs):
        try:
            result = pytesseract.image_to_string(
                pil_processed, config=config
            ).strip()
            config_name = ["general", "block", "word"][i]

            if len(result) > len(best_result):
                best_result = result
                best_config = config_name
        except Exception:
            continue

    return best_result, processed_img, best_config

# Sidebar
st.sidebar.header("Settings")
enhancement_level = st.sidebar.selectbox(
    "Enhancement Level",
    ["light", "medium", "aggressive"],
    index=1
)

st.sidebar.markdown("### 🧠 Learning Status")
num_corrections = len(st.session_state.corrections)
num_patterns = len(st.session_state.learned_patterns)

st.sidebar.metric("Saved Corrections", num_corrections)
st.sidebar.metric("Learned Patterns", num_patterns)

if st.session_state.learned_patterns:
    st.sidebar.markdown("**Top Learned Patterns:**")
    sorted_patterns = sorted(
        st.session_state.learned_patterns.items(),
        key=lambda x: -x[1]["count"]
    )[:5]
    for pattern, info in sorted_patterns:
        count = info["count"]
        replacement = info["replacement"]
        st.sidebar.caption(f'"{pattern}" → "{replacement}" ({count}x)')

# Main interface
st.title("📝 Journalist's OCR Tool")
st.markdown("*Simple learning OCR - gets smarter with each correction!*")

# File upload
uploaded_files = st.file_uploader(
    "Upload Images",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} files uploaded!")

    if st.button("🔍 Process Images", type="primary"):
        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Load and process image
                image = Image.open(uploaded_file)
                image = ImageOps.exif_transpose(image)  # Fix orientation

                # Create unique key for this image
                image_key = make_image_key(uploaded_file.name, image)

                # Extract text
                original_text, processed_img, config = extract_text(
                    image, enhancement_level
                )

                # Apply learned corrections
                corrected_text = apply_learned_corrections(
                    original_text, image_key
                )

                # Store result
                result = {
                    'filename': uploaded_file.name,
                    'image_key': image_key,
                    'original_text': original_text,
                    'text': corrected_text,
                    'config': config,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'image': image,
                    'processed_image': processed_img
                }

                st.session_state.ocr_results.append(result)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

            progress.progress((i + 1) / len(uploaded_files))

        st.success("✅ Processing complete!")

# Display results
if st.session_state.ocr_results:
    st.header("📋 Results & Learning")

    for i, result in enumerate(st.session_state.ocr_results):
        st.markdown("---")
        st.subheader(f"📄 {result['filename']}")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(result['image'], caption="Original", use_container_width=True)

        with col2:
            # Show current result
            st.text_area(
                "Current OCR Result:",
                value=result['text'],
                height=100,
                disabled=True,
                key=f"current_{i}"
            )

            # Learning form
            with st.form(f"learn_{i}"):
                st.write("**Correct any mistakes and submit to improve future OCR:**")

                corrected_text = st.text_area(
                    "Make corrections here:",
                    value=result['text'],
                    height=100,
                    key=f"correct_{i}"
                )

                submitted = st.form_submit_button("💾 Save Correction & Learn")

                if submitted:
                    if save_correction(
                        result['image_key'], result['original_text'], corrected_text
                    ):
                        # Update the result
                        result['text'] = corrected_text
                        st.session_state.ocr_results[i] = result

                        st.success("✅ Correction saved and learned! Future OCR will be improved.")
                        st.rerun()
                    else:
                        st.info("No changes detected.")

# Export section
if st.session_state.ocr_results:
    st.header("📥 Export")

    combined_text = "\n\n".join([
        f"FILE: {r['filename']}\nTIME: {r['timestamp']}\n\n{r['text']}"
        for r in st.session_state.ocr_results
    ])

    st.download_button(
        "📄 Download Text File",
        data=combined_text,
        file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Corrections manager
if st.session_state.corrections:
    st.header("🛠️ Saved Corrections")

    for image_key, correction in st.session_state.corrections.items():
        with st.expander(f"Correction: {image_key}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original:**")
                st.code(correction['original'])
            with col2:
                st.write("**Corrected:**")
                st.code(correction['corrected'])

            st.caption(f"Saved: {correction['timestamp']}")

            if st.button("🗑️ Delete", key=f"del_{image_key}"):
                del st.session_state.corrections[image_key]
                st.rerun()

# Debug section
if st.sidebar.checkbox("Show Debug Info"):
    st.header("🐞 Debug Info")
    st.write("**Corrections:**", st.session_state.corrections)
    st.write("**Learned Patterns:**", st.session_state.learned_patterns)
    st.write("**Number of Results:**", len(st.session_state.ocr_results))

st.markdown("---")
st.markdown("*💡 Tip: The more corrections you make, the smarter the OCR becomes for your handwriting!*")
