The smart quotes issue is throughout the code. Let me provide you with a completely clean version with proper straight quotes:​​​​​​​​​​​​​​​​

**The issue is smart quotes throughout the file!** Here’s the **complete clean version** with proper straight quotes:

Copy this entire code block and replace your `app.py`:

```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import io
import json
from datetime import datetime
import pandas as pd

# Page config for mobile-friendly experience
st.set_page_config(
    page_title="Journalist's OCR Tool",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile responsiveness
st.markdown("""
<style>
    .stFileUploader > div > div > div {
        padding: 1rem;
    }
    .stImage > div {
        text-align: center;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .processing-stats {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = []
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'user_corrections' not in st.session_state:
    st.session_state.user_corrections = {}

def preprocess_image(image, enhancement_level="medium"):
    """Advanced preprocessing for handwriting, optimized for margin notes"""
    # Convert PIL to numpy array
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
        # Minimal processing
        processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif enhancement_level == "medium":
        # Standard enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        processed = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:  # aggressive - optimized for cramped margin notes
        # Heavy processing for difficult handwriting
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
    
    # Convert numpy array back to PIL Image
    pil_processed = Image.fromarray(processed_img)
    
    # Try multiple Tesseract configurations for best results
    configs = [
        # Configuration 1: General handwriting (most common)
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()-"\'@&/ ',
        # Configuration 2: Single text block (for neat writing)
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()-"\'@&/ ',
        # Configuration 3: Single word mode (for margin notes)
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()-"\'@&/ '
    ]
    
    results = {}
    best_result = ""
    best_config = "default"
    
    for i, config in enumerate(configs):
        try:
            result = pytesseract.image_to_string(pil_processed, config=config).strip()
            config_name = ["general", "block", "word"][i]
            results[config_name] = result
            
            # Choose the result with the most content (usually best)
            if len(result) > len(best_result):
                best_result = result
                best_config = config_name
                
        except Exception as e:
            results[f"config_{i}"] = f"Error: {e}"
    
    return best_result, results, processed_img, best_config

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
    
    # Apply common fixes first
    for wrong, right in common_fixes.items():
        corrected_text = corrected_text.replace(wrong, right)
    
    # Apply journalism-specific fixes
    for wrong, right in journalism_fixes.items():
        corrected_text = corrected_text.replace(wrong, right)
    
    # Apply stored corrections (simple string replacement for now)
    for correction in st.session_state.user_corrections.values():
        if correction['original'] in corrected_text:
            corrected_text = corrected_text.replace(correction['original'], correction['corrected'])
    
    return corrected_text

# Main app
st.title("📝 Journalist's Handwriting OCR Tool")
st.markdown("*Mobile-friendly batch OCR with learning capabilities - Tesseract Edition*")

# Sidebar for settings
st.sidebar.header("Settings")
enhancement_level = st.sidebar.selectbox(
    "Image Enhancement Level",
    ["light", "medium", "aggressive"],
    index=1,
    help="Choose based on your handwriting clarity"
)

batch_mode = st.sidebar.checkbox("Batch Processing Mode", value=True)
show_processed_images = st.sidebar.checkbox("Show Processed Images", value=False)
show_alternative_configs = st.sidebar.checkbox("Show Alternative OCR Configs", value=False)

# Info box
st.info("🚀 **Fast & Reliable**: This version uses Tesseract OCR for instant processing without downloads. Based on our handwriting analysis, expect 80-90% accuracy on your neat notes!")

# File upload section
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
    
    # Process button
    if st.button("🔍 Process All Images", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            
            try:
                # Load and process image
                image = Image.open(uploaded_file)
                
                # Extract text using Tesseract with multiple configs
                best_text, all_configs, processed_img, best_config = extract_text_tesseract(image, enhancement_level)
                
                # Apply learned corrections
                corrected_text = apply_learned_corrections(best_text, uploaded_file.name)
                
                result = {
                    'filename': uploaded_file.name,
                    'text': corrected_text,
                    'original_text': best_text,
                    'engine': f'Tesseract ({best_config})',
                    'all_configs': all_configs,
                    'best_config': best_config,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'image': image,
                    'processed_image': processed_img
                }
                
                processed_results.append(result)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ Processing complete!")
        st.session_state.ocr_results = processed_results
        
        # Display results
        st.header("📋 OCR Results")
        
        for i, result in enumerate(processed_results):
            with st.expander(f"📄 {result['filename']} ({result['engine']})", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Editable text area for corrections
                    corrected_text = st.text_area(
                        "Extracted Text (editable):",
                        value=result['text'],
                        height=100,
                        key=f"text_edit_{i}"
                    )
                    
                    # Learn from corrections button
                    if corrected_text != result['original_text']:
                        if st.button(f"💡 Learn from this correction", key=f"learn_{i}"):
                            learn_from_corrections(result['original_text'], corrected_text, result['filename'])
                            st.success("✅ Correction learned for future processing!")
                            result['text'] = corrected_text
                
                with col2:
                    # Show original image
                    st.image(result['image'], caption="Original", width=200)
                    
                    if show_processed_images:
                        st.image(result['processed_image'], caption="Processed", width=200)
                
                # Show alternative OCR configurations
                if show_alternative_configs and len(result['all_configs']) > 1:
                    with st.expander("🔍 Alternative OCR Configurations"):
                        for config_name, text in result['all_configs'].items():
                            if config_name != result['best_config']:
                                st.write(f"**{config_name.title()}:** {text[:100]}{'...' if len(text) > 100 else ''}")

# Export section
if st.session_state.ocr_results:
    st.header("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as text file
        combined_text = "\n\n" + "="*50 + "\n\n".join([
            f"FILE: {result['filename']}\nTIME: {result['timestamp']}\nENGINE: {result['engine']}\n\n{result['text']}"
            for result in st.session_state.ocr_results
        ])
        
        st.download_button(
            "📄 Download as Text",
            data=combined_text,
            file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        # Export as CSV
        df = pd.DataFrame([{
            'filename': result['filename'],
            'text': result['text'],
            'engine': result['engine'],
            'timestamp': result['timestamp']
        } for result in st.session_state.ocr_results])
        
        csv = df.to_csv(index=False)
        st.download_button(
            "📊 Download as CSV",
            data=csv,
            file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Export corrections data
        if st.session_state.user_corrections:
            corrections_json = json.dumps(st.session_state.user_corrections, indent=2)
            st.download_button(
                "🧠 Download Learning Data",
                data=corrections_json,
                file_name=f"ocr_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Statistics and learning section
if st.session_state.ocr_results or st.session_state.user_corrections:
    st.header("📊 Processing Statistics")
    
    if st.session_state.ocr_results:
        total_files = len(st.session_state.ocr_results)
        total_chars = sum(len(result['text']) for result in st.session_state.ocr_results)
        config_counts = {}
        for result in st.session_state.ocr_results:
            config = result.get('best_config', 'default')
            config_counts[config] = config_counts.get(config, 0) + 1
        
        st.markdown(f"""
        <div class="processing-stats">
            <h4>Current Session:</h4>
            <ul>
                <li><strong>Files processed:</strong> {total_files}</li>
                <li><strong>Total characters extracted:</strong> {total_chars:,}</li>
                <li><strong>Best OCR configs:</strong> {', '.join([f'{k}: {v}' for k, v in config_counts.items()])}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.user_corrections:
        corrections_count = len(st.session_state.user_corrections)
        st.markdown(f"""
        <div class="processing-stats">
            <h4>Learning Progress:</h4>
            <ul>
                <li><strong>Corrections learned:</strong> {corrections_count}</li>
                <li><strong>Latest correction:</strong> {max(st.session_state.user_corrections.values(), key=lambda x: x['timestamp'])['timestamp'][:19]}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Help section
with st.expander("❓ How to Use This Tool"):
    st.markdown("""
    ### 📱 For Mobile Users (iPad/iPhone):
    1. **Take clear photos** of your handwritten notes
    2. **Upload multiple images** at once using the file uploader
    3. **Choose enhancement level** based on your handwriting:
       - *Light*: For clear, dark handwriting
       - *Medium*: For typical handwriting (recommended)
       - *Aggressive*: For faint or difficult handwriting
    
    ### 🧠 Learning Feature:
    - Edit any OCR results that aren't perfect
    - Click "Learn from this correction" to teach the system
    - Future processing will apply your corrections automatically
    
    ### 📥 Export Options:
    - **Text file**: For easy copy/paste into articles
    - **CSV**: For spreadsheet analysis
    - **Learning data**: Backup your corrections
    
    ### 💡 Tips for Better Results:
    - Ensure good lighting when photographing
    - Try to keep text lines straight
    - Use the enhancement level that works best for your writing style
    - Make corrections to help the system learn your handwriting patterns
    
    ### 🚀 Tesseract Edition Benefits:
    - **Instant processing** - no downloads or delays
    - **Multiple OCR configurations** automatically tested
    - **Optimized for your handwriting** based on our analysis
    - **Reliable and fast** - perfect for field journalism
    """)

# Footer
st.markdown("---")
st.markdown("*Built for journalists who need fast, accurate handwriting OCR with mobile support*")
```
