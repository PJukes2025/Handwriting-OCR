import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import json
from datetime import datetime
import pandas as pd

st.set_page_config(
page_title=“OCR Tool”,
page_icon=“📝”,
layout=“wide”
)

st.title(“📝 Handwriting OCR Tool”)
st.markdown(”*Mobile-friendly OCR for journalists*”)

if ‘ocr_results’ not in st.session_state:
st.session_state.ocr_results = []
if ‘user_corrections’ not in st.session_state:
st.session_state.user_corrections = {}

def preprocess_image(image, enhancement_level=“medium”):
img_array = np.array(image)

```
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
```

def extract_text(image, enhancement_level=“medium”):
processed_img = preprocess_image(image, enhancement_level)
pil_processed = Image.fromarray(processed_img)

```
configs = [
    '--oem 3 --psm 6',
    '--oem 3 --psm 7', 
    '--oem 3 --psm 8'
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
            
    except Exception as e:
        continue

return best_result, processed_img, best_config
```

def apply_corrections(text):
corrections = {
’ or ’: ’ a ’,
‘rn’: ‘m’,
‘cl’: ‘d’,
‘li’: ‘h’,
‘tlie’: ‘the’,
‘ornd’: ‘and’,
‘MDSCOW’: ‘MOSCOW’,
‘RUSSI’: ‘RUSSIA’,
‘SKRIPAL’: ‘SKRIPAL’,
‘CYEER’: ‘CYBER’
}

```
corrected = text
for wrong, right in corrections.items():
    corrected = corrected.replace(wrong, right)

return corrected
```

enhancement_level = st.sidebar.selectbox(
“Enhancement Level”,
[“light”, “medium”, “aggressive”],
index=1
)

uploaded_files = st.file_uploader(
“Upload Images”,
type=[‘jpg’, ‘jpeg’, ‘png’],
accept_multiple_files=True
)

if uploaded_files:
st.success(f”📁 {len(uploaded_files)} files uploaded!”)

```
if st.button("🔍 Process Images"):
    progress = st.progress(0)
    results = []
    
    for i, file in enumerate(uploaded_files):
        try:
            image = Image.open(file)
            text, processed_img, config = extract_text(image, enhancement_level)
            corrected_text = apply_corrections(text)
            
            result = {
                'filename': file.name,
                'text': corrected_text,
                'original': text,
                'config': config,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image': image,
                'processed': processed_img
            }
            results.append(result)
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
        
        progress.progress((i + 1) / len(uploaded_files))
    
    st.session_state.ocr_results = results
    st.success("✅ Processing complete!")
```

if st.session_state.ocr_results:
st.header(“📋 Results”)

```
for i, result in enumerate(st.session_state.ocr_results):
    with st.expander(f"📄 {result['filename']}", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            edited_text = st.text_area(
                "Text:",
                value=result['text'],
                height=100,
                key=f"edit_{i}"
            )
        
        with col2:
            st.image(result['image'], caption="Original", width=200)

st.header("📥 Export")

combined_text = "\n\n".join([
    f"FILE: {r['filename']}\n{r['text']}"
    for r in st.session_state.ocr_results
])

st.download_button(
    "📄 Download Text",
    data=combined_text,
    file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain"
)

df = pd.DataFrame([{
    'filename': r['filename'],
    'text': r['text'],
    'config': r['config'],
    'timestamp': r['timestamp']
} for r in st.session_state.ocr_results])

csv = df.to_csv(index=False)
st.download_button(
    "📊 Download CSV",
    data=csv,
    file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)
```
