# Journalist’s Handwriting OCR Tool

A mobile-friendly OCR application designed specifically for journalists who need to quickly digitize handwritten notes from phones and tablets.

## Features

- 📱 **Mobile-optimized** - works perfectly on iPad/iPhone browsers
- 🔄 **Batch processing** - upload multiple images at once
- 🧠 **Learning system** - adapts to your handwriting patterns over time
- 📤 **Multiple export formats** - text, CSV, and learning data backup
- 🔍 **Dual OCR engines** - uses both Tesseract and EasyOCR for best results

## Quick Start

1. **Deploy to Streamlit Cloud:**
- Upload this repo to GitHub
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub repo
- Deploy and get your permanent URL
1. **Use on Mobile:**
- Bookmark the deployed URL on your phone/tablet
- Take photos of handwritten notes
- Upload directly from your mobile browser
- Process and download results

## How to Use

1. **Upload Images:** Take clear photos of handwritten notes with good lighting
1. **Choose Enhancement Level:**
- Light: Clear, dark handwriting
- Medium: Typical handwriting (recommended)
- Aggressive: Faint or cramped margin notes
1. **Process:** Click “Process All Images”
1. **Review & Correct:** Edit any OCR mistakes in the text areas
1. **Learn:** Click “Learn from this correction” to improve future accuracy
1. **Export:** Download as text file, CSV, or backup your learning data

## Tips for Best Results

- Ensure good lighting when photographing notes
- Try to keep text lines straight in photos
- Use different enhancement levels for different writing styles
- Make corrections to help the system learn your patterns
- Process neat notes and margin annotations separately

## Technical Details

- Uses both Tesseract OCR and EasyOCR engines
- Advanced image preprocessing for handwriting optimization
- Learning system stores corrections to improve future processing
- Mobile-responsive design with touch-friendly interface

## Requirements

See `requirements.txt` for Python dependencies. All dependencies are automatically installed on Streamlit Cloud.
