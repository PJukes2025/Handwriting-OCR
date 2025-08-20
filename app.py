# app.py
import os, json
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T

st.set_page_config(page_title="Handwriting OCR (CRNN)", layout="wide")

CORRECTIONS_FILE = "corrections.json"

# ------------------ CRNN ------------------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,64,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256,512,3,1,1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.Conv2d(512,512,3,1,1), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU(True)
        )
        self.rnn = nn.LSTM(512,256,bidirectional=True,num_layers=2,batch_first=True)
        self.fc = nn.Linear(512,num_classes)
    def forward(self,x):
        conv=self.cnn(x)                         # [B,512,1,W’]
        conv=conv.squeeze(2).permute(0,2,1)      # [B,W’,512]
        rnn_out,_=self.rnn(conv)                 # [B,W’,512]
        return self.fc(rnn_out)                  # [B,W’,C]

# ------------------ Preprocess (match training) ------------------
def preprocess_img(img: Image.Image):
    img = img.convert("L")
    # Auto invert if background darker than text
    if ImageOps.invert(img).getextrema()[0] < img.getextrema()[0]:
        img = ImageOps.invert(img)
    w,h = img.size
    new_w = int(w * (48 / h))
    img = img.resize((min(new_w, 1024), 48), Image.BILINEAR)
    if img.size[0] < 1024:
        canvas = Image.new("L", (1024, 48), 255)
        canvas.paste(img, (0, 0))
        img = canvas
    x = T.ToTensor()(img)
    x = T.Normalize((0.5,), (0.5,))(x)
    return x

# ------------------ Decode ------------------
def greedy_decode(logits, charset):
    preds = logits.softmax(2).argmax(2)  # [B,W’]
    out, last = [], -1
    for p in preds[0].cpu().numpy().tolist():
        if p != last and p != 0:  # 0 is CTC blank
            out.append(charset[p])
        last = p
    return "".join(out)

# ------------------ Recognize one line ------------------
def recognize_line(img: Image.Image, model, charset, device="cpu"):
    x = preprocess_img(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return greedy_decode(logits, charset)

# ------------------ Segment page into lines ------------------
def segment_lines(pil_img: Image.Image):
    gray = np.array(pil_img.convert("L"))
    # binarize, dark text -> 255
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_sum = np.sum(thresh, axis=1)
    lines, start = [], None
    for i, val in enumerate(horizontal_sum):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            if i - start > 5:  # min line height
                lines.append(pil_img.crop((0, start, pil_img.width, i)))
            start = None
    if start is not None:
        lines.append(pil_img.crop((0, start, pil_img.width, pil_img.height)))
    return lines

# ------------------ Corrections persistence ------------------
def load_corrections():
    try:
        if os.path.exists(CORRECTIONS_FILE):
            with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def save_corrections(corrections):
    try:
        with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(corrections, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save corrections: {e}")

# Build replacement maps for auto-correct
def build_replacement_maps(corrections_list):
    """
    Returns:
      - whole_line_map: dict[original_line] -> corrected_line
      - token_map: dict[token] -> corrected_token (most frequent 1:1 replacements)
    """
    whole_line_map = {}
    token_counts = {}  # token -> {replacement -> count}

    for item in corrections_list:
        if not isinstance(item, dict):
            continue
        orig = item.get("original")
        corr = item.get("corrected")
        if isinstance(orig, str) and isinstance(corr, str) and orig.strip():
            # whole line
            whole_line_map[orig] = corr

            # token-level (simple, safe heuristic)
            o_tokens = orig.split()
            c_tokens = corr.split()
            if len(o_tokens) == len(c_tokens):
                for o_tok, c_tok in zip(o_tokens, c_tokens):
                    if o_tok != c_tok and len(o_tok) > 1 and len(c_tok) > 0:
                        token_counts.setdefault(o_tok, {})
                        token_counts[o_tok][c_tok] = token_counts[o_tok].get(c_tok, 0) + 1

    # choose most frequent replacement per token
    token_map = {}
    for tok, repls in token_counts.items():
        best = max(repls.items(), key=lambda kv: kv[1])[0]
        token_map[tok] = best

    return whole_line_map, token_map

def apply_auto_corrections(text, whole_line_map, token_map):
    """Apply whole-line correction first; otherwise token replacements."""
    if not isinstance(text, str) or not text:
        return text
    if text in whole_line_map:
        return whole_line_map[text]
    toks = text.split()
    changed = False
    for i, t in enumerate(toks):
        if t in token_map:
            toks[i] = token_map[t]
            changed = True
    return " ".join(toks) if changed else text

# ------------------ Export: user dictionary from corrections ------------------
def build_user_dictionary_text(corrections_list):
    """
    Construct a simple newline-separated list of words from corrected text.
    Useful for Tesseract user-words or as a reference list.
    """
    vocab = set()
    for item in corrections_list:
        corr = item.get("corrected", "")
        for tok in corr.split():
            tok = tok.strip().strip(".,;:!?()[]{}\"'“”’‘")
            if tok and any(ch.isalpha() for ch in tok):
                vocab.add(tok)
    # sort case-insensitively but return original casing
    return "\n".join(sorted(vocab, key=lambda s: s.lower()))

# ------------------ Load model ------------------
@st.cache_resource
def load_model():
    if not os.path.exists("crnn_handwriting.pt"):
        raise FileNotFoundError("crnn_handwriting.pt not found beside app.py")
    ckpt = torch.load("crnn_handwriting.pt", map_location="cpu")
    charset = ckpt.get("charset")
    if not isinstance(charset, (list, tuple)):
        raise RuntimeError("Checkpoint missing 'charset' list.")
    model = CRNN(len(charset))
    model.load_state_dict(ckpt["model"])
    return model, charset

# ------------------ UI ------------------
st.title("📝 Handwriting OCR (CRNN)")
st.caption("Upload a page or a single line. Rotate if needed. Auto-correct uses your saved edits.")

# Sidebar controls
rotate_choice = st.sidebar.radio("Rotate image", ["0°", "90°", "180°", "270°"], index=0)
auto_correct_on = st.sidebar.toggle("Auto-correct using saved corrections", value=True)

# Sidebar: Corrections tools
st.sidebar.markdown("### Corrections")
corrs = load_corrections()
if not corrs:
    st.sidebar.caption("No corrections saved yet.")
else:
    for c in corrs[-50:][::-1]:  # show latest 50
        o = c.get("original", "")
        a = c.get("corrected", "")
        st.sidebar.write(f"• **{o}** → {a}")

# Export / Import corrections
st.sidebar.markdown("---")
st.sidebar.markdown("**Export / Import**")

# Download corrections.json
st.sidebar.download_button(
    "⬇️ Download corrections.json",
    data=json.dumps(corrs, ensure_ascii=False, indent=2),
    file_name="corrections.json",
    mime="application/json",
    disabled=(len(corrs) == 0),
)

# Upload corrections.json
uploaded_corr = st.sidebar.file_uploader("⬆️ Import corrections.json", type=["json"])
if uploaded_corr is not None:
    try:
        data = json.load(uploaded_corr)
        if isinstance(data, list):
            save_corrections(data)
            st.sidebar.success(f"Imported {len(data)} corrections.")
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid format: expected a JSON list of {original, corrected} objects.")
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")

# Build & download user dictionary text
if len(corrs) > 0:
    user_dict_text = build_user_dictionary_text(corrs)
    st.sidebar.download_button(
        "⬇️ Download user_words.txt",
        data=user_dict_text,
        file_name="user_words.txt",
        mime="text/plain",
    )

# Clear corrections
if st.sidebar.button("🗑️ Clear all corrections"):
    save_corrections([])
    st.sidebar.success("Cleared.")
    st.experimental_rerun()

# Main upload
uploaded = st.file_uploader("Upload a handwriting image", type=["jpg", "jpeg", "png"])
if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    # Apply rotation
    if rotate_choice == "90°":
        pil_img = pil_img.rotate(-90, expand=True)
    elif rotate_choice == "180°":
        pil_img = pil_img.rotate(180, expand=True)
    elif rotate_choice == "270°":
        pil_img = pil_img.rotate(-270, expand=True)

    st.image(pil_img, caption="Uploaded image", use_container_width=True)

    # Load model
    try:
        model, charset = load_model()
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

    # Auto-correct maps
    whole_line_map, token_map = build_replacement_maps(corrs) if auto_correct_on else ({}, {})

    st.subheader("OCR Result")

    new_corrections = []
    if pil_img.height < 100:  # treat as single line
        raw = recognize_line(pil_img, model, charset)
        auto_text = apply_auto_corrections(raw, whole_line_map, token_map) if auto_correct_on else raw

        st.write("**Raw:**", raw)
        st.write("**Auto-corrected:**" if auto_correct_on else "**(Auto-correct off) Final:**", auto_text)

        edited = st.text_input("Edit & save correction (optional)", auto_text)
        if st.button("💾 Save correction for this line"):
            if edited != raw:
                new_corrections.append({"original": raw, "corrected": edited})
                st.success("Saved correction ✅")

    else:
        # Multi-line page
        lines = segment_lines(pil_img)
        full_out = []
        for i, line in enumerate(lines, 1):
            raw = recognize_line(line, model, charset)
            auto_text = apply_auto_corrections(raw, whole_line_map, token_map) if auto_correct_on else raw

            with st.expander(f"Line {i}", expanded=True):
                st.image(line, caption=f"Line {i} crop", use_container_width=True)
                st.text(f"Raw: {raw}")
                st.text_input("Auto/Final (editable):", value=auto_text, key=f"edit_{i}")
                final_text = st.session_state.get(f"edit_{i}", auto_text)
                full_out.append(final_text)
                if final_text != raw:
                    new_corrections.append({"original": raw, "corrected": final_text})

        st.markdown("---")
        st.text_area("Full text", value="\n".join(full_out), height=220)
        if st.button("💾 Save all edited lines as corrections"):
            if new_corrections:
                corrs.extend(new_corrections)
                save_corrections(corrs)
                st.success(f"Saved {len(new_corrections)} correction(s) ✅")
            else:
                st.info("No changes to save.")
