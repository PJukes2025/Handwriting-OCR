# app.py
import os, io, json, hashlib, urllib.request, tempfile
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T

st.set_page_config(page_title="Handwriting OCR (CRNN)", page_icon="📝", layout="wide")

# ==================== Model catalogue (inline, per your request) ====================
MODELS = {
    "v1a": {
        "label": "crnn_handwriting.1.pt",
        "url": "https://github.com/PJukes2025/Handwriting-OCR/releases/download/v1/crnn_handwriting.1.pt",
        "sha256": "2dffa6099d26c055c086eeadb04194230c075eaae4a0579667f73d170eec9913",
        "local": "crnn_handwriting.1.pt",
    },
    "v1b": {
        "label": "crnn_handwriting.2.pt",
        "url": "https://github.com/PJukes2025/Handwriting-OCR/releases/download/v1/crnn_handwriting.2.pt",
        "sha256": "23975a41afa2629e28a0a38a2e8745c62c88e43efe4945fd3431c23fbf08d00d",
        "local": "crnn_handwriting.2.pt",
    },
}

CORRECTIONS_FILE = "corrections.json"

# ==================== CRNN model ====================
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
            nn.Conv2d(512,512,2,1,0), nn.ReLU(True),
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc  = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B,1,48,W]
        conv = self.cnn(x)                 # [B,512,1,W’]
        conv = conv.squeeze(2).permute(0,2,1)  # [B,W’,512]
        rnn_out, _ = self.rnn(conv)        # [B,W’,512]
        return self.fc(rnn_out)            # [B,W’,C]

# ==================== Utilities: file + hash ====================
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def download_with_verify(url: str, dst: str, expected_sha256: str):
    # Download to temp, verify, then move into place
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".dl_", suffix=".pt")
    os.close(fd)
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as out:
            out.write(r.read())
        actual = sha256_file(tmp)
        if expected_sha256 and actual != expected_sha256:
            os.remove(tmp)
            raise RuntimeError(
                f"Downloaded model hash mismatch.\nExpected: {expected_sha256}\nActual:   {actual}"
            )
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

# ==================== Persistence: corrections ====================
def load_corrections():
    try:
        if os.path.exists(CORRECTIONS_FILE):
            with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def save_corrections(corrections_list):
    try:
        with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(corrections_list, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save corrections: {e}")

def build_replacement_maps(corrections_list):
    """
    Returns:
      - whole_line_map: dict[original_line] -> corrected_line
      - token_map: dict[token] -> corrected_token (most frequent 1:1)
    """
    whole_line_map = {}
    token_counts = {}
    for item in corrections_list:
        if not isinstance(item, dict): continue
        orig = item.get("original"); corr = item.get("corrected")
        if isinstance(orig, str) and isinstance(corr, str) and orig.strip():
            whole_line_map[orig] = corr
            o_toks = orig.split(); c_toks = corr.split()
            if len(o_toks) == len(c_toks):
                for a, b in zip(o_toks, c_toks):
                    if a != b and len(a) > 1 and b:
                        token_counts.setdefault(a, {})
                        token_counts[a][b] = token_counts[a].get(b, 0) + 1
    token_map = {}
    for tok, repls in token_counts.items():
        best = max(repls.items(), key=lambda kv: kv[1])[0]
        token_map[tok] = best
    return whole_line_map, token_map

def apply_auto_corrections(text: str, whole_line_map, token_map):
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

def build_user_dictionary_text(corrections_list):
    vocab = set()
    for item in corrections_list:
        corr = item.get("corrected", "")
        for tok in corr.split():
            tok = tok.strip().strip(".,;:!?()[]{}\"'“”’‘")
            if tok and any(ch.isalpha() for ch in tok):
                vocab.add(tok)
    return "\n".join(sorted(vocab, key=lambda s: s.lower()))

# ==================== Preprocess & decode ====================
def preprocess_img(img: Image.Image):
    img = img.convert("L")
    # Auto invert if background darker than text
    if ImageOps.invert(img).getextrema()[0] < img.getextrema()[0]:
        img = ImageOps.invert(img)
    w, h = img.size
    new_w = int(w * (48 / h))
    img = img.resize((min(new_w, 1024), 48), Image.BILINEAR)
    if img.size[0] < 1024:
        canvas = Image.new("L", (1024, 48), 255)
        canvas.paste(img, (0, 0))
        img = canvas
    x = T.ToTensor()(img)
    x = T.Normalize((0.5,), (0.5,))(x)
    return x

def greedy_decode(logits, charset):
    preds = logits.softmax(2).argmax(2)  # [B,W’]
    out, last = [], -1
    seq = preds[0].cpu().numpy().tolist()
    for p in seq:
        if p != last and p != 0:
            out.append(charset[p])
        last = p
    return "".join(out)

def recognize_line(img: Image.Image, model, charset, device="cpu"):
    x = preprocess_img(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return greedy_decode(logits, charset)

# ==================== Page → line segmentation ====================
def segment_lines(pil_img: Image.Image, min_line_height: int = 8):
    gray = np.array(pil_img.convert("L"))
    # binarize (dark strokes)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hor = thr.sum(axis=1)
    lines, start = [], None
    for i, v in enumerate(hor):
        if v > 0 and start is None:
            start = i
        elif v == 0 and start is not None:
            if i - start >= min_line_height:
                lines.append((start, i))
            start = None
    if start is not None:
        lines.append((start, gray.shape[0]))

    # crop & also compute x extents for nicer boxes
    crops, boxes = [], []
    for (s, e) in lines:
        strip = thr[s:e, :]  # inverted binary (text=white)
        cols = strip.sum(axis=0)
        xs = np.where(cols > 0)[0]
        if xs.size == 0:
            continue
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        crop = pil_img.crop((x0, s, x1, e))
        crops.append(crop)
        boxes.append((x0, s, x1, e))
    return crops, boxes

def draw_boxes(pil_img: Image.Image, boxes, width=2):
    overlay = pil_img.convert("RGB").copy()
    d = ImageDraw.Draw(overlay)
    for (x0, y0, x1, y1) in boxes:
        for t in range(width):
            d.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=(255, 0, 0))
    return overlay

# ==================== Model loading (with cache) ====================
@st.cache_resource
def load_model(local_path: str):
    ckpt = torch.load(local_path, map_location="cpu")
    charset = ckpt.get("charset")
    if not isinstance(charset, (list, tuple)):
        raise RuntimeError("Checkpoint missing 'charset' list.")
    model = CRNN(len(charset))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, charset

def ensure_model_available(model_key: str):
    entry = MODELS[model_key]
    local = entry["local"]
    if not os.path.exists(local):
        st.info(f"Downloading {entry['label']}…")
        download_with_verify(entry["url"], local, entry["sha256"])
        st.success(f"Downloaded {entry['label']}")
    else:
        # verify existing file
        try:
            actual = sha256_file(local)
            if entry["sha256"] and actual != entry["sha256"]:
                st.warning("Existing model file hash mismatch. Re-downloading…")
                os.remove(local)
                download_with_verify(entry["url"], local, entry["sha256"])
                st.success(f"Re-downloaded {entry['label']}")
        except Exception as e:
            st.warning(f"Model verify failed: {e}. Re-downloading…")
            try:
                if os.path.exists(local): os.remove(local)
            except Exception:
                pass
            download_with_verify(entry["url"], local, entry["sha256"])
            st.success(f"Re-downloaded {entry['label']}")
    return local

# ==================== UI ====================
st.title("📝 Handwriting OCR (CRNN)")
st.caption("Multi-line page segmentation • Rotation • Auto-correct from saved edits • Model switcher with verified download")

# Sidebar controls
model_choice = st.sidebar.selectbox(
    "Model",
    options=[("v1a", MODELS["v1a"]["label"]), ("v1b", MODELS["v1b"]["label"])],
    format_func=lambda kv: kv[1],
    index=0
)[0]

rotate_choice = st.sidebar.radio("Rotate image", ["0°", "90°", "180°", "270°"], index=0)
auto_correct_on = st.sidebar.toggle("Auto-correct using saved corrections", value=True)
show_boxes = st.sidebar.toggle("Show line boxes", value=True)

# Sidebar: Corrections tools
st.sidebar.markdown("### Corrections")
corrs = load_corrections()
if not corrs:
    st.sidebar.caption("No corrections saved yet.")
else:
    for c in corrs[-50:][::-1]:
        o = c.get("original", "")
        a = c.get("corrected", "")
        st.sidebar.write(f"• **{o}** → {a}")

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
uploaded_corr = st.sidebar.file_uploader("⬆️ Import corrections.json", type=["json"], key="import_corr")
if uploaded_corr is not None:
    try:
        data = json.load(uploaded_corr)
        if isinstance(data, list):
            save_corrections(data)
            st.sidebar.success(f"Imported {len(data)} corrections.")
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid format: expected a JSON list of {original, corrected}.")
    except Exception as e:
        st.sidebar.error(f"Import failed: {e}")

# Build & download user dictionary from corrections
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

# Main: upload & run
uploaded = st.file_uploader("Upload a handwriting image", type=["jpg", "jpeg", "png"])
if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    # Rotation
    if rotate_choice == "90°":
        pil_img = pil_img.rotate(-90, expand=True)
    elif rotate_choice == "180°":
        pil_img = pil_img.rotate(180, expand=True)
    elif rotate_choice == "270°":
        pil_img = pil_img.rotate(-270, expand=True)

    # Ensure model exists, then load cached
    try:
        local_model_path = ensure_model_available(model_choice)
        model, charset = load_model(local_model_path)
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

    # Segment
    lines, boxes = segment_lines(pil_img)
    # If the image is short, treat as single line
    if pil_img.height < 120 or len(lines) == 0:
        lines = [pil_img]
        boxes = []

    # Overlay preview
    if show_boxes and boxes:
        st.image(draw_boxes(pil_img, boxes), caption="Detected line boxes", use_container_width=True)
    else:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    # Auto-correct maps
    whole_line_map, token_map = build_replacement_maps(corrs) if auto_correct_on else ({}, {})

    # Recognize each line
    st.subheader("OCR Result")
    edited_keys = []
    outputs = []
    for i, line in enumerate(lines, 1):
        raw = recognize_line(line, model, charset)
        final = apply_auto_corrections(raw, whole_line_map, token_map) if auto_correct_on else raw

        with st.expander(f"Line {i}", expanded=True):
            st.image(line, caption=f"Line {i} crop", use_container_width=True)
            st.text(f"Raw: {raw}")
            key = f"edit_{i}"
            edited = st.text_input("Auto/Final (editable):", value=final, key=key)
            outputs.append(edited)
            edited_keys.append((raw, edited))

    st.markdown("---")
    st.text_area("Full text", value="\n".join(outputs), height=220)

    # Save all corrections
    if st.button("💾 Save all edited lines as corrections"):
        new_corr = []
        for raw, edited in edited_keys:
            if isinstance(edited, str) and isinstance(raw, str) and edited != raw and raw.strip():
                new_corr.append({"original": raw, "corrected": edited})
        if new_corr:
            all_corr = load_corrections()
            all_corr.extend(new_corr)
            save_corrections(all_corr)
            st.success(f"Saved {len(new_corr)} correction(s) ✅")
        else:
            st.info("No changes to save.")
