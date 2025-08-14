# app.py
import os, io, json, time, tempfile, hashlib, platform, shutil, zipfile, subprocess, sys, textwrap, re, csv
from datetime import datetime

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import pytesseract

# ========= Streamlit base config & upload size =========
st.set_page_config(page_title="Journalist's OCR Tool", page_icon="📝", layout="wide")
st.set_option("server.maxUploadSize", 1024)  # MB

# ========= Tesseract auto-detection =========
def auto_configure_tesseract():
    """Find tesseract across macOS/Linux/Windows and configure pytesseract."""
    candidates = []
    on_path = shutil.which("tesseract")
    if on_path:
        candidates.append(on_path)
    system = platform.system()
    if system == "Windows":
        candidates += [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    elif system == "Darwin":
        candidates += ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]
    else:
        candidates += ["/usr/bin/tesseract", "/usr/local/bin/tesseract", "/snap/bin/tesseract"]
    for p in candidates:
        if p and os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return p
    return None

_TESS_PATH = auto_configure_tesseract()

# ========= Paths =========
DATA_DIR         = "ocr_data"
CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")
PATTERNS_PATH    = os.path.join(DATA_DIR, "learned_patterns.json")
USER_WORDS_PATH  = os.path.join(DATA_DIR, "user_words.txt")
MODEL_PATH       = "crnn_handwriting.pt"

# ========= Robust JSON (“Jason”) layer =========
SCHEMA_VERSION = 1
def _now_iso(): return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
def _wrap(data): return {"_meta": {"schema": SCHEMA_VERSION, "saved_at": _now_iso()}, "data": data}
def _unwrap(payload, default):
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data", default), payload.get("_meta", {})
    return payload, {"schema": 0, "saved_at": None}
def _migrate(payload):
    if not isinstance(payload, dict):
        return {"_meta": {"schema": SCHEMA_VERSION, "saved_at": _now_iso()}, "data": payload}
    data = payload.get("data", payload)
    return {"_meta": {"schema": SCHEMA_VERSION, "saved_at": _now_iso()}, "data": data}
def _atomic_write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dir_)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
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
            payload = _migrate(payload); _atomic_write(path, payload)
        return _unwrap(payload, default)
    except Exception:
        return default, {"schema": 0, "saved_at": None}
def save_json_versioned(path, data): _atomic_write(path, _wrap(data))

# ========= Helpers =========
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
    """Simple word mapping learning."""
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

# ========= Orientation helpers =========
def rotate_image(pil_img: Image.Image, mode: str) -> Image.Image:
    """Rotate according to user selection."""
    if mode == "None":
        return pil_img
    if mode == "Auto (landscape→portrait)":
        return pil_img.rotate(90, expand=True) if pil_img.width > pil_img.height else pil_img
    if mode == "90° CW":
        return pil_img.rotate(-90, expand=True)
    if mode == "90° CCW":
        return pil_img.rotate(90, expand=True)
    if mode == "180°":
        return pil_img.rotate(180, expand=True)
    return pil_img

# ========= Image preprocessing & OCR =========
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

def extract_text_tesseract(image, enhancement_level="medium"):
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

# ========= Handwriting line segmentation (for CRNN) =========
def segment_lines(pil_img, min_line_height=12, gap_thresh=6):
    """
    Split a page into line crops using horizontal projection.
    Returns a list of PIL images (one per line) in top→bottom order.
    """
    img = np.array(pil_img.convert("L"))
    bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)
    proj = bw.sum(axis=1)
    lines = []
    h = img.shape[0]
    in_run = False
    start = 0
    for y in range(h):
        if proj[y] > 0 and not in_run:
            in_run = True; start = y
        elif proj[y] == 0 and in_run:
            end = y
            if end - start >= min_line_height:
                lines.append((start, end))
            in_run = False
    if in_run:
        end = h
        if end - start >= min_line_height:
            lines.append((start, end))
    merged = []
    for s, e in lines:
        if not merged: merged.append([s, e]); continue
        ps, pe = merged[-1]
        if s - pe <= gap_thresh:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    crops = []
    for s, e in merged:
        pad = 3
        s2 = max(0, s - pad); e2 = min(h, e + pad)
        line = img[s2:e2, :]
        col_proj = (255 - line).sum(axis=0)
        xs = np.where(col_proj > 0)[0]
        if xs.size == 0: continue
        x0, x1 = xs[0], xs[-1] + 1
        line = line[:, x0:x1]
        crops.append(Image.fromarray(line))
    return crops

# ========= Persistence =========
def persist_all():
    save_json_versioned(CORRECTIONS_PATH, st.session_state.corrections)
    save_json_versioned(PATTERNS_PATH,    st.session_state.learned_patterns)
    rebuild_user_words()

# ========= Optional: auto-write a trainer script if missing =========
TRAIN_SCRIPT_PATH = "train_crnn.py"
TRAIN_SCRIPT = textwrap.dedent("""\
    import os, csv, math, string, argparse
    from PIL import Image
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T

    CHARS = string.ascii_letters + string.digits + " .,;:!?()-'\"/"
    char2idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is CTC blank
    BLANK = 0

    def _read_labels_flex(path):
        encodings = ["utf-8-sig","utf-8","utf-16","latin-1"]
        delims = [",",";","\\t","|"]
        raw = None
        used = "utf-8"
        for enc in encodings:
            try:
                with open(path,"r",encoding=enc,newline="") as f: raw = f.read(); used=enc; break
            except: pass
        if raw is None: raise RuntimeError("Could not read labels with common encodings")
        sample = raw[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",".join(delims)); delim=dialect.delimiter
        except: delim=","
        pairs=[]
        def parse_with(d):
            nonlocal pairs; pairs=[]
            reader = csv.reader(raw.splitlines(), delimiter=d, quotechar='"', escapechar="\\")
            for row in reader:
                if not row: continue
                first = row[0].strip().lower()
                if first in {"filename","file","image"}: continue
                if len(row) >= 2:
                    fn = row[0].strip(); txt = (d.join(row[1:]) if d!="," else ",".join(row[1:])).strip()
                else:
                    ok=False
                    for alt in [";", "\\t", "|", ","]:
                        if alt in row[0]:
                            parts=row[0].split(alt,1); fn=parts[0].strip(); txt=parts[1].strip() if len(parts)>1 else ""; ok=True; break
                    if not ok: continue
                if fn and txt: pairs.append((fn, txt))
        parse_with(delim)
        if not pairs:
            for d in delims:
                if d==delim: continue
                parse_with(d)
                if pairs: break
        if not pairs: raise RuntimeError("No labeled items parsed from CSV.")
        print(f"[labels] parsed {len(pairs)} from {path} using enc={used}")
        return pairs

    class LineDataset(Dataset):
        def __init__(self, root, img_dir="images", labels="labels.csv", img_h=32, max_w=512):
            self.root = root
            self.img_dir = os.path.join(root, img_dir)
            labels_path = os.path.join(root, labels)
            if not os.path.exists(labels_path): raise FileNotFoundError(labels_path)
            self.items = _read_labels_flex(labels_path)
            if len(self.items)==0: raise RuntimeError(f"No labeled items in {labels_path}")
            self.img_h=img_h; self.max_w=max_w; self.to_tensor=T.ToTensor()
        def _resize_keep_ratio(self, img: Image.Image):
            w,h = img.size; new_h=self.img_h; new_w=int(w*(new_h/h))
            img = img.convert("L").resize((min(new_w,self.max_w), new_h), Image.BILINEAR)
            if img.size[0] < self.max_w:
                canvas = Image.new("L",(self.max_w,new_h),255); canvas.paste(img,(0,0)); img=canvas
            return img
        def __len__(self): return len(self.items)
        def __getitem__(self, idx):
            fn, text = self.items[idx]
            img = Image.open(os.path.join(self.img_dir, fn)).convert("L")
            img = self._resize_keep_ratio(img)
            return self.to_tensor(img), text

    def encode_text(s, char2idx={c:i+1 for i,c in enumerate(CHARS)}):
        return torch.tensor([char2idx[c] for c in s if c in char2idx], dtype=torch.long)

    class CRNN(nn.Module):
        def __init__(self, n_classes=len(CHARS)+1, cnn_out=256, rnn_hidden=256):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(128,256,3,1,1), nn.ReLU(),
                nn.Conv2d(256,256,3,1,1), nn.ReLU(), nn.MaxPool2d((2,1),(2,1)),
                nn.Conv2d(256,512,3,1,1), nn.ReLU(), nn.BatchNorm2d(512),
                nn.Conv2d(512,512,3,1,1), nn.ReLU(), nn.MaxPool2d((2,1),(2,1)),
                nn.Conv2d(512,cnn_out,2,1,0), nn.ReLU(),
            )
            self.rnn = nn.LSTM(cnn_out, rnn_hidden, num_layers=2, bidirectional=True, batch_first=False)
            self.fc = nn.Linear(rnn_hidden*2, n_classes)
        def forward(self, x):
            f = self.cnn(x).squeeze(2)  # [N,C,W']
            f = f.permute(2,0,1)        # [W',N,C]
            y,_ = self.rnn(f); return self.fc(y)

    def collate(batch):
        imgs, texts = zip(*batch)
        imgs = torch.stack(imgs, 0)
        targets = [encode_text(t) for t in texts]
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets = torch.cat(targets) if targets else torch.tensor([], dtype=torch.long)
        return imgs, texts, targets, target_lengths

    def train_one_epoch(model, loader, opt, crit, device):
        model.train(); total=0.0
        for imgs,_,targets,targ_lens in loader:
            imgs = imgs.to(device); opt.zero_grad()
            logits = model(imgs); T_,N,C = logits.size()
            in_lens = torch.full((N,), T_, dtype=torch.long)
            loss = crit(logits.log_softmax(2), targets, in_lens, targ_lens)
            loss.backward(); opt.step(); total += loss.item()
        return total/max(1,len(loader))
    @torch.no_grad()
    def eval_loss(model, loader, crit, device):
        model.eval(); total=0.0
        for imgs,_,targets,targ_lens in loader:
            imgs = imgs.to(device)
            logits = model(imgs); T_,N,C = logits.size()
            in_lens = torch.full((N,), T_, dtype=torch.long)
            loss = crit(logits.log_softmax(2), targets, in_lens, targ_lens)
            total += loss.item()
        return total/max(1,len(loader))

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--train", default="dataset/train")
        ap.add_argument("--val",   default="dataset/val")
        ap.add_argument("--epochs", type=int, default=20)
        ap.add_argument("--batch",  type=int, default=16)
        ap.add_argument("--lr",     type=float, default=1e-3)
        ap.add_argument("--out",    default="crnn_handwriting.pt")
        args = ap.parse_args()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_ds = LineDataset(args.train); val_ds = LineDataset(args.val)
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=2)
        val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=2)

        model = CRNN().to(device)
        crit  = nn.CTCLoss(blank=BLANK, zero_infinity=True)
        opt   = optim.AdamW(model.parameters(), lr=args.lr)

        best = math.inf
        for ep in range(1, args.epochs+1):
            tr = train_one_epoch(model, train_dl, opt, crit, device)
            va = eval_loss(model, val_dl, crit, device)
            print(f"Epoch {ep}/{args.epochs}: train {tr:.4f} | val {va:.4f}")
            if va < best:
                best = va
                torch.save({"model": model.state_dict(), "charset": CHARS}, args.out)
                print(f"  ✓ Saved best model to {args.out}")
        print("Training complete.")

    if __name__ == "__main__": main()
""")
def ensure_trainer_script():
    if not os.path.exists(TRAIN_SCRIPT_PATH):
        with open(TRAIN_SCRIPT_PATH, "w", encoding="utf-8") as f:
            f.write(TRAIN_SCRIPT)

# ========= Session state =========
ensure_data_dir()
if "ocr_results" not in st.session_state: st.session_state.ocr_results = []
if "corrections" not in st.session_state:
    st.session_state.corrections, _ = load_json_versioned(CORRECTIONS_PATH, {})
if "learned_patterns" not in st.session_state:
    st.session_state.learned_patterns, _ = load_json_versioned(PATTERNS_PATH, {})
rebuild_user_words()

# ========= Sidebar =========
st.sidebar.header("Settings")
# Tesseract status
if _TESS_PATH:
    st.sidebar.caption(f"✅ Tesseract detected: {_TESS_PATH}")
else:
    st.sidebar.warning(
        "Tesseract not found. Install it and restart the app.\n\n"
        "macOS: brew install tesseract\n"
        "Ubuntu/Debian: sudo apt install tesseract-ocr\n"
        "Windows: UB Mannheim builds; add to PATH."
    )

enhancement_level = st.sidebar.selectbox("Enhancement Level", ["light","medium","aggressive"], index=1)

# Orientation control
orientation_mode = st.sidebar.selectbox(
    "Orientation",
    ["Auto (landscape→portrait)", "None", "90° CW", "90° CCW", "180°"],
    index=0
)

# CRNN toggle appears only when model exists
model_exists = os.path.exists(MODEL_PATH)
if model_exists:
    use_crnn = st.sidebar.toggle("Use CRNN (experimental)", value=True)
else:
    use_crnn = False
    st.sidebar.info("No CRNN model found yet. Train one in the panel below, then toggle this on.")

# Load CRNN when needed
crnn_ready = False
if use_crnn:
    try:
        import importlib.util, torch, torchvision
        ensure_trainer_script()
        spec = importlib.util.spec_from_file_location("train_crnn", TRAIN_SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

        @st.cache_resource
        def load_crnn():
            ckpt = torch.load(MODEL_PATH, map_location="cpu")
            model = mod.CRNN()
            model.load_state_dict(ckpt["model"])
            model.eval()
            charset = ckpt.get("charset", getattr(mod, "CHARS", ""))
            idx2char = {i+1: c for i, c in enumerate(charset)}
            return model, charset, idx2char

        crnn_model, crnn_charset, crnn_idx2char = load_crnn()
        crnn_ready = True

        def recognize_line(img_pil):
            import torchvision.transforms as T
            img = img_pil.convert("L")
            w, h = img.size
            img_h, max_w = 32, 512
            new_w = int(w * (img_h / h))
            img = img.resize((min(new_w, max_w), img_h))
            if img.size[0] < max_w:
                canvas = Image.new("L", (max_w, img_h), 255)
                canvas.paste(img, (0, 0))
                img = canvas
            x = T.ToTensor()(img).unsqueeze(0)  # [1,1,H,W]
            with torch.no_grad():
                logits = crnn_model(x)           # [T,N,C]
                best = logits.argmax(dim=2).permute(1,0)[0].tolist()
            out, prev = [], -1
            BLANK = 0
            for t in best:
                if t != prev and t != BLANK:
                    out.append(crnn_idx2char.get(t, ""))
                prev = t
            return "".join(out)

        st.sidebar.caption("🧠 CRNN model loaded.")
    except Exception as e:
        st.sidebar.warning(f"CRNN unavailable: {e}")
        use_crnn = False

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

# ========= Main =========
st.title("📝 Journalist's OCR Tool")
st.markdown("*Handwriting-first OCR — CRNN per-line + learned corrections; rotate pages when needed.*")

uploaded_files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} files uploaded!")
    if st.button("🔍 Process Images", type="primary"):
        progress = st.progress(0)
        for i, uf in enumerate(uploaded_files):
            try:
                image = Image.open(uf)
                image = ImageOps.exif_transpose(image)         # respect camera EXIF
                image = rotate_image(image, orientation_mode)  # apply orientation option
                key = make_image_key(uf.name, image)

                if use_crnn and crnn_ready:
                    # Segment to lines and recognize with CRNN
                    lines = segment_lines(image) or [image]
                    recognized = []
                    for ln in lines:
                        txt = recognize_line(ln)
                        recognized.append(txt.strip())
                    raw_text = "\n".join([t for t in recognized if t])
                    proc_img = preprocess_image(image, enhancement_level)  # for display
                    cfg = "crnn-lines"
                else:
                    raw_text, proc_img, cfg = extract_text_tesseract(image, enhancement_level)

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
            st.image(r["image"], caption="Original (after orientation)", use_container_width=True)
        with col2:
            st.text_area("Current OCR Result:", value=r["text"], height=180, disabled=True, key=f"cur_{i}")
        with st.form(f"learn_{i}"):
            st.write("**Correct any mistakes and submit to improve future OCR:**")
            corrected = st.text_area("Corrected Text:", value=r["text"], height=320, key=f"corr_{i}")
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

# ========= In-App CRNN Trainer (no terminal) =========
st.markdown("## 🧪 Experimental: In-App CRNN Trainer")
with st.expander("Open Trainer", expanded=False):
    st.write("Upload a **ZIP** containing `dataset/train/images`, `dataset/train/labels.csv`, "
             "`dataset/val/images`, and `dataset/val/labels.csv` (CSV rows: `filename,text`).")

    z = st.file_uploader("Dataset ZIP", type=["zip"])
    colA, colB, colC = st.columns(3)
    epochs = colA.number_input("Epochs", 1, 200, 20)
    batch  = colB.number_input("Batch size", 1, 128, 16)
    lr     = colC.number_input("Learning rate", 0.00001, 0.1, 0.001, format="%.5f")

    if z is not None:
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall("./")
            st.success("✅ Dataset extracted to ./dataset/")
            # quick structure check
            missing = []
            for path in ["dataset/train/images", "dataset/train/labels.csv",
                         "dataset/val/images",   "dataset/val/labels.csv"]:
                if not os.path.exists(path):
                    missing.append(path)
            if missing:
                st.error("Your ZIP is missing:\n" + "\n".join(f"- {p}" for p in missing))
        except Exception as e:
            st.error(f"Failed to extract ZIP: {e}")

    if st.button("🚀 Train CRNN Now"):
        ensure_trainer_script()
        try:
            cmd = [
                sys.executable, "train_crnn.py",
                "--train", "dataset/train",
                "--val", "dataset/val",
                "--epochs", str(int(epochs)),
                "--batch", str(int(batch)),
                "--lr", str(float(lr)),
                "--out", MODEL_PATH,
            ]
            st.write("Running:", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            st.code(proc.stdout or "(no stdout)")
            if proc.returncode != 0:
                st.error(proc.stderr or "Training failed.")
            else:
                st.success(f"🎉 Training complete. Model saved as {MODEL_PATH}")
                try:
                    with open(MODEL_PATH, "rb") as f:
                        st.download_button("⬇️ Download CRNN model", data=f.read(),
                                           file_name=os.path.basename(MODEL_PATH), mime="application/octet-stream")
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Training error: {e}")
            st.info("Tip: Ensure `torch` and `torchvision` are installed in this environment.")

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
st.markdown("*💡 Tip: Use 'Auto (landscape→portrait)' if photos are sideways. Train CRNN on more lines for steady gains.*")
