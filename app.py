# app.py
import os, io, json, time, tempfile, hashlib, platform, shutil, zipfile, subprocess, sys, textwrap
from datetime import datetime

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import pytesseract

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
DATA_DIR        = "ocr_data"
CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")
PATTERNS_PATH    = os.path.join(DATA_DIR, "learned_patterns.json")
USER_WORDS_PATH  = os.path.join(DATA_DIR, "user_words.txt")

# ========= Robust JSON (“Jason”) layer =========
SCHEMA_VERSION = 1

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _wrap(data):
    return {"_meta": {"schema": SCHEMA_VERSION, "saved_at": _now_iso()}, "data": data}

def _unwrap(payload, default):
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data", default), payload.get("_meta", {})
    return payload, {"schema": 0, "saved_at": None}

def _migrate(payload):
    # No migrations yet; pass-through to current schema
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
            payload = _migrate(payload)
            _atomic_write(path, payload)
        return _unwrap(payload, default)
    except Exception:
        return default, {"schema": 0, "saved_at": None}

def save_json_versioned(path, data):
    _atomic_write(path, _wrap(data))

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

# ========= Optional: Auto-write a trainer script if missing =========
TRAIN_SCRIPT_PATH = "train_crnn.py"
TRAIN_SCRIPT = textwrap.dedent("""\
    import os, csv, string, argparse, math
    from PIL import Image
    import torch, torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T

    CHARS = string.ascii_letters + string.digits + " .,;:!?()-'\"/"
    char2idx = {c: i+1 for i, c in enumerate(CHARS)}  # 0=CTC blank
    idx2char = {i+1: c for i, c in enumerate(CHARS)}
    BLANK = 0

    class LineDataset(Dataset):
        def __init__(self, root, img_dir="images", labels="labels.csv", img_h=32, max_w=512):
            self.img_dir = os.path.join(root, img_dir)
            self.items = []
            with open(os.path.join(root, labels), encoding="utf-8") as f:
                for fn, text in csv.reader(f):
                    self.items.append((fn, text))
            self.img_h, self.max_w = img_h, max_w
            self.to_tensor = T.ToTensor()

        def _resize_keep_ratio(self, img: Image.Image):
            w, h = img.size
            new_w = int(w * (self.img_h / h))
            img = img.convert("L").resize((min(new_w, self.max_w), self.img_h), Image.BILINEAR)
            if img.size[0] < self.max_w:
                canvas = Image.new("L", (self.max_w, self.img_h), 255)
                canvas.paste(img, (0, 0))
                img = canvas
            return img

        def __len__(self): return len(self.items)
        def __getitem__(self, idx):
            fn, text = self.items[idx]
            img = Image.open(os.path.join(self.img_dir, fn)).convert("L")
            img = self._resize_keep_ratio(img)
            return self.to_tensor(img), text

    def encode_text(s):  return torch.tensor([char2idx[c] for c in s if c in char2idx], dtype=torch.long)

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
            y,_ = self.rnn(f)
            return self.fc(y)           # [T,N,C]

    def collate(batch):
        imgs, texts = zip(*batch)
        imgs = torch.stack(imgs, 0)
        targets = [encode_text(t) for t in texts]
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets = torch.cat(targets) if targets else torch.tensor([], dtype=torch.long)
        return imgs, texts, targets, target_lengths

    def train_one_epoch(model, loader, opt, crit, device):
        model.train(); total=0
        for imgs, _, targets, targ_lens in loader:
            imgs = imgs.to(device); opt.zero_grad()
            logits = model(imgs)
            T_,N,C = logits.size()
            in_lens = torch.full((N,), T_, dtype=torch.long)
            loss = crit(logits.log_softmax(2), targets, in_lens, targ_lens)
            loss.backward(); opt.step()
            total += loss.item()
        return total/max(1,len(loader))

    @torch.no_grad()
    def eval_loss(model, loader, crit, device):
        model.eval(); total=0
        for imgs, _, targets, targ_lens in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            T_,N,C = logits.size()
            in_lens = torch.full((N,), T_, dtype=torch.long)
            loss = crit(logits.log_softmax(2), targets, in_lens, targ_lens)
            total += loss.item()
        return total/max(1,len(loader))

    def main():
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--train", default="dataset/train")
        ap.add_argument("--val",   default="dataset/val")
        ap.add_argument("--epochs", type=int, default=20)
        ap.add_argument("--batch",  type=int, default=16)
        ap.add_argument("--lr",     type=float, default=1e-3)
        ap.add_argument("--out",    default="crnn_handwriting.pt")
        args = ap.parse_args()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_ds = LineDataset(args.train)
        val_ds   = LineDataset(args.val)
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate, num_workers=2)
        val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=2)

        model = CRNN().to(device)
        crit  = nn.CTCLoss(blank=0, zero_infinity=True)
        opt   = optim.AdamW(model.parameters(), lr=args.lr)

        best = math.inf
        for ep in range(1, args.epochs+1):
            tr = train_one_epoch(model, train_dl, opt, crit, device)
            va = eval_loss(model, val_dl, crit, device)
            print(f"Epoch {ep}: train {tr:.4f} | val {va:.4f}")
            if va < best:
                best = va
                torch.save({"model": model.state_dict(), "charset": CHARS}, args.out)
                print("  ✓ saved:", args.out)

    if __name__ == "__main__":
        main()
""")

def ensure_trainer_script():
    if not os.path.exists(TRAIN_SCRIPT_PATH):
        with open(TRAIN_SCRIPT_PATH, "w", encoding="utf-8") as f:
            f.write(TRAIN_SCRIPT)

# ========= Streamlit UI =========
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

# CRNN toggle
use_crnn = st.sidebar.toggle("Use CRNN (experimental)", value=False)
crnn_ready = False
crnn_error = None

if use_crnn:
    try:
        import torch, torchvision
        from types import SimpleNamespace

        @st.cache_resource
        def load_crnn():
            # Load model if available
            ckpt_path = "crnn_handwriting.pt"
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError("crnn_handwriting.pt not found. Train in the panel below.")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # Minimal CRNN structure (must match trainer)
            import importlib.util
            # Prefer using the same class from train_crnn.py if present
            if not os.path.exists("train_crnn.py"):
                ensure_trainer_script()
            spec = importlib.util.spec_from_file_location("train_crnn", "train_crnn.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            model = mod.CRNN()
            model.load_state_dict(ckpt["model"])
            model.eval()
            charset = ckpt.get("charset", getattr(mod, "CHARS", ""))
            return SimpleNamespace(model=model, charset=charset, BLANK=0, idx2char={i+1: c for i, c in enumerate(charset)})

        crnn = load_crnn()
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
                logits = crnn.model(x)           # [T,N,C]
                best = logits.argmax(dim=2).permute(1,0)[0].tolist()
            out, prev = [], -1
            for t in best:
                if t != prev and t != crnn.BLANK:
                    out.append(crnn.idx2char.get(t, ""))
                prev = t
            return "".join(out)

        st.sidebar.caption("🧠 CRNN model loaded.")
    except Exception as e:
        crnn_error = str(e)
        st.sidebar.warning(f"CRNN unavailable: {crnn_error}")
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

                if use_crnn and crnn_ready:
                    # NOTE: best with single-line crops. For multi-line pages, add a segmenter later.
                    raw_text = recognize_line(image)
                    proc_img = preprocess_image(image, enhancement_level)  # for display consistency
                    cfg = "crnn"
                else:
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

# ========= In-App CRNN Trainer (no terminal) =========
st.markdown("## 🧪 Experimental: In-App CRNN Trainer")
with st.expander("Open Trainer", expanded=False):
    st.write("Upload a **ZIP** containing `dataset/train/images`, `dataset/train/labels.csv`, "
             "`dataset/val/images`, and `dataset/val/labels.csv` (CSV format: `filename,text`).")

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
                "--out", "crnn_handwriting.pt",
            ]
            st.write("Running:", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            st.code(proc.stdout or "(no stdout)")
            if proc.returncode != 0:
                st.error(proc.stderr or "Training failed.")
            else:
                st.success("🎉 Training complete. Model saved as crnn_handwriting.pt")
                try:
                    with open("crnn_handwriting.pt", "rb") as f:
                        st.download_button("⬇️ Download CRNN model", data=f.read(),
                                           file_name="crnn_handwriting.pt", mime="application/octet-stream")
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
st.markdown("*💡 Tip: The more corrections you make, the smarter the OCR becomes for your handwriting!*")
