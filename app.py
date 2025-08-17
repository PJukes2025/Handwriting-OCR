# app.py
import os, io, json, time, tempfile, hashlib, platform, shutil, zipfile, subprocess, sys, textwrap, re, csv, glob, urllib.request
from datetime import datetime

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
import pytesseract

# ========= Streamlit base config =========
st.set_page_config(page_title="Journalist's OCR Tool", page_icon="📝", layout="wide")

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
MODELS_DIR       = "models"  # central place for downloaded/pasted/uploaded models
os.makedirs(MODELS_DIR, exist_ok=True)

CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")
PATTERNS_PATH    = os.path.join(DATA_DIR, "learned_patterns.json")
USER_WORDS_PATH  = os.path.join(DATA_DIR, "user_words.txt")

# Default model filename; may be replaced by sidebar selection
DEFAULT_MODEL_FILENAME = "crnn_handwriting.pt"
DEFAULT_MODEL_PATH     = os.path.join(".", DEFAULT_MODEL_FILENAME)

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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path) or ".")
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

# ---- sanitize learned patterns to avoid 'count' errors ----
def sanitize_learned_patterns():
    lp = st.session_state.learned_patterns
    if not isinstance(lp, dict):
        st.session_state.learned_patterns = {}
        return
    cleaned = {}
    for k, v in lp.items():
        if isinstance(v, dict):
            replacement = str(v.get("replacement", ""))
            try:
                count = int(v.get("count", 0))
            except Exception:
                count = 0
            examples = v.get("examples", [])
            if not isinstance(examples, list):
                examples = []
            cleaned[k] = {"replacement": replacement, "count": max(0, count), "examples": examples}
        elif isinstance(v, str):
            cleaned[k] = {"replacement": v, "count": 1, "examples": []}
    st.session_state.learned_patterns = cleaned

def learn_from_correction(original, corrected):
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
                    entry["replacement"] = str(cw)
                    entry["count"] = int(entry.get("count", 0)) + 1
                    if isinstance(entry.get("examples"), list):
                        entry["examples"].append({"original": original, "corrected": corrected})
                    else:
                        entry["examples"] = [{"original": original, "corrected": corrected}]

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
    items = []
    for pat, info in st.session_state.learned_patterns.items():
        if isinstance(info, dict):
            try: cnt = int(info.get("count", 0))
            except Exception: cnt = 0
            repl = str(info.get("replacement", ""))
            items.append((pat, cnt, repl))
        elif isinstance(info, str):
            items.append((pat, 1, info))
    for pat, _, repl in sorted(items, key=lambda x: -x[1]):
        if pat and repl and pat in corrected_text:
            corrected_text = corrected_text.replace(pat, repl)
    return corrected_text

# ========= Orientation helpers =========
def rotate_image(pil_img: Image.Image, mode: str) -> Image.Image:
    if mode == "None":
        return pil_img
    if mode == "Auto (keep text horizontal)":
        try:
            osd = pytesseract.image_to_osd(pil_img)
            m = re.search(r"Rotate:\s+(\d+)", osd)
            if m:
                angle = int(m.group(1)) % 360
                if angle in (90, 270):
                    return pil_img.rotate(90 if angle == 90 else -90, expand=True)
                elif angle == 180:
                    return pil_img.rotate(180, expand=True)
                else:
                    return pil_img
        except Exception:
            pass
        if pil_img.width < pil_img.height:
            return pil_img.rotate(90, expand=True)
        return pil_img
    if mode == "90° CW":
        return pil_img.rotate(-90, expand=True)
    if mode == "90° CCW":
        return pil_img.rotate(90, expand=True)
    if mode == "180°":
        return pil_img.rotate(180, expand=True)
    if mode == "Auto (landscape→portrait)":
        return pil_img.rotate(90, expand=True) if pil_img.width > pil_img.height else pil_img
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

# ========= Line segmentation with boxes (for previews) =========
def segment_lines_with_boxes(pil_img, min_line_height=12, gap_thresh=6):
    img = np.array(pil_img.convert("L"))
    h, w = img.shape[:2]
    bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)
    proj = bw.sum(axis=1)
    lines = []
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
    crops, boxes = [], []
    for s, e in merged:
        pad = 3
        s2 = max(0, s - pad); e2 = min(h, e + pad)
        strip = img[s2:e2, :]
        col_proj = (255 - strip).sum(axis=0)
        xs = np.where(col_proj > 0)[0]
        if xs.size == 0:
            continue
        x0, x1 = xs[0], xs[-1] + 1
        line = strip[:, x0:x1]
        crops.append(Image.fromarray(line))
        boxes.append((int(x0), int(s2), int(x1), int(e2)))
    return crops, boxes

def draw_boxes_on_image(pil_img, boxes, width=3):
    overlay = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for (x0, y0, x1, y1) in boxes:
        for t in range(width):
            draw.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=(255, 0, 0))
    return overlay

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

sanitize_learned_patterns()
persist_all()
rebuild_user_words()

# ========= Model fetch helpers (persistent options) =========
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch_model_if_needed(url, out_path, expected_sha256=None):
    """
    Download model from URL if out_path doesn't exist or hash mismatches.
    Returns out_path on success.
    """
    need = True
    if os.path.exists(out_path):
        if expected_sha256:
            need = (_sha256(out_path) != expected_sha256)
        else:
            need = False
    if need:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        if expected_sha256 and _sha256(tmp) != expected_sha256:
            os.remove(tmp)
            raise RuntimeError("Downloaded model hash mismatch")
        os.replace(tmp, out_path)
    return out_path

def fetch_model_zip_if_needed(url, out_path, expected_sha256=None):
    """
    Download a ZIP containing a .pt and extract the first .pt to out_path.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_zip = out_path + ".zip.tmp"
    urllib.request.urlretrieve(url, tmp_zip)
    if expected_sha256:
        if _sha256(tmp_zip) != expected_sha256:
            os.remove(tmp_zip)
            raise RuntimeError("Downloaded ZIP hash mismatch")
    with zipfile.ZipFile(tmp_zip) as zf:
        pt_members = [m for m in zf.namelist() if m.lower().endswith(".pt")]
        if not pt_members:
            os.remove(tmp_zip)
            raise RuntimeError("ZIP does not contain a .pt file")
        member = pt_members[0]
        tmp_pt = out_path + ".tmp"
        with zf.open(member) as src, open(tmp_pt, "wb") as dst:
            dst.write(src.read())
        os.replace(tmp_pt, out_path)
    os.remove(tmp_zip)
    return out_path

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

orientation_mode = st.sidebar.selectbox(
    "Orientation",
    ["Auto (keep text horizontal)", "None", "90° CW", "90° CCW", "180°"],
    index=0
)

show_line_boxes = st.sidebar.toggle("Show line boxes", value=True)

# ===== Model source & persistence options =====
st.sidebar.markdown("### Model source")
model_choice = st.sidebar.selectbox(
    "Load CRNN model from",
    ["Local file(s) in repo", "Remote URLs via Secrets (multiple)", "Remote URL via Secrets (single)", "Paste a URL", "Upload (ephemeral)"],
    index=1  # default to Secrets multiple for your case
)

LOCAL_MODEL_CANDIDATES = []
LOCAL_MODEL_CANDIDATES += sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
LOCAL_MODEL_CANDIDATES += sorted(glob.glob("*.pt"))

MODEL_PATH = DEFAULT_MODEL_PATH
model_exists = False
remote_loaded_error = None

if model_choice == "Local file(s) in repo":
    if not LOCAL_MODEL_CANDIDATES and os.path.exists(DEFAULT_MODEL_PATH):
        LOCAL_MODEL_CANDIDATES.append(DEFAULT_MODEL_PATH)
    if LOCAL_MODEL_CANDIDATES:
        selected_local = st.sidebar.selectbox("Select model", LOCAL_MODEL_CANDIDATES, index=0)
        MODEL_PATH = selected_local
        model_exists = os.path.exists(MODEL_PATH)
        st.sidebar.caption(f"📦 Using model file: {MODEL_PATH}")
        st.sidebar.caption("Tip: Commit *.pt with Git LFS so it persists across deploys.")
    else:
        st.sidebar.info("No .pt files found in repo. Add via Git LFS or use a remote URL.")
        model_exists = False

elif model_choice == "Remote URLs via Secrets (multiple)":
    models_map = st.secrets.get("models", {})
    shas_map   = st.secrets.get("models_sha256", {})
    if not models_map:
        st.sidebar.info("Define in Secrets:\n\n[models]\nname1=\"https://...pt\"\nname2=\"https://...pt\"")
        model_exists = False
    else:
        key = st.sidebar.selectbox("Select model", sorted(models_map.keys()))
        url = models_map.get(key, "")
        sha = None
        # allow matching checksum by same key if provided
        if isinstance(shas_map, dict):
            sha = shas_map.get(key)
        MODEL_PATH = os.path.join(MODELS_DIR, f"{key}.pt")
        try:
            if url:
                if url.lower().endswith(".zip"):
                    fetch_model_zip_if_needed(url, MODEL_PATH, expected_sha256=(sha or None))
                else:
                    fetch_model_if_needed(url, MODEL_PATH, expected_sha256=(sha or None))
            model_exists = os.path.exists(MODEL_PATH)
            if model_exists:
                st.sidebar.success(f"Remote model ready: {MODEL_PATH}")
        except Exception as e:
            model_exists = False
            remote_loaded_error = str(e)
            st.sidebar.error(f"Model download failed: {e}")

elif model_choice == "Remote URL via Secrets (single)":
    url = st.secrets.get("model", {}).get("url")
    sha = st.secrets.get("model", {}).get("sha256")
    MODEL_PATH = os.path.join(MODELS_DIR, "remote_crnn.pt")
    try:
        if url:
            if url.lower().endswith(".zip"):
                fetch_model_zip_if_needed(url, MODEL_PATH, expected_sha256=(sha or None))
            else:
                fetch_model_if_needed(url, MODEL_PATH, expected_sha256=(sha or None))
        model_exists = os.path.exists(MODEL_PATH)
        if model_exists:
            st.sidebar.success("Remote model available.")
        else:
            st.sidebar.info("Set secrets to auto-download the model.")
    except Exception as e:
        model_exists = False
        remote_loaded_error = str(e)
        st.sidebar.error(f"Model download failed: {e}")

elif model_choice == "Paste a URL":
    url = st.sidebar.text_input("Model URL (direct .pt or .zip)", "")
    sha = st.sidebar.text_input("Optional SHA256", "")
    MODEL_PATH = os.path.join(MODELS_DIR, "pasted_crnn.pt")
    if st.sidebar.button("Download model"):
        try:
            if url.lower().endswith(".zip"):
                fetch_model_zip_if_needed(url, MODEL_PATH, expected_sha256=(sha or None))
            else:
                fetch_model_if_needed(url, MODEL_PATH, expected_sha256=(sha or None))
            model_exists = True
            st.sidebar.success(f"Downloaded to {MODEL_PATH}")
        except Exception as e:
            model_exists = False
            remote_loaded_error = str(e)
            st.sidebar.error(f"Download failed: {e}")
    else:
        model_exists = os.path.exists(MODEL_PATH)
        if model_exists:
            st.sidebar.caption(f"Using previously downloaded: {MODEL_PATH}")

else:  # Upload (ephemeral)
    up = st.sidebar.file_uploader("Upload a .pt model (ephemeral)", type=["pt"], key="model_upload")
    MODEL_PATH = os.path.join(MODELS_DIR, "uploaded_crnn.pt")
    if up is not None:
        try:
            with open(MODEL_PATH, "wb") as f:
                f.write(up.read())
            model_exists = True
            st.sidebar.success(f"Uploaded to {MODEL_PATH}")
        except Exception as e:
            model_exists = False
            st.sidebar.error(f"Upload failed: {e}")
    else:
        model_exists = os.path.exists(MODEL_PATH)
        if model_exists:
            st.sidebar.caption(f"Using previously uploaded: {MODEL_PATH}")

# Toggle appears only when a model exists
if model_exists:
    use_crnn = st.sidebar.toggle("Use CRNN (experimental)", value=True)
else:
    use_crnn = False
    st.sidebar.info("No CRNN model available. Select a source above.")

# ========= Load CRNN if requested =========
crnn_ready = False
if use_crnn:
    try:
        import importlib.util, torch, torchvision
        ensure_trainer_script()
        spec = importlib.util.spec_from_file_location("train_crnn", TRAIN_SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

        @st.cache_resource
        def load_crnn_model(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
            model = mod.CRNN()
            model.load_state_dict(ckpt["model"])
            model.eval()
            charset = ckpt.get("charset", getattr(mod, "CHARS", ""))
            idx2char = {i+1: c for i, c in enumerate(charset)}
            return model, charset, idx2char

        crnn_model, crnn_charset, crnn_idx2char = load_crnn_model(MODEL_PATH)
        crnn_ready = True
        st.sidebar.caption(f"🧠 CRNN loaded from: {MODEL_PATH}")
    except Exception as e:
        st.sidebar.warning(f"CRNN unavailable: {e}")
        use_crnn = False

st.sidebar.markdown("### 🧠 Learning Status")
st.sidebar.metric("Saved Corrections", len(st.session_state.corrections))
st.sidebar.metric("Learned Patterns", len(st.session_state.learned_patterns))

# Hardened "Top Learned Patterns"
if st.session_state.learned_patterns:
    st.sidebar.markdown("**Top Learned Patterns:**")
    rows = []
    for pat, info in st.session_state.learned_patterns.items():
        if isinstance(info, dict):
            try: cnt = int(info.get("count", 0))
            except Exception: cnt = 0
            rep = str(info.get("replacement", ""))
        elif isinstance(info, str):
            cnt = 1; rep = info
        else:
            continue
        rows.append((pat, cnt, rep))
    for pat, cnt, rep in sorted(rows, key=lambda x: -x[1])[:5]:
        st.sidebar.caption(f'"{pat}" → "{rep}" ({cnt}x)')

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
        sanitize_learned_patterns()
        persist_all()
        st.sidebar.success("Patterns restored.")
    except Exception as e:
        st.sidebar.error(f"Restore failed: {e}")

# ========= OCR engine dispatch =========
def extract_text_tesseract_or_crnn(pil_image):
    """Run the selected OCR engine on a PIL image and return (raw_text, processed_img, cfg, boxes, overlay_img)."""
    if use_crnn and crnn_ready:
        import torchvision.transforms as T, torch
        def recognize_line(img_pil):
            img = img_pil.convert("L")
            w, h = img.size
            img_h, max_w = 32, 512
            new_w = int(w * (img_h / h))
            img = img.resize((min(new_w, max_w), img_h))
            if img.size[0] < max_w:
                canvas = Image.new("L", (max_w, img_h), 255)
                canvas.paste(img, (0, 0))
                img = canvas
            x = T.ToTensor()(img).unsqueeze(0)
            with torch.no_grad():
                logits = crnn_model(x)
                best = logits.argmax(dim=2).permute(1,0)[0].tolist()
            out, prev = [], -1
            BLANK = 0
            for t in best:
                if t != prev and t != BLANK:
                    out.append(crnn_idx2char.get(t, ""))
                prev = t
            return "".join(out)

        crops, boxes = segment_lines_with_boxes(pil_image)
        lines = crops or [pil_image]
        recognized = [recognize_line(ln).strip() for ln in lines]
        raw_text = "\n".join([t for t in recognized if t])
        proc_img = preprocess_image(pil_image, enhancement_level)  # for display
        cfg = "crnn-lines"
        overlay = draw_boxes_on_image(pil_image, boxes) if show_line_boxes and boxes else None
        return raw_text, proc_img, cfg, boxes, overlay
    else:
        raw_text, proc_img, cfg = extract_text_tesseract(pil_image, enhancement_level)
        return raw_text, proc_img, cfg, [], None

# ========= Main =========
st.title("📝 Journalist's OCR Tool")
engine_badge = "CRNN" if (use_crnn and crnn_ready) else "Tesseract"
st.caption(f"Model status: **{engine_badge}**")

st.markdown("*Handwriting-first OCR — CRNN per-line + learned corrections; orientation keeps text horizontal. Line boxes preview enabled. Persistent model loading via repo/URL.*")

uploaded_files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} files uploaded!")
    if st.button("🔍 Process Images", type="primary"):
        progress = st.progress(0)
        for i, uf in enumerate(uploaded_files):
            try:
                image = Image.open(uf)
                image = ImageOps.exif_transpose(image)
                baseline = rotate_image(image, orientation_mode)

                key = make_image_key(uf.name, baseline)

                raw_text, proc_img, cfg, boxes, overlay = extract_text_tesseract_or_crnn(baseline)
                fixed = apply_learned_corrections(raw_text, key)

                st.session_state.ocr_results.append({
                    "filename": uf.name,
                    "image_key": key,
                    "original_text": raw_text,
                    "text": fixed,
                    "config": cfg,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "baseline_image": baseline,
                    "image": baseline,
                    "processed_image": proc_img,
                    "line_boxes": boxes,
                    "overlay_image": overlay
                })
            except Exception as e:
                st.error(f"Error processing {uf.name}: {e}")
            progress.progress((i+1)/len(uploaded_files))
        st.success("✅ Processing complete!")

# Results & learning (rotate controls + overlays)
if st.session_state.ocr_results:
    st.header("📋 Results & Learning")
    for i, r in enumerate(st.session_state.ocr_results):
        st.markdown("---")
        st.subheader(f"📄 {r['filename']}")

        bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns([1,1,1,1,3])
        with bcol1:
            if st.button("↩︎ 90° CCW", key=f"rot_ccw_{i}"):
                try:
                    new_img = r["image"].rotate(90, expand=True)
                    raw_text, proc_img, cfg, boxes, overlay = extract_text_tesseract_or_crnn(new_img)
                    r.update({"image": new_img, "original_text": raw_text, "text": apply_learned_corrections(raw_text, r["image_key"]),
                              "config": cfg, "processed_image": proc_img, "line_boxes": boxes, "overlay_image": overlay})
                    st.session_state.ocr_results[i] = r; st.rerun()
                except Exception as e:
                    st.error(f"Rotate CCW failed: {e}")
        with bcol2:
            if st.button("↪︎ 90° CW", key=f"rot_cw_{i}"):
                try:
                    new_img = r["image"].rotate(-90, expand=True)
                    raw_text, proc_img, cfg, boxes, overlay = extract_text_tesseract_or_crnn(new_img)
                    r.update({"image": new_img, "original_text": raw_text, "text": apply_learned_corrections(raw_text, r["image_key"]),
                              "config": cfg, "processed_image": proc_img, "line_boxes": boxes, "overlay_image": overlay})
                    st.session_state.ocr_results[i] = r; st.rerun()
                except Exception as e:
                    st.error(f"Rotate CW failed: {e}")
        with bcol3:
            if st.button("⟲ 180°", key=f"rot_180_{i}"):
                try:
                    new_img = r["image"].rotate(180, expand=True)
                    raw_text, proc_img, cfg, boxes, overlay = extract_text_tesseract_or_crnn(new_img)
                    r.update({"image": new_img, "original_text": raw_text, "text": apply_learned_corrections(raw_text, r["image_key"]),
                              "config": cfg, "processed_image": proc_img, "line_boxes": boxes, "overlay_image": overlay})
                    st.session_state.ocr_results[i] = r; st.rerun()
                except Exception as e:
                    st.error(f"Rotate 180 failed: {e}")
        with bcol4:
            if st.button("Reset to Auto", key=f"rot_reset_{i}"):
                try:
                    new_img = rotate_image(r["baseline_image"], "Auto (keep text horizontal)")
                    raw_text, proc_img, cfg, boxes, overlay = extract_text_tesseract_or_crnn(new_img)
                    r.update({"image": new_img, "original_text": raw_text, "text": apply_learned_corrections(raw_text, r["image_key"]),
                              "config": cfg, "processed_image": proc_img, "line_boxes": boxes, "overlay_image": overlay})
                    st.session_state.ocr_results[i] = r; st.rerun()
                except Exception as e:
                    st.error(f"Reset failed: {e}")
        with bcol5:
            st.caption("Rotate this image and re-run OCR instantly.")

        col1, col2 = st.columns([1, 2])
        with col1:
            if show_line_boxes and r.get("overlay_image") is not None:
                st.image(r["overlay_image"], caption="Detected line boxes", use_container_width=True)
            else:
                st.image(r["image"], caption="Current image", use_container_width=True)
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

# ========= In-App CRNN Trainer =========
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
                "--out", DEFAULT_MODEL_FILENAME,  # save in repo root for LFS if desired
            ]
            st.write("Running:", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            st.code(proc.stdout or "(no stdout)")
            if proc.returncode != 0:
                st.error(proc.stderr or "Training failed.")
            else:
                st.success(f"🎉 Training complete. Model saved as {DEFAULT_MODEL_FILENAME}")
                try:
                    with open(DEFAULT_MODEL_FILENAME, "rb") as f:
                        st.download_button("⬇️ Download CRNN model", data=f.read(),
                                           file_name=os.path.basename(DEFAULT_MODEL_FILENAME), mime="application/octet-stream")
                    try:
                        shutil.copy2(DEFAULT_MODEL_FILENAME, os.path.join(MODELS_DIR, DEFAULT_MODEL_FILENAME))
                    except Exception:
                        pass
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
st.markdown("*💡 Tip: Use **Remote URLs via Secrets (multiple)** to switch between release models without re-uploading.*")
