# app.py - Enhanced Handwriting OCR with Improved CRNN
# ---------------------------------------------------
import os, io, json, time, tempfile, hashlib, platform, shutil, zipfile, subprocess, sys, textwrap, re, csv, glob, urllib.request, contextlib, pathlib, difflib
from datetime import datetime
from typing import List, Tuple

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw

import pytesseract

# ---------- Optional SciPy smoothing ----------
try:
    from scipy.ndimage import gaussian_filter1d as _gf1d
    SCIPY_AVAILABLE = True
    def gaussian_filter1d(data, sigma=1.0): return _gf1d(data, sigma=sigma)
except Exception:
    SCIPY_AVAILABLE = False
    def gaussian_filter1d(data, sigma=1.0):
        # Fallback: simple moving average of width ~ 6*sigma
        ks = max(3, int(2 * sigma * 3))
        if ks % 2 == 0: ks += 1
        k = np.ones(ks) / ks
        return np.convolve(data, k, mode="same")

# ---------- Streamlit config ----------
st.set_page_config(page_title="Journalist’s OCR Tool", page_icon="📝", layout="wide")

# ---------- Paths & storage ----------
DATA_DIR         = "ocr_data"
MODELS_DIR       = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")
PATTERNS_PATH    = os.path.join(DATA_DIR, "learned_patterns.json")
USER_WORDS_PATH  = os.path.join(DATA_DIR, "user_words.txt")

DEFAULT_MODEL_FILENAME = "crnn_handwriting.pt"
DEFAULT_MODEL_PATH     = os.path.join(".", DEFAULT_MODEL_FILENAME)

TRAIN_SCRIPT_PATH = "train_crnn.py"  # will be created if missing (EnhancedCRNN)

# ---------- Robust JSON layer ----------
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
    fd, tmp = tempfile.mkstemp(prefix=".tmp*", dir=os.path.dirname(path) or ".")
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

# ---------- Tesseract auto-detection ----------
def auto_configure_tesseract():
    """Find tesseract across macOS/Linux/Windows and configure pytesseract."""
    candidates = []
    on_path = shutil.which("tesseract")
    if on_path: candidates.append(on_path)
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

# ---------- Ensure trainer script exists (EnhancedCRNN) ----------
ENHANCED_TRAIN_SCRIPT = textwrap.dedent("""
# train_crnn.py  — Enhanced CRNN for handwriting lines
import os, csv, string, random
from typing import List, Tuple
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

CHARS = string.ascii_letters + string.digits + " .,;:!?()-'\\\"/&@#$%"
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is CTC blank
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
BLANK = 0

def _read_labels_flex(path):
    encodings = ["utf-8-sig","utf-8","utf-16","latin-1"]
    delims = [",",";","\\t","|"]
    raw = None; used = "utf-8"
    for enc in encodings:
        try:
            with open(path,"r",encoding=enc,newline="") as f:
                raw = f.read(); used=enc; break
        except: pass
    if raw is None: raise RuntimeError("Could not read labels")
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
                        parts=row[0].split(alt,1)
                        fn=parts[0].strip(); txt=parts[1].strip() if len(parts)>1 else ""
                        ok=True; break
                if not ok: continue
            if fn and txt: pairs.append((fn, txt))
    parse_with(delim)
    if not pairs:
        for d in delims:
            if d==delim: continue
            parse_with(d)
            if pairs: break
    if not pairs: raise RuntimeError("No labeled items parsed from labels file.")
    return pairs

class HandwritingLineDataset(Dataset):
    def __init__(self, root, img_dir="images", labels="labels.csv",
                 img_h=48, max_w=1024, augment=True):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        labels_path = os.path.join(root, labels)
        if not os.path.exists(labels_path): raise FileNotFoundError(labels_path)
        self.items = _read_labels_flex(labels_path)
        if len(self.items)==0: raise RuntimeError(f"No labeled items in {labels_path}")
        self.img_h=img_h; self.max_w=max_w
        self.to_tensor=T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])
        self.augment = augment

    def _resize_center(self, img: Image.Image):
        w, h = img.size
        if h <= 0:
            return Image.new("L", (self.max_w, self.img_h), 255)
        aspect = max(1e-6, w/h)
        new_w = int(self.img_h * aspect)
        img = img.convert("L").resize((new_w, self.img_h), Image.LANCZOS)
        if new_w < self.max_w:
            canvas = Image.new("L", (self.max_w, self.img_h), 255)
            canvas.paste(img, ((self.max_w-new_w)//2, 0))
            img = canvas
        elif new_w > self.max_w:
            sx = (new_w - self.max_w)//2
            img = img.crop((sx, 0, sx + self.max_w, self.img_h))
        return img

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fn, txt = self.items[idx]
        path = os.path.join(self.img_dir, fn)
        img = Image.open(path).convert("L")
        img = self._resize_center(img)
        x = self.to_tensor(img)             # [1,H,W]
        x = self.normalize(x)               # [-1,1]
        y = [char2idx.get(c, 0) for c in txt]  # unknown -> blank (0)
        y = torch.LongTensor(y)
        return x, y, len(y)

def collate_fn(batch):
    xs, ys, lens = zip(*batch)
    xs = torch.stack(xs, dim=0)   # [N,1,H,W]
    y_concat = torch.cat(ys, dim=0)  # [sum(len)]
    y_lens = torch.IntTensor([len(y) for y in ys])
    return xs, y_concat, y_lens

class EnhancedCRNN(nn.Module):
    def __init__(self, n_classes=len(CHARS)+1, cnn_out=256, rnn_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, cnn_out, 2, padding=0), nn.ReLU(True),
        )
        self.rnn = nn.LSTM(cnn_out, rnn_hidden, num_layers=2,
                           bidirectional=True, batch_first=False, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(rnn_hidden*2, n_classes)

    def forward(self, x):
        f = self.cnn(x)            # [N,C,H',W']
        if f.size(2) > 1:
            f = f.mean(dim=2)      # [N,C,W']
        else:
            f = f.squeeze(2)
        f = f.permute(2,0,1)       # [T,N,C]
        y,_ = self.rnn(f)          # [T,N,2H]
        y = self.dropout(y)
        y = self.fc(y)             # [T,N,K]
        return y
""")

def ensure_trainer_script():
    if not os.path.exists(TRAIN_SCRIPT_PATH):
        with open(TRAIN_SCRIPT_PATH, "w", encoding="utf-8") as f:
            f.write(ENHANCED_TRAIN_SCRIPT)

# ---------- Session state ----------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USER_WORDS_PATH):
        with open(USER_WORDS_PATH, "w", encoding="utf-8") as f:
            pass

ensure_trainer_script()
ensure_data_dir()

if "ocr_results" not in st.session_state: st.session_state.ocr_results = []
if "corrections" not in st.session_state:
    st.session_state.corrections, _ = load_json_versioned(CORRECTIONS_PATH, {})
if "learned_patterns" not in st.session_state:
    st.session_state.learned_patterns, _ = load_json_versioned(PATTERNS_PATH, {})

# ---------- Utilities ----------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

def fetch_model(url, out_path, expected_sha256=None, github_token=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    req = urllib.request.Request(url)
    if github_token:
        req.add_header("Authorization", f"token {github_token}")
        req.add_header("Accept", "application/octet-stream")
    with contextlib.closing(urllib.request.urlopen(req)) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    if expected_sha256 and _sha256(tmp) != expected_sha256:
        os.remove(tmp)
        raise RuntimeError("Downloaded model hash mismatch")
    os.replace(tmp, out_path)
    return out_path

def make_image_key(filename, image: Image.Image) -> str:
    bio = io.BytesIO(); image.save(bio, format="PNG")
    digest = hashlib.md5(bio.getvalue()).hexdigest()[:8]
    return f"{filename}_{digest}"

def cleanup_token(w: str) -> str:
    return w.strip().strip('.,;:!?()[]{}"\'“”‘’')

def rebuild_user_words():
    """Build Tesseract user-words from corrections."""
    def add_from_text(t, acc):
        if not isinstance(t, str): return
        for w in t.split():
            w2 = cleanup_token(w)
            if w2 and any(ch.isalpha() for ch in w2):
                acc.add(w2)
    words = set()
    store = st.session_state.get("corrections", {})
    if isinstance(store, dict):
        for v in store.values():
            if isinstance(v, dict):
                add_from_text(v.get("corrected"), words)
            elif isinstance(v, str):
                add_from_text(v, words)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        add_from_text(item.get("corrected"), words)
                    elif isinstance(item, str):
                        add_from_text(item, words)
    elif isinstance(store, list):
        for item in store:
            if isinstance(item, dict):
                add_from_text(item.get("corrected"), words)
            elif isinstance(item, str):
                add_from_text(item, words)
    with open(USER_WORDS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(words)))

def sanitize_learned_patterns():
    lp = st.session_state.learned_patterns
    if not isinstance(lp, dict):
        st.session_state.learned_patterns = {}; return
    cleaned = {}
    for k, v in lp.items():
        if isinstance(v, dict):
            replacement = str(v.get("replacement", ""))
            try: count = int(v.get("count", 0))
            except: count = 0
            examples = v.get("examples", [])
            if not isinstance(examples, list): examples = []
            cleaned[k] = {"replacement": replacement, "count": max(0, count), "examples": examples}
        elif isinstance(v, str):
            cleaned[k] = {"replacement": v, "count": 1, "examples": []}
    st.session_state.learned_patterns = cleaned
sanitize_learned_patterns()

def learn_from_correction(original, corrected):
    if not original or not corrected or original == corrected: return
    ow, cw = original.split(), corrected.split()
    if len(ow) == len(cw):
        for a,b in zip(ow,cw):
            if a != b and len(a) > 1:
                entry = st.session_state.learned_patterns.get(a)
                if entry is None:
                    st.session_state.learned_patterns[a] = {
                        "replacement": b, "count": 1,
                        "examples": [{"original": original, "corrected": corrected}],
                    }
                else:
                    entry["replacement"] = str(b)
                    entry["count"] = int(entry.get("count", 0)) + 1
                    ex = entry.get("examples", [])
                    if isinstance(ex, list):
                        ex.append({"original": original, "corrected": corrected})
                        entry["examples"] = ex

def save_correction(image_key, original_text, corrected_text):
    if original_text == corrected_text: return False
    st.session_state.corrections[image_key] = {
        "original": original_text,
        "corrected": corrected_text,
        "timestamp": datetime.now().isoformat(),
    }
    learn_from_correction(original_text, corrected_text)
    persist_all()
    return True

def _lexicon():
    lex = set()
    for v in st.session_state.get("corrections", {}).values():
        if isinstance(v, dict):
            lex.update(w for w in v.get("corrected","").split() if w)
    for info in st.session_state.get("learned_patterns", {}).values():
        if isinstance(info, dict):
            lex.update(info.get("replacement","").split())
        elif isinstance(info, str):
            lex.update(info.split())
    return {w for w in lex if any(ch.isalpha() for ch in w)}

def apply_lexicon_pass(text: str, cutoff=0.86) -> str:
    words = text.split()
    lex = list(_lexicon())
    if not lex: return text
    out = []
    for w in words:
        cand = difflib.get_close_matches(w, lex, n=1, cutoff=cutoff)
        out.append(cand[0] if cand else w)
    return " ".join(out)

def apply_learned_corrections(text, image_key):
    if image_key in st.session_state.corrections:
        return st.session_state.corrections[image_key].get("corrected", text)
    corrected_text = text
    built_in_fixes = {
        " tlie ": " the ", " or ": " a ", "rn": "m", "cl": "d",
        "l1": "h", "wl1": "wh", "l1e": "he"
    }
    for a, b in built_in_fixes.items():
        corrected_text = corrected_text.replace(a, b)
    items = []
    for pat, info in st.session_state.learned_patterns.items():
        if isinstance(info, dict):
            try: cnt = int(info.get("count", 0))
            except: cnt = 0
            repl = str(info.get("replacement", ""))
            items.append((pat, cnt, repl))
        elif isinstance(info, str):
            items.append((pat, 1, info))
    for pat, _, repl in sorted(items, key=lambda x: -x[1]):
        if pat and repl and pat in corrected_text:
            corrected_text = corrected_text.replace(pat, repl)
    return corrected_text

def persist_all():
    save_json_versioned(CORRECTIONS_PATH, st.session_state.corrections)
    save_json_versioned(PATTERNS_PATH,    st.session_state.learned_patterns)
    rebuild_user_words()

# ---------- Preprocess / segmentation ----------
def _unsharp(gray, amount=1.0, radius=1.0):
    blur = cv2.GaussianBlur(gray, (0,0), radius)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def preprocess_image_handwriting(image: Image.Image, enhancement_level="medium"):
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if len(arr.shape) == 3 else arr
    if upscale_factor and upscale_factor > 1:
        gray = cv2.resize(gray, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
    h_val = 10 if enhancement_level == "light" else 15 if enhancement_level == "medium" else 20
    gray = cv2.fastNlMeansDenoising(gray, h=h_val)
    clip = 2.0 if enhancement_level == "light" else 2.5 if enhancement_level == "medium" else 4.0
    tile = 8 if enhancement_level in ("light","medium") else 6
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    gray = clahe.apply(gray)
    if sharpen:
        gray = _unsharp(gray, amount=0.8 if enhancement_level != "aggressive" else 1.2, radius=1.0)
    if enhancement_level == "aggressive":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,1))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    block = 15 if enhancement_level != "aggressive" else 21
    C = 8 if enhancement_level != "aggressive" else 10
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block, C)
    return processed

def deskew_gray(gray_bin_inv: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray_bin_inv, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=120)
    if lines is None: 
        return gray_bin_inv
    angles = []
    for rho, theta in lines[:,0]:
        ang = (theta - np.pi/2) * 180/np.pi
        if -20 < ang < 20:
            angles.append(ang)
    if not angles:
        return gray_bin_inv
    angle = float(np.median(angles))
    (h,w) = gray_bin_inv.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(gray_bin_inv, M, (w,h), flags=cv2.INTER_LINEAR, borderValue=0)

def segment_lines_handwriting_optimized(pil_img: Image.Image, min_line_height=20, gap_thresh=8,
                                        merge_close_lines=True):
    img = np.array(pil_img.convert("L"))
    h, w = img.shape[:2]
    binary_inv = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 8)
    binary_inv = deskew_gray(binary_inv)
    proj = binary_inv.sum(axis=1)
    proj_smooth = gaussian_filter1d(proj.astype(float), sigma=float(seg_sigma))
    thresh = proj_smooth.max() * float(seg_thresh)
    lines = []
    in_run = False; start = 0
    for y in range(h):
        if proj_smooth[y] > thresh and not in_run:
            in_run = True; start = y
        elif proj_smooth[y] <= thresh and in_run:
            end = y
            if end - start >= int(min_line_height):
                lines.append((start, end))
            in_run = False
    if in_run:
        end = h
        if end - start >= int(min_line_height):
            lines.append((start, end))
    if merge_close_lines and lines:
        merged = []
        for s, e in lines:
            if not merged: merged.append([s,e]); continue
            if s - merged[-1][1] <= int(gap_thresh):
                merged[-1][1] = e
            else:
                merged.append([s,e])
        lines = [(s,e) for s,e in merged]
    crops, boxes = [], []
    for s, e in lines:
        line_h = e - s
        v_pad = max(3, line_h // 6); h_pad = 6
        s2 = max(0, s - v_pad); e2 = min(h, e + v_pad)
        strip = img[s2:e2, :]
        col_proj = (255 - strip).sum(axis=0)
        col_s = gaussian_filter1d(col_proj.astype(float), sigma=1.2)
        xs = np.where(col_s > col_s.max() * 0.05)[0]
        if xs.size == 0: continue
        x0, x1 = max(0, xs[0]-h_pad), min(w, xs[-1]+1+h_pad)
        crops.append(Image.fromarray(strip[:, x0:x1]))
        boxes.append((int(x0), int(s2), int(x1), int(e2)))
    return crops, boxes

def prepare_line_for_crnn(img_pil, target_height=48, max_width=1024, add_padding=True):
    img = img_pil.convert("L")
    w, h = img.size
    if h == 0: return Image.new("L", (max_width, target_height), 255)
    aspect_ratio = w / max(1, h)
    new_width = int(target_height * aspect_ratio)
    if new_width > max_width:
        new_width = max_width
        new_height = int(max_width / max(1e-6, aspect_ratio))
    else:
        new_height = target_height
    img = img.resize((new_width, new_height), Image.LANCZOS)
    if add_padding:
        canvas = Image.new("L", (max_width, target_height), 255)
        paste_x = (max_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas
    return img

def draw_boxes_on_image(pil_img, boxes, width=3):
    overlay = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for (x0, y0, x1, y1) in boxes:
        for t in range(width):
            draw.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=(255, 0, 0))
    return overlay

# ---------- Orientation ----------
def rotate_image(pil_img: Image.Image, mode: str) -> Image.Image:
    if mode == "None": return pil_img
    if mode == "Auto (keep text horizontal)":
        try:
            osd = pytesseract.image_to_osd(pil_img)
            m = re.search(r"Rotate:\\s+(\\d+)", osd)
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
    if mode == "90° CW":  return pil_img.rotate(-90, expand=True)
    if mode == "90° CCW": return pil_img.rotate(90, expand=True)
    if mode == "180°":   return pil_img.rotate(180, expand=True)
    if mode == "Auto (landscape→portrait)":
        return pil_img.rotate(90, expand=True) if pil_img.width > pil_img.height else pil_img
    return pil_img

# ---------- Legacy Tesseract ----------
def extract_text_tesseract(image, enhancement_level="medium"):
    processed = preprocess_image_handwriting(image, enhancement_level)
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
    names = ["general", "line", "word"]
    for i, cfg in enumerate(configs):
        try:
            out = pytesseract.image_to_string(pil_proc, config=cfg).strip()
            if len(out) > len(best):
                best, best_cfg = out, names[i]
        except Exception:
            continue
    return best, processed, best_cfg

# ---------- Sidebar UI ----------
st.sidebar.header("🎛️ OCR Settings")

if _TESS_PATH:
    st.sidebar.success(f"✅ Tesseract: {os.path.basename(_TESS_PATH)}")
else:
    st.sidebar.error("❌ Tesseract not found. Install it and restart.")

enhancement_level = st.sidebar.selectbox(
    "📈 Enhancement Level",
    ["light", "medium", "aggressive"],
    index=1,
    help="Light: clear writing • Medium: typical • Aggressive: faint/margins",
)

orientation_mode = st.sidebar.selectbox(
    "🔄 Auto-Rotation",
    ["Auto (keep text horizontal)", "None", "90° CW", "90° CCW", "180°"],
    index=0,
    help="Auto keeps text horizontal for better OCR",
)

st.sidebar.markdown("### ✂️ Line Segmentation")
seg_sigma = st.sidebar.slider("Projection smoothing (σ)", 0.5, 4.0, 2.0, 0.1)
seg_thresh = st.sidebar.slider("Projection threshold factor", 0.02, 0.30, 0.10, 0.01,
                               help="Lower = more sensitive (more lines)")
min_line_height = st.sidebar.slider("Min line height (px)", 10, 60, 22, 1)
gap_thresh = st.sidebar.slider("Merge gap (px)", 0, 20, 8, 1)

st.sidebar.markdown("### 🔍 Preprocess")
upscale_factor = st.sidebar.select_slider("Upscale before OCR", options=[1,2,3], value=2)
sharpen = st.sidebar.toggle("Unsharp mask", value=True)

# ---------- Model selection ----------
st.sidebar.markdown("### 🧠 CRNN Model")
model_mode = st.sidebar.selectbox(
    "Model Source",
    ["Upload .pt", "Local file path", "Download from URL"],
    index=0,
)

MODEL_PATHS = []
remote_error = None

if model_mode == "Upload .pt":
    up = st.sidebar.file_uploader("📤 Upload .pt (ephemeral)", type=["pt"], key="model_upload", accept_multiple_files=True)
    if up:
        MODEL_PATHS = []
        for i, f in enumerate(up):
            dst = os.path.join(MODELS_DIR, f"uploaded_{i}.pt")
            with open(dst, "wb") as out:
                out.write(f.read())
            MODEL_PATHS.append(dst)
        st.sidebar.success(f"✅ Loaded {len(MODEL_PATHS)} model(s)")
elif model_mode == "Local file path":
    candidates = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")) + glob.glob("*.pt"))
    if candidates:
        picks = st.sidebar.multiselect("Select model(s)", candidates, default=[candidates[0]])
        MODEL_PATHS = picks
        st.sidebar.info(f"📦 Using {len(MODEL_PATHS)} model(s)")
    else:
        st.sidebar.warning("No .pt files found. Upload or download one.")
elif model_mode == "Download from URL":
    url = st.sidebar.text_input("🔗 Model URL (.pt)", "")
    sha = st.sidebar.text_input("🔒 SHA256 (optional)", "")
    token = st.sidebar.text_input("🎫 GitHub Token (if private)", type="password")
    dst = os.path.join(MODELS_DIR, "remote.pt")
    if st.sidebar.button("⬇️ Download"):
        try:
            fetch_model(url, dst, expected_sha256=(sha or None), github_token=(token or None))
            MODEL_PATHS = [dst]
            st.sidebar.success("✅ Downloaded")
        except Exception as e:
            remote_error = str(e)
            st.sidebar.error(f"❌ Failed: {e}")

use_crnn = st.sidebar.toggle("🧠 Use CRNN", value=bool(MODEL_PATHS))

# ---------- Load CRNN (via train_crnn.py) ----------
def _file_hash(path:str)->str:
    try: return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()
    except Exception: return "no-file"

crnn_ready = False
crnn_models = []
crnn_idx2char = {}

if use_crnn and MODEL_PATHS:
    try:
        import importlib.util, torch, torchvision  # noqa
        ensure_trainer_script()
        spec = importlib.util.spec_from_file_location("train_crnn", TRAIN_SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        # Compatibility shim
        if not hasattr(mod, "EnhancedCRNN"):
            if hasattr(mod, "CRNN"):
                mod.EnhancedCRNN = mod.CRNN
            else:
                raise AttributeError("train_crnn.py has neither EnhancedCRNN nor CRNN.")
        @st.cache_resource(show_spinner=False)
        def load_crnn_models(paths, script_hash: str):
            ms = []; common = None
            for p in paths:
                ckpt = torch.load(p, map_location="cpu")
                if "model" in ckpt:
                    state = ckpt["model"]; charset = ckpt.get("charset", getattr(mod, "CHARS", ""))
                else:
                    state = ckpt.get("model_state_dict", ckpt); charset = getattr(mod, "CHARS", "")
                m = mod.EnhancedCRNN()
                m.load_state_dict(state); m.eval()
                idx2char = {i+1: c for i, c in enumerate(charset)}
                if common is None:
                    common = idx2char
                ms.append(m)
            return ms, common
        script_hash = _file_hash(TRAIN_SCRIPT_PATH)
        crnn_models, crnn_idx2char = load_crnn_models(MODEL_PATHS, script_hash)
        crnn_ready = True
        st.sidebar.success(f"🧠 CRNN ready ({len(crnn_models)} model{'s' if len(crnn_models)>1 else ''})")
    except Exception as e:
        st.sidebar.error(f"❌ CRNN failed: {e}")
        use_crnn = False

# ---------- OCR dispatch ----------
def extract_text_tesseract_or_crnn(pil_image: Image.Image):
    if use_crnn and crnn_ready:
        import torchvision.transforms as T, torch  # noqa
        def recognize_line_enhanced(img_pil):
            arr = np.array(img_pil)
            processed = preprocess_image_handwriting(Image.fromarray(arr), enhancement_level)
            img = Image.fromarray(processed)
            img = prepare_line_for_crnn(img, target_height=48, max_width=1024)
            x = T.ToTensor()(img)
            x = T.Normalize(mean=[0.5], std=[0.5])(x)
            x = x.unsqueeze(0)
            with torch.no_grad():
                logits_sum = None
                for m in crnn_models:
                    logits = m(x)  # [T,N,C]
                    logits_sum = logits if logits_sum is None else logits_sum + logits
                if len(crnn_models) > 1:
                    logits_sum = logits_sum / len(crnn_models)
                best = logits_sum.argmax(dim=2).permute(1,0)[0].tolist()
            out, prev = [], -1
            BLANK = 0
            for t in best:
                if t != prev and t != BLANK:
                    ch = crnn_idx2char.get(t, "")
                    if ch: out.append(ch)
                prev = t
            return "".join(out)
        crops, boxes = segment_lines_handwriting_optimized(pil_image, min_line_height=min_line_height, gap_thresh=gap_thresh)
        lines = crops or [pil_image]
        recognized = []
        for ln in lines:
            text = recognize_line_enhanced(ln).strip()
            if text: recognized.append(text)
        raw_text = "\n".join(recognized)
        proc_img = preprocess_image_handwriting(pil_image, enhancement_level)
        overlay = draw_boxes_on_image(pil_image, boxes) if boxes else None
        return raw_text, proc_img, f"enhanced-crnn-{len(crnn_models)}", boxes, overlay
    else:
        raw_text, proc_img, cfg = extract_text_tesseract(pil_image, enhancement_level)
        return raw_text, proc_img, f"tesseract-{cfg}", [], None

# ---------- UI: header ----------
st.title("📝 Enhanced Journalist’s OCR Tool")
col1, col2, col3 = st.columns(3)
with col1:
    engine_status = "🧠 CRNN" if (use_crnn and crnn_ready) else "📖 Tesseract"
    st.metric("OCR Engine", engine_status)
with col2:
    st.metric("Enhancement", {"light":"🌟 Light","medium":"⚡ Medium","aggressive":"🔥 Aggressive"}[enhancement_level])
with col3:
    num_corr = len(st.session_state.corrections); num_pat = len(st.session_state.learned_patterns)
    st.metric("Learning", f"{num_corr} corrections • {num_pat} patterns")

st.markdown("*Advanced handwriting OCR with CRNN, intelligent line segmentation, deskew and adaptive learning from your corrections.*")

# ---------- Upload ----------
st.header("📤 Upload Handwritten Images")
uploaded_files = st.file_uploader(
    "Select handwritten document images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    help="📱 Use clear, well-lit photos. The app will auto‑orient and enhance them.",
)

# ---------- Process all ----------
if uploaded_files and st.button("🚀 Process All Images", type="primary"):
    progress = st.progress(0); status = st.empty()
    for i, uf in enumerate(uploaded_files):
        status.text(f"Processing {uf.name}... ({i+1}/{len(uploaded_files)})")
        try:
            image = Image.open(uf)
            image = ImageOps.exif_transpose(image)
            baseline = rotate_image(image, orientation_mode)
            key = make_image_key(uf.name, baseline)
            raw_text, proc_img, cfg, boxes, overlay = extract_text_tesseract_or_crnn(baseline)
            corrected_text = apply_learned_corrections(raw_text, key)
            corrected_text = apply_lexicon_pass(corrected_text)
            st.session_state.ocr_results.append({
                "filename": uf.name,
                "image_key": key,
                "original_text": raw_text,
                "text": corrected_text,
                "config": cfg,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image": baseline,
                "processed_image": proc_img,
                "line_boxes": boxes,
                "overlay_image": overlay,
                "enhancement_level": enhancement_level,
                "orientation_mode": orientation_mode
            })
        except Exception as e:
            st.error(f"❌ Error processing {uf.name}: {e}")
        progress.progress((i+1)/len(uploaded_files))
    status.text("✅ All images processed!")
    st.success("🎉 Processing complete! Scroll down to review and correct results.")

# ---------- Results & Learning ----------
if st.session_state.ocr_results:
    st.header("📋 OCR Results & Learning")
    for i, r in enumerate(st.session_state.ocr_results):
        st.markdown("---")
        col_h1, col_h2 = st.columns([3,1])
        with col_h1:
            st.subheader(f"📄 {r['filename']}")
            st.caption(f"⚙️ {r['config']} • 📅 {r['timestamp']} • 🎛️ {r.get('enhancement_level','medium')}")
        with col_h2:
            if r.get("original_text") and r.get("text"):
                # Very rough similarity
                o = set(r["original_text"].split()); c = set(r["text"].split())
                sim = len(o & c) / max(1, len(o))
                st.metric("✨ Similarity", f"{sim:.0%}")

        imgcol, txtcol = st.columns([1,1])
        with imgcol:
            st.image(r["overlay_image"] or r["image"], caption="Image (line boxes shown in red)" if r["overlay_image"] else "Image", use_column_width=True)
            # quick rotation re-run
            b1,b2,b3 = st.columns(3)
            if b1.button("↩︎ 90° CCW", key=f"rot_ccw_{i}"):
                new_img = r["image"].rotate(90, expand=True)
                raw_text, proc_img, cfg, boxes, overlay = extract_text_tesseract_or_crnn(new_img)
                r.update({"image": new_img, "original_text": raw_text, "text": apply_lexicon_pass(apply_learned_corrections(raw_text, r["image_key"])),
                          "config": cfg, "processed_image": proc_img, "line_boxes": boxes, "overlay_image": overlay})
            if b2.button("⟲ 180°", key=f"rot_180_{i}"):
                new_img = r["image"].rotate(180, expand=True)
                raw_text, proc_img, cfg, boxes, overlay = extract_text_tesseract_or_crnn(new_img)
                r.update({"image": new_img, "original_text": raw_text, "text": apply_lexicon_pass(apply_learned_corrections(raw_text, r["image_key"])),
                          "config": cfg, "processed_image": proc_img, "line_boxes": boxes, "overlay_image": overlay})
            if b3.button("↪︎ 90° CW", key=f"rot_cw_{i}"):
                new_img = r["image"].rotate(-90, expand=True)
                raw_text, proc_img, cfg, boxes, overlay = extract_text_tesseract_or_crnn(new_img)
                r.update({"image": new_img, "original_text": raw_text, "text": apply_lexicon_pass(apply_learned_corrections(raw_text, r["image_key"])),
                          "config": cfg, "processed_image": proc_img, "line_boxes": boxes, "overlay_image": overlay})

        with txtcol:
            st.caption("Raw OCR")
            st.code(r["original_text"] or "", language="text")
            st.caption("✏️ Corrected (editable)")
            new_text = st.text_area("",
                                    value=r["text"] or "",
                                    key=f"edit_{i}",
                                    height=160)
            c1, c2, c3 = st.columns(3)
            if c1.button("💾 Save correction", key=f"save_{i}"):
                if save_correction(r["image_key"], r["original_text"], new_text):
                    r["text"] = new_text
                    st.success("Saved & learned from correction.")
                else:
                    st.info("No changes to save.")
            if c2.button("↺ Re-apply learning", key=f"relearn_{i}"):
                r["text"] = apply_lexicon_pass(apply_learned_corrections(r["original_text"], r["image_key"]))
                st.success("Re-applied learned corrections.")
            if c3.button("⬇️ Download text", key=f"dl_{i}"):
                st.download_button("Download .txt",
                                   data=(r["text"] or r["original_text"]).encode("utf-8"),
                                   file_name=os.path.splitext(r["filename"])[0] + ".txt",
                                   mime="text/plain",
                                   key=f"dl_btn_{i}")

    # Export all
    st.markdown("---")
    st.subheader("📦 Export all texts")
    all_txt = []
    for r in st.session_state.ocr_results:
        name = os.path.splitext(r["filename"])[0]
        all_txt.append(f"### {name}\n{r['text'] or r['original_text']}\n")
    st.download_button("⬇️ Download combined.txt",
                       data=("\n\n".join(all_txt)).encode("utf-8"),
                       file_name="combined.txt",
                       mime="text/plain")

# ---------- Sidebar: learning status & data mgmt ----------
st.sidebar.markdown("### 🎓 Learning Status")
st.sidebar.metric("💾 Saved Corrections", len(st.session_state.corrections))
st.sidebar.metric("🧩 Learned Patterns", len(st.session_state.learned_patterns))

if st.session_state.learned_patterns:
    st.sidebar.markdown("**🔝 Top Patterns:**")
    rows = []
    for pat, info in st.session_state.learned_patterns.items():
        if isinstance(info, dict):
            try: cnt = int(info.get("count", 0))
            except: cnt = 0
            rep = str(info.get("replacement", ""))
            rows.append((pat, cnt, rep))
    for pat, cnt, rep in sorted(rows, key=lambda x: -x[1])[:3]:
        st.sidebar.caption(f"`{pat}` → `{rep}` ({cnt}×)")

if st.sidebar.button("💾 Save All Data"):
    persist_all()
    st.sidebar.success("✅ Data saved!")

st.sidebar.markdown("### 📥📤 Data Management")
st.sidebar.download_button(
    "⬇️ Export Corrections",
    data=json.dumps(_wrap(st.session_state.corrections), ensure_ascii=False, indent=2),
    file_name="corrections.json",
    mime="application/json",
)
st.sidebar.download_button(
    "⬇️ Export Patterns",
    data=json.dumps(_wrap(st.session_state.learned_patterns), ensure_ascii=False, indent=2),
    file_name="learned_patterns.json",
    mime="application/json",
)

imp_corr = st.sidebar.file_uploader("📤 Import Corrections", type=["json"], key="restore_corr")
if imp_corr:
    try:
        payload = json.load(imp_corr)
        data, _ = _unwrap(payload, {})
        # normalize
        norm = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and isinstance(v.get("corrected"), str):
                    norm[str(k)] = v
                elif isinstance(v, str):
                    norm[str(k)] = {"original": "", "corrected": v, "timestamp": _now_iso()}
        st.session_state.corrections = norm
        sanitize_learned_patterns()
        persist_all()
        st.sidebar.success("✅ Corrections imported!")
    except Exception as e:
        st.sidebar.error(f"❌ Import failed: {e}")

imp_pat = st.sidebar.file_uploader("📤 Import Patterns", type=["json"], key="restore_pat")
if imp_pat:
    try:
        payload = json.load(imp_pat)
        data, _ = _unwrap(payload, {})
        st.session_state.learned_patterns = data if isinstance(data, dict) else {}
        sanitize_learned_patterns()
        persist_all()
        st.sidebar.success("✅ Patterns imported!")
    except Exception as e:
        st.sidebar.error(f"❌ Import failed: {e}")
