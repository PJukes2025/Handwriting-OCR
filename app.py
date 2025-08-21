# app.py - Enhanced Handwriting OCR with Improved CRNN
# ----------------------------------------------------
# Streamlit app for line-segmented handwriting OCR using CRNN + Tesseract fallback.
# Includes robust preprocessing, segmentation, ensemble CRNN loading, and learning from corrections.

import os, io, json, time, tempfile, hashlib, platform, shutil, zipfile, sys, textwrap, re, csv, glob, urllib.request, contextlib
from datetime import datetime
from typing import List, Tuple

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
import pytesseract

# ===== Optional SciPy smoothing (fallback to simple moving average) =====
SCIPY_AVAILABLE = False
try:
    from scipy.ndimage import gaussian_filter1d as _gauss_1d
    SCIPY_AVAILABLE = True
except Exception:
    pass

def gaussian_filter1d(data, sigma=1.0):
    if SCIPY_AVAILABLE:
        return _gauss_1d(data, sigma=sigma)
    # Simple moving-average fallback approximating smoothing
    k = max(3, int(2 * sigma * 3) | 1)  # odd
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(np.asarray(data, dtype=np.float32), kernel, mode="same")


# ========= Streamlit base config =========
st.set_page_config(page_title="Journalist’s OCR Tool", page_icon="📝", layout="wide")

# ========= Paths / constants =========
DATA_DIR         = "ocr_data"
MODELS_DIR       = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CORRECTIONS_PATH = os.path.join(DATA_DIR, "corrections.json")
PATTERNS_PATH    = os.path.join(DATA_DIR, "learned_patterns.json")
USER_WORDS_PATH  = os.path.join(DATA_DIR, "user_words.txt")

DEFAULT_MODEL_FILENAME = "crnn_handwriting.pt"
DEFAULT_MODEL_PATH     = os.path.join(".", DEFAULT_MODEL_FILENAME)

TRAIN_SCRIPT_PATH = "train_crnn.py"  # will be written out if missing (see TRAIN_SCRIPT content at bottom)

SCHEMA_VERSION = 1


# ========= Robust JSON layer =========
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _wrap(data):
    return {"_meta": {"schema": SCHEMA_VERSION, "saved_at": _now_iso()}, "data": data}

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
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

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


# ========= Small utilities =========
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USER_WORDS_PATH):
        with open(USER_WORDS_PATH, "w", encoding="utf-8") as f:
            pass

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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


# ========= Model fetching helpers =========
def fetch_model_if_needed(url, out_path, expected_sha256=None, github_token=None):
    need = True
    if os.path.exists(out_path):
        if expected_sha256:
            need = (_sha256(out_path) != expected_sha256)
        else:
            need = False
    if not need:
        return out_path

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

def fetch_model_zip_if_needed(url, out_path, expected_sha256=None, github_token=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_zip = out_path + ".zip.tmp"

    req = urllib.request.Request(url)
    if github_token:
        req.add_header("Authorization", f"token {github_token}")
        req.add_header("Accept", "application/octet-stream")

    with contextlib.closing(urllib.request.urlopen(req)) as resp, open(tmp_zip, "wb") as f:
        f.write(resp.read())

    if expected_sha256 and _sha256(tmp_zip) != expected_sha256:
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


# ========= Persistence =========
def persist_all():
    save_json_versioned(CORRECTIONS_PATH, st.session_state.corrections)
    save_json_versioned(PATTERNS_PATH,    st.session_state.learned_patterns)
    rebuild_user_words()


# ========= Session state initialization =========
ensure_data_dir()
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = []
if "corrections" not in st.session_state:
    st.session_state.corrections, _ = load_json_versioned(CORRECTIONS_PATH, {})
if "learned_patterns" not in st.session_state:
    st.session_state.learned_patterns, _ = load_json_versioned(PATTERNS_PATH, {})


# ========= Learning helpers =========
def cleanup_token(w: str) -> str:
    return w.strip().strip(".,;:!?()[]{}\"'")

def rebuild_user_words():
    """Build Tesseract user-words from corrections and learned patterns."""
    def add_from_text(t, acc):
        if not isinstance(t, str):
            return
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

    # Also add 'replacements' from learned patterns
    lp = st.session_state.get("learned_patterns", {})
    if isinstance(lp, dict):
        for info in lp.values():
            if isinstance(info, dict):
                add_from_text(info.get("replacement"), words)
            elif isinstance(info, str):
                add_from_text(info, words)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(USER_WORDS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(words)))

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
    # If user already corrected this image, return that correction
    if image_key in st.session_state.corrections:
        return st.session_state.corrections[image_key].get("corrected", text)

    corrected_text = text

    # Built-in frequent handwriting confusions
    built_in_fixes = {
        "rn": "m", "cl": "d", "li": "h",
        "tlie": "the", "ornd": "and",
        "witl1": "with", "l1is": "his", "l1er": "her",
        "tl1e": "the", "wl1en": "when", "l1ow": "how",
        "l1ave": "have",
    }
    for a, b in built_in_fixes.items():
        corrected_text = corrected_text.replace(a, b)

    # Learned patterns
    items = []
    for pat, info in st.session_state.learned_patterns.items():
        if isinstance(info, dict):
            try:
                cnt = int(info.get("count", 0))
            except Exception:
                cnt = 0
            repl = str(info.get("replacement", ""))
            items.append((pat, cnt, repl))
        elif isinstance(info, str):
            items.append((pat, 1, info))
    for pat, _, repl in sorted(items, key=lambda x: -x[1]):
        if pat and repl and pat in corrected_text:
            corrected_text = corrected_text.replace(pat, repl)

    return corrected_text


# ========= Image helpers =========
def make_image_key(filename, image: Image.Image) -> str:
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    digest = hashlib.md5(bio.getvalue()).hexdigest()[:8]
    return f"{filename}_{digest}"

def draw_boxes_on_image(pil_img, boxes, width=2):
    overlay = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for (x0, y0, x1, y1) in boxes or []:
        for t in range(width):
            draw.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=(255, 0, 0))
    return overlay


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
                    return pil_img.rotate(-angle, expand=True)
                elif angle == 180:
                    return pil_img.rotate(180, expand=True)
                else:
                    return pil_img
        except Exception:
            pass
        # Heuristic fallback: if portrait, assume needs 90° CW
        if pil_img.width < pil_img.height:
            return pil_img.rotate(-90, expand=True)
        return pil_img

    if mode == "90° CW":
        return pil_img.rotate(-90, expand=True)
    if mode == "90° CCW":
        return pil_img.rotate(90, expand=True)
    if mode == "180°":
        return pil_img.rotate(180, expand=True)

    return pil_img


# ========= Handwriting preprocessing =========
def preprocess_image_handwriting(image: Image.Image, enhancement_level="medium") -> np.ndarray:
    """Handwriting-optimized preprocessing pipeline."""
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if len(arr.shape) == 3 else arr

    if enhancement_level == "light":
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    elif enhancement_level == "medium":
        denoised = cv2.fastNlMeansDenoising(gray, h=15)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        processed = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )

    else:  # aggressive
        denoised = cv2.fastNlMeansDenoising(gray, h=20)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        enhanced = clahe.apply(denoised)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        processed = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel2)

    return processed


# ========= Line segmentation (handwriting-optimized) =========
def segment_lines_handwriting_optimized(pil_img: Image.Image, min_line_height=20, gap_thresh=8, merge_close_lines=True):
    img = np.array(pil_img.convert("L"))
    h, w = img.shape[:2]

    # Conservative binarization (invert so text=white on black for sums)
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)

    # Horizontal projection
    proj = binary.sum(axis=1)
    proj_smooth = gaussian_filter1d(proj.astype(float), sigma=2)

    lines = []
    in_run, start = False, 0
    threshold = proj_smooth.max() * 0.1

    for y in range(h):
        if proj_smooth[y] > threshold and not in_run:
            in_run = True
            start = y
        elif proj_smooth[y] <= threshold and in_run:
            end = y
            if end - start >= min_line_height:
                lines.append((start, end))
            in_run = False
    if in_run:
        end = h
        if end - start >= min_line_height:
            lines.append((start, end))

    # Merge close lines (common descender/ascender overlaps)
    if merge_close_lines and lines:
        merged = []
        for s, e in lines:
            if not merged:
                merged.append([s, e]); continue
            ps, pe = merged[-1]
            if s - pe <= gap_thresh:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        lines = [(s, e) for s, e in merged]

    # Extract crops with dynamic padding + horizontal bounds
    crops, boxes = [], []
    for s, e in lines:
        line_height = e - s
        v_pad = max(3, line_height // 6)
        h_pad = 5
        s2 = max(0, s - v_pad)
        e2 = min(h, e + v_pad)
        strip = img[s2:e2, :]

        col_proj = (255 - strip).sum(axis=0)
        col_proj_smooth = gaussian_filter1d(col_proj.astype(float), sigma=1)
        xs = np.where(col_proj_smooth > col_proj_smooth.max() * 0.05)[0]
        if xs.size == 0:
            continue
        x0, x1 = xs[0], xs[-1] + 1
        x0 = max(0, x0 - h_pad)
        x1 = min(w, x1 + h_pad)
        line_crop = strip[:, x0:x1]
        crops.append(Image.fromarray(line_crop))
        boxes.append((int(x0), int(s2), int(x1), int(e2)))

    return crops, boxes


def prepare_line_for_crnn(img_pil: Image.Image, target_height=32, max_width=512, add_padding=True) -> Image.Image:
    img = img_pil.convert("L")
    w, h = img.size
    if h <= 0:
        return Image.new("L", (max_width, target_height), 255)

    aspect_ratio = max(1e-6, w / h)
    new_width = int(target_height * aspect_ratio)
    if new_width > max_width:
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
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


# ========= Legacy Tesseract OCR =========
def extract_text_tesseract(image: Image.Image, enhancement_level="medium"):
    processed = preprocess_image_handwriting(image, enhancement_level)
    pil_proc = Image.fromarray(processed)

    extra = ""
    try:
        if os.path.getsize(USER_WORDS_PATH) > 0:
            # NOTE: pytesseract doesn't pass --user-words directly; but we keep this path
            # for you to integrate custom runners if needed. We still benefit from training via corrections.
            pass
    except Exception:
        pass

    # Try different PSMs
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 7",
        "--oem 3 --psm 8",
    ]
    best, best_cfg = "", "general"
    for i, cfg in enumerate(configs):
        try:
            out = pytesseract.image_to_string(pil_proc, config=cfg).strip()
            if len(out) > len(best):
                best, best_cfg = out, ["general", "block", "word"][i]
        except Exception:
            continue
    return best, processed, best_cfg


# ========= Enhanced line segmentation + CRNN dispatch =========
def extract_text_tesseract_or_crnn(pil_image: Image.Image, enhancement_level="medium",
                                   use_crnn=False, crnn_models=None, crnn_idx2char=None,
                                   show_line_boxes=True):
    if use_crnn and crnn_models:
        import torchvision.transforms as T
        import torch

        def recognize_line_enhanced(img_pil):
            processed = preprocess_image_handwriting(img_pil, enhancement_level)
            img = Image.fromarray(processed)
            img = prepare_line_for_crnn(img, target_height=32, max_width=512)
            x = T.ToTensor()(img)
            x = T.Normalize(mean=[0.5], std=[0.5])(x)  # [-1,1]
            x = x.unsqueeze(0)  # [N=1,C,H,W]

            with torch.no_grad():
                logits_sum = None
                for m in crnn_models:
                    logits = m(x)  # [T,N,C]
                    logits_sum = logits if logits_sum is None else (logits_sum + logits)
                if len(crnn_models) > 1:
                    logits_sum = logits_sum / len(crnn_models)

                # Greedy CTC decode
                best = logits_sum.argmax(dim=2).permute(1, 0)[0].tolist()

            out, prev = [], -1
            BLANK = 0
            for t in best:
                if t != prev and t != BLANK:
                    ch = crnn_idx2char.get(t, "")
                    if ch:
                        out.append(ch)
                prev = t
            return "".join(out)

        crops, boxes = segment_lines_handwriting_optimized(pil_image, min_line_height=20, gap_thresh=8)
        lines = crops or [pil_image]

        recognized = []
        for ln in lines:
            text = recognize_line_enhanced(ln).strip()
            if text:
                recognized.append(text)

        raw_text = "\n".join(recognized)
        proc_img = preprocess_image_handwriting(pil_image, enhancement_level)
        cfg = f"enhanced-crnn-{len(crnn_models)}model(s)"
        overlay = draw_boxes_on_image(pil_image, boxes) if show_line_boxes and boxes else None
        return raw_text, proc_img, cfg, boxes, overlay

    # Fallback to Tesseract
    raw_text, proc_img, cfg = extract_text_tesseract(pil_image, enhancement_level)
    return raw_text, proc_img, cfg, [], None


# ========= Sidebar Configuration =========
st.sidebar.header("🎛️ OCR Settings")

if _TESS_PATH:
    st.sidebar.success(f"✅ Tesseract: {os.path.basename(_TESS_PATH)}")
else:
    st.sidebar.error(
        "❌ Tesseract not found.\n\n"
        "macOS: brew install tesseract\n"
        "Ubuntu: sudo apt install tesseract-ocr\n"
        "Windows: install the UB Mannheim build"
    )

enhancement_level = st.sidebar.selectbox(
    "📈 Enhancement Level",
    ["light", "medium", "aggressive"],
    index=1,
    help="Light: clear writing • Medium: typical • Aggressive: faint/marginal notes",
)

orientation_mode = st.sidebar.selectbox(
    "🔄 Auto-Rotation",
    ["Auto (keep text horizontal)", "None", "90° CW", "90° CCW", "180°"],
    index=0,
    help="Auto tries to keep text horizontal using Tesseract OSD or heuristics.",
)

show_line_boxes = st.sidebar.toggle("📦 Show Line Boxes", value=True)


# ===== Model Selection & Management =====
st.sidebar.markdown("### 🧠 CRNN Model")

model_choice = st.sidebar.selectbox(
    "Model Source",
    [
        "Remote URLs via Secrets (multiple)",
        "Remote URL via Secrets (single)",
        "Local file(s) in repo",
        "Paste a URL",
        "Upload (ephemeral)"
    ],
    index=0
)

LOCAL_MODEL_CANDIDATES = []
LOCAL_MODEL_CANDIDATES += sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
LOCAL_MODEL_CANDIDATES += sorted(glob.glob("*.pt"))

MODEL_PATH = DEFAULT_MODEL_PATH
MODEL_PATHS: List[str] = []
model_exists = False
remote_loaded_error = None

if model_choice == "Local file(s) in repo":
    if not LOCAL_MODEL_CANDIDATES and os.path.exists(DEFAULT_MODEL_PATH):
        LOCAL_MODEL_CANDIDATES.append(DEFAULT_MODEL_PATH)
    if LOCAL_MODEL_CANDIDATES:
        selected_local = st.sidebar.selectbox("Select model", LOCAL_MODEL_CANDIDATES, index=0)
        MODEL_PATH = selected_local
        model_exists = os.path.exists(MODEL_PATH)
        st.sidebar.info(f"📦 Using: {os.path.basename(MODEL_PATH)}")
    else:
        st.sidebar.warning("No .pt files found. Use Git LFS or remote URL.")
        model_exists = False

elif model_choice == "Remote URLs via Secrets (multiple)":
    models_map = st.secrets.get("models", {})
    shas_map   = st.secrets.get("models_sha256", {})
    gh_token   = st.secrets.get("github", {}).get("token") if isinstance(st.secrets.get("github", {}), dict) else None

    if not models_map:
        st.sidebar.info("Configure in Secrets:\n[models]\nmodel1=\"https://.../m1.pt\"\nmodel2=\"https://.../m2.pt\"")
        model_exists = False
    else:
        keys = st.sidebar.multiselect(
            "Select model(s)",
            sorted(models_map.keys()),
            default=[sorted(models_map.keys())[0]]
        )
        try:
            for key in keys:
                url = models_map.get(key, "")
                sha = (shas_map or {}).get(key)
                path = os.path.join(MODELS_DIR, f"{key}.pt")
                if url:
                    if url.lower().endswith(".zip"):
                        fetch_model_zip_if_needed(url, path, expected_sha256=(sha or None), github_token=gh_token)
                    else:
                        fetch_model_if_needed(url, path, expected_sha256=(sha or None), github_token=gh_token)
                    MODEL_PATHS.append(path)
            model_exists = len(MODEL_PATHS) > 0
            if model_exists:
                st.sidebar.success(f"✅ Loaded {len(MODEL_PATHS)} model(s)")
        except Exception as e:
            model_exists = False
            remote_loaded_error = str(e)
            st.sidebar.error(f"❌ Download failed: {e}")

elif model_choice == "Remote URL via Secrets (single)":
    url = st.secrets.get("model", {}).get("url")
    sha = st.secrets.get("model", {}).get("sha256")
    gh_token = st.secrets.get("github", {}).get("token") if isinstance(st.secrets.get("github", {}), dict) else None
    MODEL_PATH = os.path.join(MODELS_DIR, "remote_crnn.pt")
    try:
        if url:
            if url.lower().endswith(".zip"):
                fetch_model_zip_if_needed(url, MODEL_PATH, expected_sha256=(sha or None), github_token=gh_token)
            else:
                fetch_model_if_needed(url, MODEL_PATH, expected_sha256=(sha or None), github_token=gh_token)
        model_exists = os.path.exists(MODEL_PATH)
        if model_exists:
            st.sidebar.success("✅ Remote model ready")
    except Exception as e:
        model_exists = False
        remote_loaded_error = str(e)
        st.sidebar.error(f"❌ Download failed: {e}")

elif model_choice == "Paste a URL":
    url = st.sidebar.text_input("🔗 Model URL (.pt or .zip)", "")
    sha = st.sidebar.text_input("🔒 SHA256 (optional)", "")
    token = st.sidebar.text_input("🎫 GitHub Token (for private)", type="password")
    MODEL_PATH = os.path.join(MODELS_DIR, "pasted_crnn.pt")
    if st.sidebar.button("⬇️ Download"):
        try:
            if url.lower().endswith(".zip"):
                fetch_model_zip_if_needed(url, MODEL_PATH, expected_sha256=(sha or None), github_token=(token or None))
            else:
                fetch_model_if_needed(url, MODEL_PATH, expected_sha256=(sha or None), github_token=(token or None))
            model_exists = True
            st.sidebar.success("✅ Downloaded successfully")
        except Exception as e:
            model_exists = False
            remote_loaded_error = str(e)
            st.sidebar.error(f"❌ Failed: {e}")
    else:
        model_exists = os.path.exists(MODEL_PATH)

else:  # Upload (ephemeral)
    up = st.sidebar.file_uploader("📤 Upload .pt (ephemeral)", type=["pt"], key="model_upload")
    MODEL_PATH = os.path.join(MODELS_DIR, "uploaded_crnn.pt")
    if up is not None:
        try:
            with open(MODEL_PATH, "wb") as f:
                f.write(up.read())
            model_exists = True
            st.sidebar.success("✅ Upload successful")
        except Exception as e:
            model_exists = False
            st.sidebar.error(f"❌ Upload failed: {e}")
    else:
        model_exists = os.path.exists(MODEL_PATH)


# Model toggle
use_crnn = st.sidebar.toggle("🧠 Use CRNN", value=model_exists) if model_exists else False
if not model_exists:
    st.sidebar.info("🔧 CRNN unavailable — will use Tesseract only")


# ========= Load CRNN Models =========
crnn_ready = False
crnn_models = None
crnn_idx2char = None

if use_crnn and model_exists:
    try:
        import importlib.util, torch
        def ensure_trainer_script():
            if not os.path.exists(TRAIN_SCRIPT_PATH):
                with open(TRAIN_SCRIPT_PATH, "w", encoding="utf-8") as f:
                    f.write(TRAIN_SCRIPT)

        ensure_trainer_script()
        spec = importlib.util.spec_from_file_location("train_crnn", TRAIN_SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore

        @st.cache_resource
        def load_crnn_models(paths: List[str]):
            models = []
            common_idx2char = None
            for p in paths:
                ckpt = torch.load(p, map_location="cpu")
                if "model" in ckpt:
                    model_state = ckpt["model"]
                    charset = ckpt.get("charset", getattr(mod, "CHARS", ""))
                else:
                    model_state = ckpt.get("model_state_dict", ckpt)
                    charset = getattr(mod, "CHARS", "")
                m = mod.EnhancedCRNN()
                m.load_state_dict(model_state)
                m.eval()
                idx2char = {i + 1: c for i, c in enumerate(charset)}
                if common_idx2char is None:
                    common_idx2char = idx2char
                else:
                    if "".join(common_idx2char.values()) != "".join(idx2char.values()):
                        st.sidebar.warning("⚠️ Ensemble models have different charsets")
                models.append(m)
            return models, common_idx2char

        if model_choice == "Remote URLs via Secrets (multiple)" and MODEL_PATHS:
            crnn_models, crnn_idx2char = load_crnn_models(MODEL_PATHS)
        else:
            crnn_models, crnn_idx2char = load_crnn_models([MODEL_PATH])

        crnn_ready = True
        st.sidebar.success(f"🧠 CRNN loaded ({len(crnn_models)} model{'s' if len(crnn_models)>1 else ''})")
    except Exception as e:
        st.sidebar.error(f"❌ CRNN failed: {e}")
        use_crnn = False


# ========= Learning Status =========
sanitize_learned_patterns()
persist_all()

st.sidebar.markdown("### 🎓 Learning Status")
num_corrections = len(st.session_state.corrections)
num_patterns = len(st.session_state.learned_patterns)
st.sidebar.metric("💾 Saved Corrections", num_corrections)
st.sidebar.metric("🧩 Learned Patterns", num_patterns)

if st.session_state.learned_patterns:
    st.sidebar.markdown("**🔝 Top Patterns:**")
    rows = []
    for pat, info in st.session_state.learned_patterns.items():
        if isinstance(info, dict):
            try:
                cnt = int(info.get("count", 0))
            except Exception:
                cnt = 0
            rep = str(info.get("replacement", ""))
        elif isinstance(info, str):
            cnt = 1; rep = info
        else:
            continue
        rows.append((pat, cnt, rep))
    for pat, cnt, rep in sorted(rows, key=lambda x: -x[1])[:3]:
        st.sidebar.caption(f"`{pat}` → `{rep}` ({cnt}×)")

if st.sidebar.button("💾 Save All Data"):
    persist_all()
    st.sidebar.success("✅ Data saved!")


# ========= Import/Export =========
def _normalize_corrections(data):
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(v.get("corrected"), str):
                out[str(k)] = v
            elif isinstance(v, str):
                out[str(k)] = {"original": "", "corrected": v, "timestamp": _now_iso()}
        return out
    if isinstance(data, list):
        out = {}
        for i, item in enumerate(data):
            key = (item.get("image_key") if isinstance(item, dict) else None) or f"import*{i}"
            if isinstance(item, dict) and isinstance(item.get("corrected"), str):
                out[key] = item
            elif isinstance(item, str):
                out[key] = {"original": "", "corrected": item, "timestamp": _now_iso()}
        return out
    return {}

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
        st.session_state.corrections = _normalize_corrections(data)
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


# ========= Main Application Interface =========
st.title("📝 Enhanced Journalist’s OCR Tool")

col1, col2, col3 = st.columns(3)
with col1:
    engine_status = "🧠 CRNN" if (use_crnn and crnn_ready) else "📖 Tesseract"
    st.metric("OCR Engine", engine_status)
with col2:
    enhancement_status = {"light": "🌟 Light", "medium": "⚡ Medium", "aggressive": "🔥 Aggressive"}[enhancement_level]
    st.metric("Enhancement", enhancement_status)
with col3:
    learning_status = f"{num_corrections} corrections, {num_patterns} patterns"
    st.metric("Learning", learning_status)

st.markdown(
    "*Advanced handwriting OCR with CRNN neural networks, intelligent line segmentation, "
    "and adaptive learning from corrections. Optimized for journalism workflows.*"
)

# File upload
st.header("📤 Upload Handwritten Images")
uploaded_files = st.file_uploader(
    "Select handwritten document images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="📱 Take clear photos with good lighting. The app will auto-orient and enhance them."
)

if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} files ready for processing!")

if uploaded_files and st.button("🚀 Process All Images", type="primary"):
    progress = st.progress(0)
    status = st.empty()

    for i, uf in enumerate(uploaded_files):
        status.text(f"Processing {uf.name}... ({i+1}/{len(uploaded_files)})")
        try:
            image = Image.open(uf)
            image = ImageOps.exif_transpose(image)
            baseline = rotate_image(image, orientation_mode)
            key = make_image_key(uf.name, baseline)

            raw_text, proc_img, cfg, boxes, overlay = extract_text_tesseract_or_crnn(
                baseline, enhancement_level=enhancement_level,
                use_crnn=(use_crnn and crnn_ready),
                crnn_models=(crnn_models if (use_crnn and crnn_ready) else None),
                crnn_idx2char=crnn_idx2char,
                show_line_boxes=show_line_boxes
            )

            corrected_text = apply_learned_corrections(raw_text, key)

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
                "orientation_mode": orientation_mode,
            })

        except Exception as e:
            st.error(f"❌ Error processing {uf.name}: {e}")

        progress.progress((i + 1) / len(uploaded_files))

    status.text("✅ All images processed!")
    st.success("🎉 Processing complete! Scroll down to review and correct results.")


# ========= Results Display & Learning Interface =========
def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    sa, sb = set(a.split()), set(b.split())
    if not sa:
        return 0.0
    return len(sa & sb) / max(1, len(sa))

if st.session_state.ocr_results:
    st.header("📋 OCR Results & Learning Interface")
    for i, r in enumerate(st.session_state.ocr_results):
        st.markdown("---")
        colA, colB = st.columns([3, 1])
        with colA:
            st.subheader(f"📄 {r['filename']}")
            st.caption(f"⚙️ {r['config']} • 📅 {r['timestamp']} • 🎛️ {r.get('enhancement_level', 'medium')}")
        with colB:
            if r.get("original_text") and r.get("text"):
                st.metric("✨ Similarity", f"{_similarity(r['text'], r['original_text']):.0%}")

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(r["image"], caption="Input (auto-rotated)", use_column_width=True)
            if r.get("overlay_image") is not None:
                st.image(r["overlay_image"], caption="Line boxes (segmentation)", use_column_width=True)
        with img_col2:
            st.image(r["processed_image"], caption="Processed (for OCR)", use_column_width=True)

        st.markdown("**✍️ Edit & Save Correction:**")
        corrected = st.text_area("Corrected text", value=r.get("text", ""), key=f"corr_txt_{i}", height=140)
        save_btn = st.button("💾 Save Correction", key=f"save_corr_{i}")
        if save_btn:
            changed = save_correction(r["image_key"], r.get("original_text", ""), corrected)
            if changed:
                r["text"] = corrected
                st.success("✅ Correction saved (learning updated)")
            else:
                st.info("No change detected.")

        st.markdown("**🔄 Re-run OCR with different rotation:**")
        rr_col1, rr_col2, rr_col3, rr_col4 = st.columns(4)
        if rr_col1.button("↩︎ 90° CCW", key=f"rot_ccw_{i}"):
            new_img = r["image"].rotate(90, expand=True)
            rt, pi, cfg, bxs, ov = extract_text_tesseract_or_crnn(
                new_img, enhancement_level,
                use_crnn=(use_crnn and crnn_ready),
                crnn_models=crnn_models, crnn_idx2char=crnn_idx2char,
                show_line_boxes=show_line_boxes
            )
            r.update({"image": new_img, "original_text": rt, "text": apply_learned_corrections(rt, r["image_key"]),
                      "processed_image": pi, "config": cfg, "line_boxes": bxs, "overlay_image": ov})
        if rr_col2.button("↪︎ 90° CW", key=f"rot_cw_{i}"):
            new_img = r["image"].rotate(-90, expand=True)
            rt, pi, cfg, bxs, ov = extract_text_tesseract_or_crnn(
                new_img, enhancement_level,
                use_crnn=(use_crnn and crnn_ready),
                crnn_models=crnn_models, crnn_idx2char=crnn_idx2char,
                show_line_boxes=show_line_boxes
            )
            r.update({"image": new_img, "original_text": rt, "text": apply_learned_corrections(rt, r["image_key"]),
                      "processed_image": pi, "config": cfg, "line_boxes": bxs, "overlay_image": ov})
        if rr_col3.button("⟲ 180°", key=f"rot_180_{i}"):
            new_img = r["image"].rotate(180, expand=True)
            rt, pi, cfg, bxs, ov = extract_text_tesseract_or_crnn(
                new_img, enhancement_level,
                use_crnn=(use_crnn and crnn_ready),
                crnn_models=crnn_models, crnn_idx2char=crnn_idx2char,
                show_line_boxes=show_line_boxes
            )
            r.update({"image": new_img, "original_text": rt, "text": apply_learned_corrections(rt, r["image_key"]),
                      "processed_image": pi, "config": cfg, "line_boxes": bxs, "overlay_image": ov})
        if rr_col4.button("🔁 Re-OCR", key=f"re_ocr_{i}"):
            rt, pi, cfg, bxs, ov = extract_text_tesseract_or_crnn(
                r["image"], enhancement_level,
                use_crnn=(use_crnn and crnn_ready),
                crnn_models=crnn_models, crnn_idx2char=crnn_idx2char,
                show_line_boxes=show_line_boxes
            )
            r.update({"original_text": rt, "text": apply_learned_corrections(rt, r["image_key"]),
                      "processed_image": pi, "config": cfg, "line_boxes": bxs, "overlay_image": ov})

    # Export corrected text
    st.markdown("---")
    st.subheader("📦 Export")
    all_text = "\n\n".join([f"# {r['filename']}\n{r.get('text','')}" for r in st.session_state.ocr_results])
    st.download_button("⬇️ Download All (TXT)", data=all_text, file_name="ocr_export.txt", mime="text/plain")


# ========= Embedded CRNN training script (written if missing) =========
TRAIN_SCRIPT = textwrap.dedent("""
import os, csv, string, random
from typing import List, Tuple
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

# Character set suited to journalism/poetry (can be passed in checkpoint too)
CHARS = string.ascii_letters + string.digits + " .,;:!?()-'\\\"/&@#$%"
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is CTC blank
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
BLANK = 0

class HandwritingAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, img):
        if random.random() > self.prob:
            return img
        arr = np.array(img)
        # light random rotation
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            img = img.rotate(angle, fillcolor=255, expand=False); arr = np.array(img)
        # mild noise
        if random.random() < 0.4:
            noise = np.random.normal(0, 5, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

class EnhancedCRNN(nn.Module):
    def __init__(self, n_classes=len(CHARS)+1, cnn_out=256, rnn_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, cnn_out, 2, padding=0), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(cnn_out, rnn_hidden, num_layers=2, bidirectional=True, batch_first=False, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(rnn_hidden*2, n_classes)

    def forward(self, x):
        f = self.cnn(x)                 # [N,C,H',W']
        if f.size(2) > 1:
            f = f.mean(dim=2)           # [N,C,W'] (reduce height)
        else:
            f = f.squeeze(2)            # [N,C,W']
        f = f.permute(2,0,1)            # [T,N,C]
        y,_ = self.rnn(f)               # [T,N,2*H]
        y = self.dropout(y)
        y = self.fc(y)                  # [T,N,K]
        return y

def _read_labels_flex(path):
    encodings = ["utf-8-sig","utf-8","utf-16","latin-1"]
    delims = [",",";","\\t","|"]
    raw = None; used = "utf-8"
    for enc in encodings:
        try:
            with open(path,"r",encoding=enc,newline="") as f: raw = f.read(); used=enc; break
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
                        parts=row[0].split(alt,1); fn=parts[0].strip(); txt=parts[1].strip() if len(parts)>1 else ""; ok=True; break
                if not ok: continue
            if fn and txt: pairs.append((fn, txt))
    parse_with(delim)
    if not pairs:
        for d in delims:
            if d==delim: continue
            parse_with(d)
            if pairs: break
    if not pairs: raise RuntimeError("No labeled items parsed.")
    print(f"[labels] parsed {len(pairs)} from {path} using enc={used}")
    return pairs

class HandwritingLineDataset(Dataset):
    def __init__(self, root, img_dir="images", labels="labels.csv", img_h=32, max_w=512, augment=True):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        labels_path = os.path.join(root, labels)
        if not os.path.exists(labels_path): raise FileNotFoundError(labels_path)
        self.items = _read_labels_flex(labels_path)
        if len(self.items)==0: raise RuntimeError(f"No labeled items in {labels_path}")
        self.img_h=img_h; self.max_w=max_w
        self.to_tensor=T.ToTensor()
        self.augment = HandwritingAugmentation(prob=0.7) if augment else None
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])
        
    def _resize_for_handwriting(self, img):
        w, h = img.size
        if h <= 0:
            return Image.new("L", (self.max_w, self.img_h), 255)
        aspect = max(1e-6, w/h)
        new_w = int(self.img_h*aspect)
        img = img.convert("L").resize((new_w, self.img_h), Image.LANCZOS)
        if new_w < self.max_w:
            canvas = Image.new("L", (self.max_w, self.img_h), 255)
            canvas.paste(img, ((self.max_w-new_w)//2, 0))
            img = canvas
        elif new_w > self.max_w:
            sx = (new_w-self.max_w)//2
            img = img.crop((sx, 0, sx+self.max_w, self.img_h))
        return img

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fn, txt = self.items[idx]
        path = os.path.join(self.img_dir, fn)
        img = Image.open(path).convert("L")
        if self.augment: img = self.augment(img)
        img = self._resize_for_handwriting(img)
        x = self.to_tensor(img)          # [1,H,W]
        x = self.normalize(x)
        # Convert label to indices with CTC blank=0
        y = [char2idx.get(c, 0) for c in txt]
        y = torch.LongTensor(y)
        return x, y, len(y)

def collate_fn(batch):
    xs, ys, lens = zip(*batch)
    xs = torch.stack(xs, dim=0)
    y_concat = torch.cat(ys, dim=0)
    y_lens = torch.IntTensor([len(y) for y in ys])
    return xs, y_concat, y_lens

def train_main(data_train, data_val, epochs=10, lr=1e-3, batch_size=32, device="cpu", out_path="crnn_out.pt"):
    model = EnhancedCRNN().to(device)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt  = optim.AdamW(model.parameters(), lr=lr)
    tr_loader = DataLoader(HandwritingLineDataset(data_train), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(HandwritingLineDataset(data_val, augment=False), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for ep in range(1, epochs+1):
        model.train()
        tloss=0.0; steps=0
        for xs, y_concat, y_lens in tr_loader:
            xs = xs.to(device)
            y_concat = y_concat.to(device)
            opt.zero_grad()
            logits = model(xs)                  # [T,N,C]
            Tlen, N, C = logits.size()
            log_probs = nn.functional.log_softmax(logits, dim=2)
            input_lengths = torch.full(size=(N,), fill_value=Tlen, dtype=torch.int32)
            loss = crit(log_probs, y_concat, input_lengths, y_lens)
            loss.backward()
            opt.step()
            tloss += float(loss.item()); steps += 1
        print(f"[{ep}] train_loss={tloss/max(1,steps):.4f}")

        model.eval()
        with torch.no_grad():
            vloss=0.0; vsteps=0
            for xs, y_concat, y_lens in va_loader:
                xs = xs.to(device); y_concat = y_concat.to(device)
                logits = model(xs)
                Tlen, N, C = logits.size()
                log_probs = nn.functional.log_softmax(logits, dim=2)
                input_lengths = torch.full(size=(N,), fill_value=Tlen, dtype=torch.int32)
                loss = crit(log_probs, y_concat, input_lengths, y_lens)
                vloss += float(loss.item()); vsteps+=1
            print(f"[{ep}] valid_loss={vloss/max(1,vsteps):.4f}")

    ckpt = {"model": model.state_dict(), "charset": CHARS}
    torch.save(ckpt, out_path)
    print("[done] saved", out_path)
""")
