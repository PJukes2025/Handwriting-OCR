# train_crnn.py  — Enhanced CRNN for handwriting lines
# ----------------------------------------------------
import os, csv, string, random
from typing import List, Tuple
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

# Character set suited to journalism/poetry (can be overridden by checkpoint)
CHARS = string.ascii_letters + string.digits + " .,;:!?()-'\"/&@#$%"
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is CTC blank
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
BLANK = 0

# ---------- Data ----------
def _read_labels_flex(path):
    encodings = ["utf-8-sig","utf-8","utf-16","latin-1"]
    delims = [",",";","\t","|"]
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
                for alt in [";", "\t", "|", ","]:
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

class HandwritingAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, img: Image.Image):
        if random.random() > self.prob: return img
        arr = np.array(img)
        # minor rotation
        if random.random() < 0.30:
            angle = random.uniform(-5, 5)
            img = img.rotate(angle, fillcolor=255, expand=False)
            arr = np.array(img)
        # mild noise
        if random.random() < 0.40:
            noise = np.random.normal(0, 5, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

class HandwritingLineDataset(Dataset):
    def __init__(self, root, img_dir="images", labels="labels.csv",
                 img_h=32, max_w=512, augment=True):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        labels_path = os.path.join(root, labels)
        if not os.path.exists(labels_path): raise FileNotFoundError(labels_path)
        self.items = _read_labels_flex(labels_path)
        if len(self.items)==0: raise RuntimeError(f"No labeled items in {labels_path}")
        self.img_h=img_h; self.max_w=max_w
        self.to_tensor=T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])
        self.augment = HandwritingAugmentation(prob=0.7) if augment else None

    def _resize_center(self, img: Image.Image):
        # keep aspect ratio, height -> img_h, center/pad to width max_w
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
        if self.augment: img = self.augment(img)
        img = self._resize_center(img)
        x = self.to_tensor(img)             # [1,H,W]
        x = self.normalize(x)               # [-1,1]
        y = [char2idx.get(c, 0) for c in txt]  # map unknown -> blank (0)
        y = torch.LongTensor(y)
        return x, y, len(y)

def collate_fn(batch):
    xs, ys, lens = zip(*batch)
    xs = torch.stack(xs, dim=0)   # [N,1,H,W]
    y_concat = torch.cat(ys, dim=0)  # [sum(len)]
    y_lens = torch.IntTensor([len(y) for y in ys])
    return xs, y_concat, y_lens

# ---------- Model ----------
class EnhancedCRNN(nn.Module):
    """
    Slightly deeper CNN + 2-layer BiLSTM.
    Input: [N, 1, 32, W]
    Output: [T, N, K] for CTC
    """
    def __init__(self, n_classes=len(CHARS)+1, cnn_out=256, rnn_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2,2),  # 32xW -> 16xW/2

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2,2),  # 16x -> 8x

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1)),  # 8x -> 4x

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
            f = f.mean(dim=2)      # [N,C,W'] reduce height
        else:
            f = f.squeeze(2)       # [N,C,W']
        f = f.permute(2,0,1)       # [T,N,C]
        y,_ = self.rnn(f)          # [T,N,2H]
        y = self.dropout(y)
        y = self.fc(y)             # [T,N,K]
        return y

# ---------- Train (optional) ----------
def train_main(data_train, data_val, epochs=10, lr=1e-3, batch_size=32, device="cpu", out_path="crnn_out.pt"):
    model = EnhancedCRNN().to(device)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt  = optim.AdamW(model.parameters(), lr=lr)
    tr_loader = DataLoader(HandwritingLineDataset(data_train), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(HandwritingLineDataset(data_val, augment=False), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for ep in range(1, epochs+1):
        model.train(); tloss=0.0; steps=0
        for xs, y_concat, y_lens in tr_loader:
            xs = xs.to(device); y_concat = y_concat.to(device)
            opt.zero_grad()
            logits = model(xs)                          # [T,N,K]
            Tlen, N, K = logits.size()
            logp = nn.functional.log_softmax(logits, dim=2)
            input_lengths = torch.full(size=(N,), fill_value=Tlen, dtype=torch.int32)
            loss = crit(logp, y_concat, input_lengths, y_lens)
            loss.backward(); opt.step()
            tloss += float(loss.item()); steps += 1
        print(f"[{ep}] train_loss={tloss/max(1,steps):.4f}")

        model.eval(); vloss=0.0; vsteps=0
        with torch.no_grad():
            for xs, y_concat, y_lens in va_loader:
                xs = xs.to(device); y_concat = y_concat.to(device)
                logits = model(xs)
                Tlen, N, K = logits.size()
                logp = nn.functional.log_softmax(logits, dim=2)
                input_lengths = torch.full(size=(N,), fill_value=Tlen, dtype=torch.int32)
                loss = crit(logp, y_concat, input_lengths, y_lens)
                vloss += float(loss.item()); vsteps+=1
        print(f"[{ep}] valid_loss={vloss/max(1,vsteps):.4f}")

    ckpt = {"model": model.state_dict(), "charset": CHARS}
    torch.save(ckpt, out_path)
    print("[done] saved", out_path)
