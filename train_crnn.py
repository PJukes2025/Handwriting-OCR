# train_crnn.py
import os
import csv
import math
import string
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ===== Charset (extend if you need more punctuation) =====
CHARS = string.ascii_letters + string.digits + " .,;:!?()-'\"/"
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is CTC blank
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
BLANK = 0


# ===== Dataset with robust CSV parsing =====
class LineDataset(Dataset):
    """
    Expects a folder structure:
      root/
        images/
          0001.jpg
          ...
        labels.csv  # rows: filename,text   (commas in text allowed if quoted)
    """

    def __init__(self, root, img_dir="images", labels="labels.csv", img_h=32, max_w=512):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.items = []

        labels_path = os.path.join(root, labels)
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels file not found at: {labels_path}")

        # Robust CSV load: tolerate commas, headers, BOM; join extra columns as text
        with open(labels_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"', escapechar="\\")
            for row in reader:
                if not row:
                    continue
                # Skip a possible header row
                first = row[0].strip().lower()
                if first in {"filename", "file", "image"}:
                    continue

                if len(row) >= 2:
                    fn = row[0].strip()
                    text = ",".join(row[1:]).strip()
                else:
                    # Fallback for accidental single-field rows using semicolon
                    if ";" in row[0]:
                        parts = row[0].split(";", 1)
                        fn = parts[0].strip()
                        text = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        # Skip malformed rows
                        continue

                if fn and text:
                    self.items.append((fn, text))

        if len(self.items) == 0:
            raise RuntimeError(f"No labeled items found in {labels_path}. Check CSV formatting.")

        self.img_h = img_h
        self.max_w = max_w
        self.to_tensor = T.ToTensor()

    def _resize_keep_ratio(self, img: Image.Image):
        w, h = img.size
        new_h = self.img_h
        new_w = int(w * (new_h / h))
        img = img.convert("L").resize((min(new_w, self.max_w), new_h), Image.BILINEAR)
        # Pad to max_w on the right
        if img.size[0] < self.max_w:
            canvas = Image.new("L", (self.max_w, new_h), 255)
            canvas.paste(img, (0, 0))
            img = canvas
        return img

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, text = self.items[idx]
        img_path = os.path.join(self.img_dir, fn)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("L")
        img = self._resize_keep_ratio(img)
        return self.to_tensor(img), text


# ===== Tokenization helpers =====
def encode_text(s: str) -> torch.Tensor:
    """Map string to indices (skip unknown chars)."""
    return torch.tensor([char2idx[c] for c in s if c in char2idx], dtype=torch.long)


# ===== CRNN model =====
class CRNN(nn.Module):
    def __init__(self, n_classes=len(CHARS) + 1, cnn_out=256, rnn_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),          # H/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),        # H/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),  # H/8
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),  # H/16
            nn.Conv2d(512, cnn_out, 2, 1, 0), nn.ReLU(),                        # squeeze height
        )
        self.rnn = nn.LSTM(cnn_out, rnn_hidden, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(rnn_hidden * 2, n_classes)

    def forward(self, x):
        # x: [N,1,H,W]
        feats = self.cnn(x)           # [N,C,H',W']
        feats = feats.squeeze(2)      # [N,C,W']  (H' -> 1 ideally)
        feats = feats.permute(2, 0, 1)  # [W',N,C] treat width as time
        seq, _ = self.rnn(feats)      # [T,N,2H]
        logits = self.fc(seq)         # [T,N,C]
        return logits


# ===== Collate & training loops =====
def collate(batch):
    imgs, texts = zip(*batch)
    imgs = torch.stack(imgs, 0)  # [N,1,H,W]
    targets = [encode_text(t) for t in texts]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    if targets:
        targets = torch.cat(targets)
    else:
        targets = torch.tensor([], dtype=torch.long)
    return imgs, texts, targets, target_lengths


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for imgs, _, targets, target_lengths in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        logits = model(imgs)  # [T,N,C]
        T_, N, C = logits.size()
        input_lengths = torch.full((N,), T_, dtype=torch.long)
        loss = criterion(logits.log_softmax(2), targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for imgs, _, targets, target_lengths in loader:
        imgs = imgs.to(device)
        logits = model(imgs)  # [T,N,C]
        T_, N, C = logits.size()
        input_lengths = torch.full((N,), T_, dtype=torch.long)
        loss = criterion(logits.log_softmax(2), targets, input_lengths, target_lengths)
        total += loss.item()
    return total / max(1, len(loader))


# ===== Main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dataset/train", help="Path to train split folder")
    ap.add_argument("--val",   default="dataset/val",   help="Path to val split folder")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--out",    default="crnn_handwriting.pt")
    args = ap.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Datasets & loaders
    train_ds = LineDataset(args.train)
    val_ds   = LineDataset(args.val)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )
    val_dl   = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )

    # Model, loss, optimizer
    model = CRNN().to(device)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best = math.inf
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_dl, optimizer, criterion, device)
        va = eval_loss(model, val_dl, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}: train {tr:.4f} | val {va:.4f}")
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "charset": CHARS}, args.out)
            print(f"  ✓ Saved best model to {args.out}")

    print("Training complete.")


if __name__ == "__main__":
    main()
