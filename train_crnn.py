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
        img = img.resize((min(new_w, self.max_w), self.img_h), Image.BILINEAR).convert("L")
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
        f = self.cnn(x).squeeze(2)     # [N,C,W']
        f = f.permute(2,0,1)           # [W',N,C]
        y,_ = self.rnn(f)              # [T,N,2H]
        return self.fc(y)              # [T,N,C]

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
        logits = model(imgs)                # [T,N,C]
        T_,N,C = logits.size()
        in_lens = torch.full((N,), T_, dtype=torch.long)
        loss = crit(logits.log_softmax(2), targets, in_lens, targ_lens)
        loss.backward(); opt.step()
        total += loss.item()
    return total/len(loader)

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
    return total/len(loader)

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
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate, num_workers=2)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=2)

    model = CRNN().to(device)
    crit  = nn.CTCLoss(blank=0, zero_infinity=True)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
