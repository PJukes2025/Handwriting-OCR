import os, csv, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import torchvision.transforms as T

# ---------- CRNN model ----------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B,1,48,W]
        conv = self.cnn(x)       # [B,512,1,W’]
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)   # [B,512,W’]
        conv = conv.permute(0,2,1) # [B,W’,512]
        rnn_out, _ = self.rnn(conv)
        out = self.fc(rnn_out)   # [B,W’,C]
        return out

# ---------- Preprocessing (same as inference) ----------
def preprocess_img(img):
    img = img.convert("L")
    if ImageOps.invert(img).getextrema()[0] < img.getextrema()[0]:
        img = ImageOps.invert(img)
    w, h = img.size
    new_w = int(w * (48 / h))
    img = img.resize((min(new_w,1024), 48), Image.BILINEAR)
    if img.size[0] < 1024:
        canvas = Image.new("L",(1024,48),255)
        canvas.paste(img,(0,0))
        img = canvas
    x = T.ToTensor()(img)
    x = T.Normalize((0.5,), (0.5,))(x)
    return x

# ---------- Dataset ----------
class LineDataset(Dataset):
    def __init__(self, root):
        labels_path = os.path.join(root,"labels.csv")
        self.items = []
        with open(labels_path, encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if len(row) < 2: continue
                fn,text = row[0].strip(),",".join(row[1:]).strip()
                self.items.append((os.path.join(root,"images",fn), text))
        if not self.items:
            raise RuntimeError(f"No items in {labels_path}")
        self.charset = ["<blank>"] + sorted(set("".join(t for _,t in self.items)))

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        fn,text = self.items[i]
        img = Image.open(fn)
        x = preprocess_img(img)
        target = [self.charset.index(c) for c in text if c in self.charset]
        return x, torch.tensor(target)

# ---------- Collate ----------
def collate(batch):
    xs,ys = zip(*batch)
    xs = torch.stack(xs,0)
    y_lens = [len(y) for y in ys]
    y = torch.cat(ys)
    return xs, y, y_lens

# ---------- Train ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",required=True)
    ap.add_argument("--val",required=True)
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--out",default="crnn_handwriting.pt")
    args=ap.parse_args()

    train_ds = LineDataset(args.train)
    val_ds = LineDataset(args.val)
    charset = train_ds.charset
    model = CRNN(len(charset)).to("cpu")
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    train_loader = DataLoader(train_ds,batch_size=args.batch,shuffle=True,collate_fn=collate)
    val_loader = DataLoader(val_ds,batch_size=args.batch,shuffle=False,collate_fn=collate)

    for epoch in range(args.epochs):
        model.train()
        for xs,y,y_lens in train_loader:
            xs=xs
            logits=model(xs)
            log_probs=logits.log_softmax(2)
            input_lengths=torch.full((xs.size(0),),logits.size(1),dtype=torch.long)
            target_lengths=torch.tensor(y_lens)
            loss=criterion(log_probs.permute(1,0,2),y,input_lengths,target_lengths)
            opt.zero_grad(); loss.backward(); opt.step()
        print("Epoch",epoch+1,"loss",loss.item())

    torch.save({"model":model.state_dict(),"charset":charset},args.out)

if __name__=="__main__":
    main()
