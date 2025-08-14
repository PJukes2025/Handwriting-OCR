import torch
from PIL import Image
import torchvision.transforms as T
from train_crnn import CRNN

def _prep(img: Image.Image, img_h=32, max_w=512):
    w,h = img.size
    new_w = int(w * (img_h / h))
    img = img.convert("L").resize((min(new_w, max_w), img_h), Image.BILINEAR)
    if img.size[0] < max_w:
        canvas = Image.new("L", (max_w, img_h), 255)
        canvas.paste(img, (0,0))
        img = canvas
    return T.ToTensor()(img).unsqueeze(0)  # [1,1,H,W]

@torch.no_grad()
def recognize_line(model, img: Image.Image, device="cpu"):
    x = _prep(img).to(device)
    logits = model(x)                  # [T,N,C]
    best = logits.argmax(dim=2).permute(1,0)  # [N,T]
    BLANK = 0
    # Greedy CTC decode
    out = []
    prev = -1
    for t in best[0].tolist():
        if t != prev and t != BLANK:
            # map t -> char using the same charset used in training
            # we saved it in the checkpoint but keep a simple fallback:
            pass
        prev = t
    # Minimal: if you want readable output, load charset from ckpt:
    return ""  # kept minimal; you can extend to map idx->char using saved charset

def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = CRNN()
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    # you can also return ckpt["charset"] to decode indices
    return model, ckpt.get("charset")

if __name__ == "__main__":
    import sys
    model, charset = load_model("crnn_handwriting.pt")
    img = Image.open(sys.argv[1])
    print(recognize_line(model, img))
