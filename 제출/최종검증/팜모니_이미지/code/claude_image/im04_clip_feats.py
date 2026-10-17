"""IM04: CLIP ViT-L/14(openai) 이미지 특징 추출 → npz 캐시.
보기(view) 10개: v0 = 전체(짧은 변 224 리사이즈 + 중앙 크롭), v1~v9 = 원해상도 224 크롭 3x3 격자(전체 커버).
사용: python im04_clip_feats.py <name> <img_dir> <list.txt|manifest.csv>
"""
import sys, csv, time, numpy as np, torch, open_clip
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "local"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

def load_model(dev):
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai", cache_dir=str(LOCAL / "weights/openclip"))
    v = m.visual.eval().to(dev)
    if dev == "cuda": v = v.half()
    return v

def views(img):
    w, h = img.size
    s = 224 / min(w, h)
    r = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    rw, rh = r.size; x0, y0 = (rw - 224) // 2, (rh - 224) // 2
    out = [r.crop((x0, y0, x0 + 224, y0 + 224))]
    for yy in np.linspace(0, h - 224, 3).astype(int):
        for xx in np.linspace(0, w - 224, 3).astype(int):
            out.append(img.crop((xx, yy, xx + 224, yy + 224)))
    return out

def to_t(im):
    a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)
    return (a - MEAN) / STD

@torch.no_grad()
def run(name, img_dir, files, bs=4):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    v = load_model(dev); feats = []; t = time.time()
    for i in range(0, len(files), bs):
        batch = []
        for f in files[i:i + bs]:
            batch += [to_t(x) for x in views(Image.open(Path(img_dir) / f).convert("RGB"))]
        x = torch.stack(batch).to(dev)
        if dev == "cuda": x = x.half()
        e = v(x).float().cpu().numpy().reshape(-1, 10, 768 if e_dim(v) == 768 else e_dim(v))
        feats.append(e)
        if (i // bs) % 50 == 0: print(f"{i}/{len(files)} {time.time()-t:.0f}s", flush=True)
    F = np.concatenate(feats)
    out = LOCAL / "feats"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{name}_clipL14.npz", feats=F.astype(np.float16), files=np.array(files))
    print("saved", F.shape, f"{time.time()-t:.0f}s")

def e_dim(v): return v.output_dim

if __name__ == "__main__":
    name, img_dir, lst = sys.argv[1], sys.argv[2], sys.argv[3]
    if lst.endswith(".csv"):
        files = [r["file"] for r in csv.DictReader(open(lst, encoding="utf-8"))]
    else:
        files = [l.strip() for l in open(lst, encoding="utf-8") if l.strip()]
    run(name, img_dir, files)
