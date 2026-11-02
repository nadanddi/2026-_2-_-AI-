"""IM04b: 후보 D1·D3용 CLIP ViT-L/14 특징 — 한 번의 실행으로
 - D3: cover 격자 원해상도 224 조각 전부(768x1024 → 4x5=20, 1024x768 → 5x4=20)의 최종 임베딩(768) → (N, 20, 768)
 - D1: 같은 조각들의 중간층 CLS 토큰(층 8·12·16·20, 각 1024차원)의 조각 평균 → (N, 4, 1024)
       + 전체 축소 보기(v0)의 같은 층 CLS → (N, 4, 1024)
저장: local/feats/<name>_clipML.npz (fp16)
사용: python im04b_clip_multi.py <name> <img_dir> <manifest.csv|list>
"""
import sys, csv, time, math, numpy as np, torch, open_clip
from pathlib import Path
from PIL import Image

LOCAL = Path(__file__).resolve().parents[1] / "local"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1); STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
LAYERS = (8, 12, 16, 20)

def grid(img):
    w, h = img.size; xs = np.linspace(0, w - 224, math.ceil(w / 224)).round().astype(int); ys = np.linspace(0, h - 224, math.ceil(h / 224)).round().astype(int)
    return [img.crop((x, y, x + 224, y + 224)) for y in ys for x in xs]

def glob_view(img):
    w, h = img.size; s = 224 / min(w, h); r = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    x0, y0 = (r.size[0] - 224) // 2, (r.size[1] - 224) // 2; return r.crop((x0, y0, x0 + 224, y0 + 224))

def tt(im): return (torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1) - MEAN) / STD

@torch.no_grad()
def main(name, img_dir, lst):
    files = [r["file"] for r in csv.DictReader(open(lst, encoding="utf-8"))] if lst.endswith(".csv") else [l.strip() for l in open(lst, encoding="utf-8") if l.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai", cache_dir=str(LOCAL / "weights/openclip"))
    v = m.visual.eval().to(dev); v = v.half() if dev == "cuda" else v
    cap = {}
    hooks = [v.transformer.resblocks[L - 1].register_forward_hook(lambda mod, i, o, L=L: cap.__setitem__(L, o)) for L in LAYERS]
    P, MLp, MLg = [], [], []; t = time.time()
    for n, f in enumerate(files):
        img = Image.open(Path(img_dir) / f).convert("RGB"); views = [glob_view(img)] + grid(img)
        x = torch.stack([tt(z) for z in views]).to(dev); x = x.half() if dev == "cuda" else x
        e = v(x).float().cpu().numpy()
        # open_clip 트랜스포머 출력 모양: (L, N, D) 또는 (N, L, D) — CLS(토큰 0) 추출
        cls = []
        for L in LAYERS:
            o = cap[L]; o = o.permute(1, 0, 2) if o.shape[1] == len(views) and o.shape[0] != len(views) else o
            cls.append(o[:, 0].float().cpu().numpy())
        cls = np.stack(cls, 1)   # (views, 4, 1024)
        P.append(e[1:]); MLp.append(cls[1:].mean(0)); MLg.append(cls[0])
        if n % 200 == 0: print(f"{n}/{len(files)} {time.time()-t:.0f}s", flush=True)
    for h in hooks: h.remove()
    np.savez_compressed(LOCAL / "feats" / f"{name}_clipML.npz", files=np.array(files), patches=np.stack(P).astype(np.float16),
                        ml_patch=np.stack(MLp).astype(np.float16), ml_glob=np.stack(MLg).astype(np.float16))
    print("saved", len(files), f"{time.time()-t:.0f}s")

if __name__ == "__main__":
    main(*sys.argv[1:4])
