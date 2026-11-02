"""IM04c: 후보 D2용 DINOv2-B/14 특징 (HF facebook/dinov2-base, 로컬 캐시).
보기: 전체 축소(v0) + cover 격자 원해상도 224 조각 20개 → 각 CLS(768)와 패치토큰 평균(768)을 이어 1536차원.
저장: local/feats/<name>_dino.npz  feats=(N, 21, 1536) fp16  (0번 = 전체 보기)
사용: python im04c_dino_feats.py <name> <img_dir> <manifest.csv|list>
"""
import sys, csv, time, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im04b_clip_multi import grid, glob_view  # noqa

LOCAL = Path(__file__).resolve().parents[1] / "local"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

@torch.no_grad()
def main(name, img_dir, lst):
    from transformers import AutoModel
    files = [r["file"] for r in csv.DictReader(open(lst, encoding="utf-8"))] if lst.endswith(".csv") else [l.strip() for l in open(lst, encoding="utf-8") if l.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=str(LOCAL / "weights/hf")).eval().to(dev)
    if dev == "cuda": m = m.half()
    out = []; t = time.time()
    for n, f in enumerate(files):
        img = Image.open(Path(img_dir) / f).convert("RGB"); views = [glob_view(img)] + grid(img)
        x = torch.stack([(torch.from_numpy(np.asarray(z, np.float32) / 255).permute(2, 0, 1) - MEAN) / STD for z in views]).to(dev)
        h = m(pixel_values=x.half() if dev == "cuda" else x).last_hidden_state.float()
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy().astype(np.float16))
        if n % 200 == 0: print(f"{n}/{len(files)} {time.time()-t:.0f}s", flush=True)
    np.savez_compressed(LOCAL / "feats" / f"{name}_dino.npz", files=np.array(files), feats=np.stack(out)); print("saved", len(files), f"{time.time()-t:.0f}s")

if __name__ == "__main__":
    main(*sys.argv[1:4])
