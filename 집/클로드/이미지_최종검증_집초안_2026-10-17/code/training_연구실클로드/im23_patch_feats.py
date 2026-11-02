"""IM23: 후보 L 조각 특징 — open_clip ViT-B-16(openai), im22 grid(224 창, 112 간격, 54조각), 조각마다 L2 정규화, fp16. 서버 GPU.
사용: python im23_patch_feats.py <name> <img_dir> <manifest.csv>   출력 local/feats/<name>_P16.npz (files, feats=(N,54,512))
"""
import sys, csv, time, numpy as np, torch, open_clip
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im22_patch_masks import grid, S  # noqa
LOCAL = Path(__file__).resolve().parents[1] / "local"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1); STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
_M = {}

def model():
    if "m" not in _M:
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai", cache_dir=str(LOCAL / "weights/openclip")); _M["m"] = m.visual.eval().cuda().half()
    return _M["m"]

@torch.no_grad()
def embed(img):
    a = np.asarray(img.convert("RGB")); h, w = a.shape[:2]
    x = torch.stack([(torch.from_numpy(a[y:y + S, x0:x0 + S].astype(np.float32) / 255).permute(2, 0, 1) - MEAN) / STD for x0, y in grid(w, h)]).cuda().half()
    e = model()(x).float(); return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16)

def main(name, img_dir, man):
    files = [r["file"] for r in csv.DictReader(open(man, encoding="utf-8"))]; out = []; t = time.time()
    for n, f in enumerate(files):
        e = embed(Image.open(Path(img_dir) / f)); assert len(e) == 54, (f, len(e)); out.append(e)
        if n % 2000 == 0: print(name, n, f"{time.time()-t:.0f}s", flush=True)
    np.savez(LOCAL / "feats" / f"{name}_P16.npz", files=np.array(files), feats=np.stack(out)); print("saved", name, len(out), f"{time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    main(*sys.argv[1:4])
