"""IM16b: 후보 묶음 E 특징 추출 (IM_해상도후보E_v1.md). 서버 실행용.
변형 ID → (open_clip 모델, pretrained, 조각 배치, 반전 TTA)
 B2c20 : ViT-L-14-336/openai, cover 4×5(20), 반전 없음
 B2c20f: 위 + 좌우 반전 평균
 B2f   : ViT-L-14-336/openai, 3×3(9), 좌우 반전 평균
 SG384 : ViT-SO400M-14-SigLIP-384/webli, 3×3(9), 반전 없음
출력: local/feats/<set>_<ID>.npz  feats=(N, 1+V, D) — 0번은 자리채움(im05k bb()가 [:,1:] 평균을 쓰므로 형식 맞춤)
사용: python im16b_feats_E.py <ID> <name> <img_dir> <manifest.csv>
"""
import sys, csv, time, math, numpy as np, torch, open_clip
from pathlib import Path
from PIL import Image
LOCAL = Path(__file__).resolve().parents[1] / "local"
CFG = {"B2c20": ("ViT-L-14-336", "openai", "cover", False), "B2c20f": ("ViT-L-14-336", "openai", "cover", True),
       "B2f": ("ViT-L-14-336", "openai", "3x3", True), "SG384": ("ViT-SO400M-14-SigLIP-384", "webli", "3x3", False)}

def crops(img, S, mode):
    w, h = img.size
    if mode == "cover":
        xs = np.linspace(0, w - S, math.ceil(w / S)).round().astype(int); ys = np.linspace(0, h - S, math.ceil(h / S)).round().astype(int)
    else:
        xs = np.linspace(0, w - S, 3).astype(int); ys = np.linspace(0, h - S, 3).astype(int)
    return [img.crop((int(x), int(y), int(x) + S, int(y) + S)) for y in ys for x in xs]

@torch.no_grad()
def main(vid, name, img_dir, man):
    arch, pre, mode, flip = CFG[vid]
    m, _, tf = open_clip.create_model_and_transforms(arch, pretrained=pre, cache_dir=str(LOCAL / "weights/openclip"))
    norm = [t for t in tf.transforms if t.__class__.__name__ == "Normalize"][0]; S = m.visual.image_size; S = S[0] if isinstance(S, (tuple, list)) else S
    MEAN, STD = torch.tensor(norm.mean).view(3, 1, 1), torch.tensor(norm.std).view(3, 1, 1)
    v = m.visual.eval().cuda().half(); files = [r["file"] for r in csv.DictReader(open(man, encoding="utf-8"))]; out = []; t = time.time()
    for n, f in enumerate(files):
        cs = crops(Image.open(Path(img_dir) / f).convert("RGB"), S, mode)
        x = torch.stack([(torch.from_numpy(np.asarray(c, np.float32) / 255).permute(2, 0, 1) - MEAN) / STD for c in cs]).cuda().half()
        e = v(x).float()
        if flip: e = (e / e.norm(dim=-1, keepdim=True) + (lambda q: q / q.norm(dim=-1, keepdim=True))(v(torch.flip(x, dims=[3])).float())) / 2
        e = e.cpu().numpy(); out.append(np.concatenate([e[:1] * 0, e], 0).astype(np.float16))
        if n % 1000 == 0: print(vid, name, n, f"{time.time()-t:.0f}s", flush=True)
    V = max(len(o) for o in out); D = out[0].shape[1]
    F = np.zeros((len(out), V, D), np.float16)
    for i, o in enumerate(out): F[i, :len(o)] = o
    np.savez_compressed(LOCAL / "feats" / f"{name}_{vid}.npz", files=np.array(files), feats=F); print("saved", vid, name, F.shape, f"{time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    main(*sys.argv[1:5])
