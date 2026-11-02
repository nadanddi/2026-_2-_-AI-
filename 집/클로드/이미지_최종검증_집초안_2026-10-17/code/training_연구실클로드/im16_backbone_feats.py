"""IM16: 백본 비교용 특징 추출 (IM_백본비교_v1.md). 서버 실행용(CUDA_VISIBLE_DEVICES로 GPU 고정).
보기: 전체(짧은 변 S로 축소 + 가운데 S 크롭) 1 + 원해상도 S×S 3x3 격자 9 (S = 모델 입력 크기). 각 보기의 이미지 임베딩 → (N, 10, D) fp16.
사용: python im16_backbone_feats.py <백본ID> <name> <img_dir> <manifest.csv|list>
"""
import sys, csv, time, numpy as np, torch
from pathlib import Path
from PIL import Image

LOCAL = Path(__file__).resolve().parents[1] / "local"
OC = {"B1": ("ViT-L-14-quickgelu", "openai"), "B2": ("ViT-L-14-336", "openai"), "B3": ("ViT-L-14", "laion2b_s32b_b82k"),
      "B4": ("ViT-SO400M-14-SigLIP", "webli"), "B5": ("EVA02-L-14", "merged2b_s4b_b131k"), "B6": ("convnext_large_d_320", "laion2b_s29b_b131k_ft_soup")}

def load(bid, dev):
    if bid in OC:
        import open_clip
        name, pre = OC[bid]; m, _, tf = open_clip.create_model_and_transforms(name, pretrained=pre, cache_dir=str(LOCAL / "weights/openclip"))
        norm = [t for t in tf.transforms if t.__class__.__name__ == "Normalize"][0]; S = m.visual.image_size
        S = S[0] if isinstance(S, (tuple, list)) else S; v = m.visual.eval().to(dev).half()
        return (lambda x: v(x)), S, torch.tensor(norm.mean).view(3, 1, 1), torch.tensor(norm.std).view(3, 1, 1)
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-large", cache_dir=str(LOCAL / "weights/hf")).eval().to(dev).half()
    def f(x): h = m(pixel_values=x).last_hidden_state; return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
    return f, 224, torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def views(img, S):
    w, h = img.size; s = S / min(w, h); r = img.resize((max(S, round(w * s)), max(S, round(h * s))), Image.BICUBIC)
    x0, y0 = (r.size[0] - S) // 2, (r.size[1] - S) // 2; out = [r.crop((x0, y0, x0 + S, y0 + S))]
    for yy in np.linspace(0, h - S, 3).astype(int):
        for xx in np.linspace(0, w - S, 3).astype(int): out.append(img.crop((xx, yy, xx + S, yy + S)))
    return out

@torch.no_grad()
def main(bid, name, img_dir, lst):
    files = [r["file"] for r in csv.DictReader(open(lst, encoding="utf-8"))] if lst.endswith(".csv") else [l.strip() for l in open(lst, encoding="utf-8") if l.strip()]
    dev = "cuda"; f, S, MEAN, STD = load(bid, dev); out = []; t = time.time(); B = 8
    for i in range(0, len(files), B):
        xs = []
        for fn in files[i:i + B]:
            xs += [(torch.from_numpy(np.asarray(z, np.float32) / 255).permute(2, 0, 1) - MEAN) / STD for z in views(Image.open(Path(img_dir) / fn).convert("RGB"), S)]
        e = f(torch.stack(xs).to(dev).half()).float().cpu().numpy(); out.append(e.reshape(-1, 10, e.shape[-1]).astype(np.float16))
        if (i // B) % 100 == 0: print(f"{bid} {name} {i}/{len(files)} {time.time()-t:.0f}s", flush=True)
    (LOCAL / "feats").mkdir(exist_ok=True)
    np.savez_compressed(LOCAL / "feats" / f"{name}_{bid}.npz", files=np.array(files), feats=np.concatenate(out)); print("saved", bid, name, f"{time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    main(*sys.argv[1:5])
