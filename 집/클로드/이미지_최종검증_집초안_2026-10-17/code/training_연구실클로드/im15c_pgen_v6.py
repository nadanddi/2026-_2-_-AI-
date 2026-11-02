"""IM15c: 가짜 제작 6판 — 부분생성 되붙임판 (IM_부분생성_D1_v1.md). 서버 GPU.
v5 partial_gen_xl은 사진 전체가 VAE를 두 번 거쳐 전역 지문이 남았다(비평 10-13). 6판은:
 - 전역 img2img 없음. 마스크 둘레 창(정사각, 마스크 상자의 1.3~2배, 최소 384)만 잘라 인페인팅 → 원래 크기로 → **마스크 영역만** 깃털로 원본에 되붙임.
 - 인페인터 반반: SDXL-inpaint(창 1024) / SD1.5-inpaint(창 512).
 - 마스크 넓이 2~15% (물체 마스크 50% / 덩어리 50%).
사용: python im15c_pgen_v6.py <n> [gpu=0]   출력 local/im_set_v6/pgen6_*.jpg, manifest.csv (재개 가능)
"""
import sys, csv, random, zlib, time, os
import numpy as np, cv2
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, to_np, to_img, feather, blob_mask, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
from im15_fakes_v5 import INP, NEG, rawp, sources, object_mask  # noqa
OUT = LOCAL / "im_set_v6"; _P = {}

def pipe(which):
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline
    if which not in _P:
        W = LOCAL / "weights"
        p = (StableDiffusionXLInpaintPipeline.from_pretrained(W / "sdxl-inpainting", torch_dtype=torch.float16, variant="fp16") if which == "xl"
             else StableDiffusionInpaintPipeline.from_pretrained(W / "stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16", safety_checker=None))
        p = p.to("cuda"); p.set_progress_bar_config(disable=True); _P[which] = p
    return _P[which]

def make(i, man):
    import torch
    lr = random.Random(zlib.crc32(f"v6|pgen|{i}".encode())); g = torch.Generator("cpu").manual_seed(lr.randint(0, 2**31))
    r = man[lr.randrange(len(man))]; base = dist_cover(Image.open(rawp(r["file"])).convert("RGB"), lr); a = to_np(base); h, w = a.shape[:2]
    m = object_mask(a, lr, .02, .15) if lr.random() < .5 else blob_mask(a.shape, lr, (.02, .15))
    ys, xs = np.where(m > 0); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = int(min(max(max(y1 - y0, x1 - x0) * lr.uniform(1.3, 2.0), 384), min(h, w)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2; wy = int(np.clip(cy - side // 2, 0, h - side)); wx = int(np.clip(cx - side // 2, 0, w - side))
    which = "xl" if lr.random() < .5 else "sd15"; S = 1024 if which == "xl" else 512
    win = Image.fromarray(a[wy:wy + side, wx:wx + side]).resize((S, S), Image.LANCZOS)
    mw = Image.fromarray((m[wy:wy + side, wx:wx + side] * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
    prompt = f"{lr.choice(INP)}, photo, realistic"
    out = pipe(which)(prompt, image=win, mask_image=mw, width=S, height=S, strength=lr.uniform(.85, 1.0), num_inference_steps=lr.randint(25, 35),
                      guidance_scale=lr.uniform(5, 8), negative_prompt=NEG, generator=g).images[0]
    patch = np.asarray(out.resize((side, side), Image.LANCZOS)).astype(np.float32)
    o = a.astype(np.float32).copy(); f = feather(m, lr.randint(1, 4))[..., None][wy:wy + side, wx:wx + side]
    o[wy:wy + side, wx:wx + side] = patch * f + o[wy:wy + side, wx:wx + side] * (1 - f)
    meta = dict(source=r["file"], group=r["group"], farm_code=r["farm_code"], tar=r["tar"])
    return to_img(o), meta, f"{which}|{prompt}|area={m.mean():.3f}|win={side}"

def main(n, gpu="0"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    (OUT / "img").mkdir(parents=True, exist_ok=True); man = sources(); mp = OUT / "manifest.csv"
    done = {r["file"] for r in csv.DictReader(open(mp, encoding="utf-8"))} if mp.exists() else set()
    f = open(mp, "a", newline="", encoding="utf-8"); wr = csv.DictWriter(f, fieldnames=["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"])
    if f.tell() == 0: wr.writeheader()
    t = time.time(); k = 0
    for i in range(n):
        fn = f"pgen6_{i:05d}.jpg"
        if fn in done: continue
        try: img, meta, info = make(i, man)
        except Exception as e: print("FAIL", i, repr(e)[:200], flush=True); continue
        (OUT / "img" / fn).write_bytes(jpeg_bytes(img)); wr.writerow(dict(file=fn, label=1, kind="partial_gen_v6", info=info, other="", **meta)); f.flush(); k += 1
        if k % 50 == 0: print(f"pgen6 {k} {time.time()-t:.0f}s", flush=True)
    print("done pgen6", k, f"{time.time()-t:.0f}s"); print("GEN_V6_DONE", flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "0")
