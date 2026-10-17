"""IM53: 약점 3(작은 부분 생성) 학습 자료 — htrain 원본(im_h_raw, role htrain)에 마스크 넓이 5~15% 되붙임 국소 편집(im42 v7과 같은 계산, 넓이만 다름) → im_set_h3s.
서버(nadan, GPU 0). 사용: python im53_small_v7.py <n> [role=htrain] [set=im_set_h3s] [prefix=h3s]  (재개 가능)"""
import sys, csv, random, zlib, time, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, to_np, to_img, feather, blob_mask, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
from im15_fakes_v5 import INP, NEG, object_mask  # noqa
from im15c_pgen_v6 import pipe  # noqa
N = int(sys.argv[1]); ROLE = sys.argv[2] if len(sys.argv) > 2 else "htrain"; SN = sys.argv[3] if len(sys.argv) > 3 else "im_set_h3s"; PFX = sys.argv[4] if len(sys.argv) > 4 else "h3s"
RAW = LOCAL / (sys.argv[5] if len(sys.argv) > 5 else "im_h_raw"); OUT = LOCAL / SN; (OUT / "img").mkdir(parents=True, exist_ok=True); (OUT / "mask").mkdir(exist_ok=True)
FN = ["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"]
rows = [r for r in csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")) if r["role"] == ROLE]
rows = sorted(rows, key=lambda r: zlib.crc32(f"{PFX}|{r['file']}".encode()))[:N]
mp = OUT / "manifest.csv"; done = {r["file"] for r in csv.DictReader(open(mp, encoding="utf-8"))} if mp.exists() else set()
f = open(mp, "a", newline="", encoding="utf-8"); w = csv.DictWriter(f, fieldnames=FN)
if f.tell() == 0: w.writeheader()
t = time.time(); k = 0
for i, r in enumerate(rows):
    fn = f"{PFX}_{i:05d}.jpg"
    if fn in done: continue
    try:
        lr = random.Random(zlib.crc32(f"{fn}|v7s".encode())); g = torch.Generator("cpu").manual_seed(lr.randint(0, 2**31))
        a = to_np(dist_cover(Image.open(RAW / r["tar"] / r["file"]).convert("RGB"), lr)); h, ww = a.shape[:2]
        m = object_mask(a, lr, .05, .15) if lr.random() < .5 else blob_mask(a.shape, lr, (.05, .15))
        ys, xs = np.where(m > 0); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        side = int(min(max(max(y1 - y0, x1 - x0) * lr.uniform(1.3, 2.0), 384), min(h, ww)))
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2; wy = int(np.clip(cy - side // 2, 0, h - side)); wx = int(np.clip(cx - side // 2, 0, ww - side))
        which = "xl" if lr.random() < .5 else "sd15"; S = 1024 if which == "xl" else 512
        win = Image.fromarray(a[wy:wy + side, wx:wx + side]).resize((S, S), Image.LANCZOS)
        mw = Image.fromarray((m[wy:wy + side, wx:wx + side] * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
        prompt = f"{lr.choice(INP)}, photo, realistic"
        out = pipe(which)(prompt, image=win, mask_image=mw, width=S, height=S, strength=lr.uniform(.85, 1.0), num_inference_steps=lr.randint(25, 35), guidance_scale=lr.uniform(5, 8), negative_prompt=NEG, generator=g).images[0]
        patch = np.asarray(out.resize((side, side), Image.LANCZOS)).astype(np.float32)
        o = a.astype(np.float32).copy(); fm = feather(m, lr.randint(1, 4))[..., None][wy:wy + side, wx:wx + side]
        o[wy:wy + side, wx:wx + side] = patch * fm + o[wy:wy + side, wx:wx + side] * (1 - fm)
        (OUT / "img" / fn).write_bytes(jpeg_bytes(to_img(o))); Image.fromarray((m * 255).astype(np.uint8)).save(OUT / "mask" / fn.replace(".jpg", ".png"))
        w.writerow(dict(file=fn, label=1, kind="partial_gen_v7s", info=f"{which}|{prompt}|area={m.mean():.3f}|win={side}", source=r["file"], group=r["group"], farm_code=r["farm_code"], tar=r["tar"], other="")); f.flush(); k += 1
    except Exception as e: print("FAIL", fn, repr(e)[:200], flush=True)
    if k and k % 50 == 0: print(f"v7s {k} {time.time() - t:.0f}s", flush=True)
print("V7S_DONE", k, flush=True)
