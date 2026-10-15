"""IM15d: 가짜 제작 7판 — 되붙임 국소 편집(6판 im15c와 같은 방식, IM_필요이미지_정리_v1.md 1번). 서버 GPU 0, 환경 nadan.
6판과 다른 점: 원본 = 새 AI Hub 추출(im_new_raw, role=gen, 뒤쪽 원본부터), 마스크 넓이 10~30%(6판 진단: <10%는 진짜와 구분 안 돼 라벨 잡음), 마스크를 PNG로 저장.
사용: python im15d_pgen_v7.py <n>   출력 local/im_set_v7/img/pgen7_*.jpg, mask/*.png, manifest.csv (재개 가능)
"""
import sys, csv, random, zlib, time, os
import numpy as np
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, to_np, to_img, feather, blob_mask, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
from im15_fakes_v5 import INP, NEG, object_mask  # noqa
from im15c_pgen_v6 import pipe  # noqa
OUT = LOCAL / "im_set_v7"

def make(i, src):
    import torch
    lr = random.Random(zlib.crc32(f"v7|pgen|{i}".encode())); g = torch.Generator("cpu").manual_seed(lr.randint(0, 2**31))
    base = dist_cover(Image.open(LOCAL / "im_new_raw" / src["tar"] / src["file"]).convert("RGB"), lr); a = to_np(base); h, w = a.shape[:2]
    m = object_mask(a, lr, .10, .30) if lr.random() < .5 else blob_mask(a.shape, lr, (.10, .30))
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
    return to_img(o), m, f"{which}|{prompt}|area={m.mean():.3f}|win={side}"

def main(n):
    rows = [r for r in csv.DictReader(open(LOCAL / "im_new_raw/manifest.csv", encoding="utf-8")) if r["role"] == "gen"][::-1]   # Qwen(앞쪽)과 최대한 안 겹치게 뒤쪽부터
    (OUT / "img").mkdir(parents=True, exist_ok=True); (OUT / "mask").mkdir(exist_ok=True); mp = OUT / "manifest.csv"
    done = {r["file"] for r in csv.DictReader(open(mp, encoding="utf-8"))} if mp.exists() else set()
    f = open(mp, "a", newline="", encoding="utf-8"); wr = csv.DictWriter(f, fieldnames=["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"])
    if f.tell() == 0: wr.writeheader()
    t = time.time(); k = 0
    for i, src in enumerate(rows[:n]):
        fn = f"pgen7_{i:05d}.jpg"
        if fn in done: continue
        try: img, m, info = make(i, src)
        except Exception as e: print("FAIL", i, repr(e)[:200], flush=True); continue
        (OUT / "img" / fn).write_bytes(jpeg_bytes(img)); Image.fromarray((m * 255).astype(np.uint8)).save(OUT / "mask" / fn.replace(".jpg", ".png"))
        wr.writerow(dict(file=fn, label=1, kind="partial_gen_v7", info=info, source=src["file"], group=src["group"], farm_code=src["farm_code"], tar=src["tar"], other="")); f.flush(); k += 1
        if k % 50 == 0: print(f"pgen7 {k} {time.time() - t:.0f}s", flush=True)
    print("done pgen7", k)

if __name__ == "__main__":
    main(int(sys.argv[1]))
