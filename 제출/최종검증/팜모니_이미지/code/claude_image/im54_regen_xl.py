"""IM54: 약점 2(GPT식 전체 재그리기) 대리 자료 — SDXL 인페인팅에 전체 마스크·강도 0.3~0.6 → 구도 유지 전체 재그리기(img2img와 같은 효과).
서버(nadan, GPU 0). 사용: python im54_regen_xl.py <n> <role> <set> <prefix> <raw_dir>   예) 500 htrain im_set_h3g h3g im_h_raw  (재개 가능)"""
import sys, csv, random, zlib, time, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, to_img, LOCAL  # noqa
from im03c_make_fakes_v3 import dist_cover  # noqa
from im15_fakes_v5 import NEG  # noqa
from im15c_pgen_v6 import pipe  # noqa
N, ROLE, SN, PFX, RAWN = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
RAW = LOCAL / RAWN; OUT = LOCAL / SN; (OUT / "img").mkdir(parents=True, exist_ok=True)
PR = ["a photo of strawberry plants growing in a greenhouse", "strawberry plants on a raised bed in a vertical farm, photo", "close-up photo of strawberry leaves and fruits in a greenhouse",
      "hydroponic strawberry cultivation, realistic photo"]
FN = ["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"]
rows = sorted([r for r in csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")) if r["role"] == ROLE], key=lambda r: zlib.crc32(f"{PFX}|{r['file']}".encode()))[:N]
mp = OUT / "manifest.csv"; done = {r["file"] for r in csv.DictReader(open(mp, encoding="utf-8"))} if mp.exists() else set()
f = open(mp, "a", newline="", encoding="utf-8"); w = csv.DictWriter(f, fieldnames=FN)
if f.tell() == 0: w.writeheader()
t = time.time(); k = 0; P = pipe("xl")
for i, r in enumerate(rows):
    fn = f"{PFX}_{i:05d}.jpg"
    if fn in done: continue
    try:
        lr = random.Random(zlib.crc32(f"{fn}|rg".encode())); g = torch.Generator("cpu").manual_seed(lr.randint(0, 2**31))
        base = dist_cover(Image.open(RAW / r["tar"] / r["file"]).convert("RGB"), lr); W_, H_ = base.size
        st = lr.uniform(.3, .6); pr = lr.choice(PR) + ", realistic, natural lighting"
        out = P(pr, image=base, mask_image=Image.new("L", base.size, 255), width=W_, height=H_, strength=st, num_inference_steps=lr.randint(25, 35),
                guidance_scale=lr.uniform(4, 7), negative_prompt=NEG, generator=g).images[0]
        if out.size != base.size: out = out.resize(base.size, Image.LANCZOS)
        (OUT / "img" / fn).write_bytes(jpeg_bytes(out))
        w.writerow(dict(file=fn, label=1, kind="regen_xl", info=f"strength={st:.2f}|{pr}", source=r["file"], group=r["group"], farm_code=r["farm_code"], tar=r["tar"], other="")); f.flush(); k += 1
    except Exception as e: print("FAIL", fn, repr(e)[:200], flush=True)
    if k and k % 50 == 0: print(f"rg {k} {time.time() - t:.0f}s", flush=True)
print("RG_DONE", k, flush=True)
