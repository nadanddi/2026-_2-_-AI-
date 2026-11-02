"""IM42: 후보 H 자료 제작 — IM_유형1to5_H_v1.md. 서버(nadan, GPU 0). 원본 = local/im_h_raw(im41 추출).
- htrain 원본 1800 → im_set_h3: 원본마다 진짜 1장(배포 처리) + 가짜 1장
    가짜 유형: filter_jpeg / filter_blur / filter_noise (원본 해상도 필터 → 배포 처리, im03f와 같은 제작기)
               partial_style (im03b와 같은 스타일·마스크, 마스크 PNG 저장)
               partial_gen_v7 (im15d와 같은 되붙임 국소 편집, GPU, 마스크 PNG 저장)
    비율 3:3:3:5:4 (원본 순서 고정 난수)
- hold 원본 900 → im_set_r3(봉인): 원본마다 진짜 또는 가짜 한 장만(반반), 가짜 유형은 위 5종 균등. 묶음은 htrain과 겹치지 않음(im41).
사용: python im42_make_h.py cpu   (필터·부분 스타일·진짜)   /   python im42_make_h.py gpu  (v7, 재개 가능)   /   python im42_make_h.py merge
비평 반영(v2): 모드별 manifest_cpu/gpu.csv → merge로 manifest.csv, 스타일 참조는 같은 role 안에서만, 필터 후 크롭은 별도 시드.
"""
import sys, csv, random, zlib, time, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, to_np, to_img, plant_mask, blob_mask, feather, LOCAL  # noqa
import im03b_make_fakes_v2 as v2  # noqa
from im03c_make_fakes_v3 import dist_cover, raw_filter  # noqa

RAWH = LOCAL / "im_h_raw"; KT = ["filter_jpeg"] * 3 + ["filter_blur"] * 3 + ["filter_noise"] * 3 + ["partial_style"] * 5 + ["partial_gen_v7"] * 4
KH = ["filter_jpeg", "filter_blur", "filter_noise", "partial_style", "partial_gen_v7"]
FN = ["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"]

def pstyle_mask(a, rng, other):   # im03b partial_style과 같은 계산 + 마스크 반환
    st, info = v2.photo_style(a, rng, other)
    m = plant_mask(a) * blob_mask(a.shape, rng, (.1, .45)) if rng.random() < .6 else blob_mask(a.shape, rng, (.05, .35))
    if m.sum() < .02 * m.size: m = blob_mask(a.shape, rng, (.05, .35))
    f = feather(m, rng.randint(1, 10)); return np.clip(st * f[..., None] + a * (1 - f[..., None]), 0, 255).astype(np.uint8), f, info

def plan():
    rows = list(csv.DictReader(open(RAWH / "manifest.csv", encoding="utf-8"))); jobs = []
    tr = [r for r in rows if r["role"] == "htrain"]; ho = [r for r in rows if r["role"] == "hold"]
    for i, r in enumerate(tr):
        lr = random.Random(zlib.crc32(f"h3|{r['file']}".encode())); jobs.append(("im_set_h3", f"h3_{i:05d}", r, True, KT[lr.randrange(len(KT))]))
    for i, r in enumerate(ho):
        lr = random.Random(zlib.crc32(f"r3|{r['file']}".encode())); real = lr.random() < .5; jobs.append(("im_set_r3", f"r3_{i:05d}", r, real, None if real else KH[lr.randrange(len(KH))]))
    return jobs, rows

def writer(sn, mode):
    d = LOCAL / sn; (d / "img").mkdir(parents=True, exist_ok=True); (d / "mask").mkdir(exist_ok=True); mp = d / f"manifest_{mode}.csv"
    done = {r["file"] for r in csv.DictReader(open(mp, encoding="utf-8"))} if mp.exists() else set()
    f = open(mp, "a", newline="", encoding="utf-8"); w = csv.DictWriter(f, fieldnames=FN)
    if f.tell() == 0: w.writeheader()
    return d, done, f, w

def main(mode):
    if mode == "merge":
        for sn in ("im_set_h3", "im_set_r3"):
            d = LOCAL / sn; rr = []
            for md in ("cpu", "gpu"):
                if (d / f"manifest_{md}.csv").exists(): rr += list(csv.DictReader(open(d / f"manifest_{md}.csv", encoding="utf-8")))
            rr.sort(key=lambda r: r["file"]); assert len({r["file"] for r in rr}) == len(rr)
            with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=FN); w.writeheader(); w.writerows(rr)
            print(sn, len(rr), flush=True)
        return
    jobs, rows = plan(); W = {sn: writer(sn, mode) for sn in ("im_set_h3", "im_set_r3")}; t = time.time(); k = 0
    bycode = {}
    for r in rows: bycode.setdefault((r["role"], r["farm_code"]), []).append(r)   # 같은 role 안에서만 참조
    if mode == "gpu":
        import torch
        from im15_fakes_v5 import INP, NEG, object_mask  # noqa
        from im15c_pgen_v6 import pipe  # noqa
    for sn, sid, r, real_pair, kind in jobs:
        d, done, f, w = W[sn]; raw_p = RAWH / r["tar"] / r["file"]; base_row = dict(source=r["file"], group=r["group"], farm_code=r["farm_code"], tar=r["tar"])
        try:
            if mode == "cpu":
                raw = Image.open(raw_p).convert("RGB")
                if real_pair and f"{sid}_real.jpg" not in done:
                    lr = random.Random(zlib.crc32(f"{sid}|real".encode())); (d / "img" / f"{sid}_real.jpg").write_bytes(jpeg_bytes(dist_cover(raw, lr)))
                    w.writerow(dict(file=f"{sid}_real.jpg", label=0, kind="real", info="", other="", **base_row))
                if kind and kind != "partial_gen_v7" and f"{sid}_fake.jpg" not in done:
                    lr = random.Random(zlib.crc32(f"{sid}|fake".encode()))
                    if kind.startswith("filter"):
                        fr, info = raw_filter(raw, kind, lr); img = dist_cover(fr, random.Random(zlib.crc32(f"{sid}|crop".encode()))); m = None; oth = ""
                    else:
                        a = to_np(dist_cover(raw, lr)); pool = [x for x in bycode[(r["role"], r["farm_code"])] if x["file"] != r["file"]] or [r]; o = lr.choice(pool); oth = o["file"]
                        other = to_np(dist_cover(Image.open(RAWH / o["tar"] / o["file"]).convert("RGB"), lr))
                        if other.shape != a.shape: other = np.asarray(Image.fromarray(other).resize((a.shape[1], a.shape[0]), Image.BILINEAR))
                        fa, m, info = pstyle_mask(a, lr, other); img = to_img(fa)
                    (d / "img" / f"{sid}_fake.jpg").write_bytes(jpeg_bytes(img))
                    if m is not None: Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).save(d / "mask" / f"{sid}_fake.png")
                    w.writerow(dict(file=f"{sid}_fake.jpg", label=1, kind=kind, info=info, other=oth, **base_row))
            elif kind == "partial_gen_v7" and f"{sid}_fake.jpg" not in done:   # im15d.make와 같은 계산(원본 경로만 다름)
                lr = random.Random(zlib.crc32(f"{sid}|v7".encode())); g = torch.Generator("cpu").manual_seed(lr.randint(0, 2**31))
                a = to_np(dist_cover(Image.open(raw_p).convert("RGB"), lr)); h, ww = a.shape[:2]
                m = object_mask(a, lr, .10, .30) if lr.random() < .5 else blob_mask(a.shape, lr, (.10, .30))
                ys, xs = np.where(m > 0); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                side = int(min(max(max(y1 - y0, x1 - x0) * lr.uniform(1.3, 2.0), 384), min(h, ww)))
                cy, cx = (y0 + y1) // 2, (x0 + x1) // 2; wy = int(np.clip(cy - side // 2, 0, h - side)); wx = int(np.clip(cx - side // 2, 0, ww - side))
                which = "xl" if lr.random() < .5 else "sd15"; S = 1024 if which == "xl" else 512
                win = Image.fromarray(a[wy:wy + side, wx:wx + side]).resize((S, S), Image.LANCZOS)
                mw = Image.fromarray((m[wy:wy + side, wx:wx + side] * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
                prompt = f"{lr.choice(INP)}, photo, realistic"
                out = pipe(which)(prompt, image=win, mask_image=mw, width=S, height=S, strength=lr.uniform(.85, 1.0), num_inference_steps=lr.randint(25, 35),
                                  guidance_scale=lr.uniform(5, 8), negative_prompt=NEG, generator=g).images[0]
                patch = np.asarray(out.resize((side, side), Image.LANCZOS)).astype(np.float32)
                o = a.astype(np.float32).copy(); fm = feather(m, lr.randint(1, 4))[..., None][wy:wy + side, wx:wx + side]
                o[wy:wy + side, wx:wx + side] = patch * fm + o[wy:wy + side, wx:wx + side] * (1 - fm)
                (d / "img" / f"{sid}_fake.jpg").write_bytes(jpeg_bytes(to_img(o))); Image.fromarray((m * 255).astype(np.uint8)).save(d / "mask" / f"{sid}_fake.png")
                w.writerow(dict(file=f"{sid}_fake.jpg", label=1, kind=kind, info=f"{which}|{prompt}|area={m.mean():.3f}|win={side}", other="", **base_row))
            else: continue
        except Exception as e:
            print("FAIL", sid, repr(e)[:200], flush=True); continue
        f.flush(); k += 1
        if k % 100 == 0: print(f"{mode} {k} {time.time() - t:.0f}s", flush=True)
    print(f"H_{mode.upper()}_DONE {k}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
