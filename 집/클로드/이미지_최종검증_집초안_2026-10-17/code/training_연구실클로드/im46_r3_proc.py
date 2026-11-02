"""IM46: H2 중간 비평 반영 — R3 진짜 사진(429)에 학습에 없던 처리 5종 → im_set_r3p_<처리>(진짜). 서버(nadan).
 sharp: R3 사진 선명화(UnsharpMask r2, 100%) 후 Q92 / png: R3 사진을 PNG로 저장 / area: 원본 → cv2 INTER_AREA 덮게 축소·가운데 크롭 → Q92
 webp: R3 사진을 WebP q90 / c444: 원본 → 배포 처리(dist_cover) → JPEG Q92 크로마 4:4:4"""
import sys, csv, random, zlib, io, numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageFilter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03c_make_fakes_v3 import dist_cover  # noqa
from im36_finefp_eval import L  # noqa
R3 = L / "im_set_r3"; RAWH = L / "im_h_raw"
rows = [r for r in csv.DictReader(open(R3 / "manifest.csv", encoding="utf-8")) if r["kind"] == "real"]
for kind, ext in (("sharp", "jpg"), ("png", "png"), ("area", "jpg"), ("webp", "webp"), ("c444", "jpg")):
    d = L / f"im_set_r3p_{kind}"; (d / "img").mkdir(parents=True, exist_ok=True); out = []
    for r in rows:
        im = Image.open(R3 / "img" / r["file"]).convert("RGB"); fn = r["file"].replace(".jpg", f".{ext}"); p = d / "img" / fn
        if kind == "sharp": im.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=2)).save(p, quality=92)
        elif kind == "png": im.save(p)
        elif kind == "webp": im.save(p, quality=90)
        else:
            raw = Image.open(RAWH / r["tar"] / r["source"]).convert("RGB")
            if kind == "area":
                w, h = raw.size; tw, th = (768, 1024) if h >= w else (1024, 768); s = max(tw / w, th / h); nw, nh = max(tw, round(w * s)), max(th, round(h * s))
                a = cv2.resize(np.asarray(raw), (nw, nh), interpolation=cv2.INTER_AREA); x0, y0 = (nw - tw) // 2, (nh - th) // 2; Image.fromarray(a[y0:y0 + th, x0:x0 + tw]).save(p, quality=92)
            else: dist_cover(raw, random.Random(zlib.crc32(f"{r['file']}|444".encode()))).save(p, quality=92, subsampling=0)
        out.append(dict(r, file=fn))
    with open(d / "manifest.csv", "w", newline="", encoding="utf-8") as f: w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(kind, len(out), flush=True)
print("R3P_DONE")
