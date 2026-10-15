"""IM03f: im03e 일반화 — 사용: python im03f_make_pair_set.py <manifest 파일명(im_real_raw 안)> <출력 세트 이름> <id 접두>
 후보 R2: manifest_r2.csv im_set_r2 r
원본 설명: IM03e: 후보 H1 자료 — 어려운 실제(manifest_hard.csv) + 원본마다 가짜 1장(v3/v4와 같은 제작기: 필터는 원본 해상도, 합성은 v4, 스타일은 v2/v3).
출력 local/im_set_h1 (manifest에 hard 열). '흐림=가짜' 지름길이 생기지 않도록 어려운 실제에도 짝 가짜를 만듦.
"""
import csv, random, zlib, sys, numpy as np, cv2
from pathlib import Path
from PIL import Image
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im03_make_fakes import jpeg_bytes, to_np, to_img, RAW, LOCAL  # noqa
import im03b_make_fakes_v2 as v2  # noqa
from im03c_make_fakes_v3 import dist_cover, raw_filter, KINDS  # noqa
from im03d_composite_v4 import compose  # noqa

OUT = LOCAL / sys.argv[2]; MAN = sys.argv[1]; PFX = sys.argv[3]

def main():
    hard = list(csv.DictReader(open(RAW / MAN, encoding="utf-8")))
    allm = list(csv.DictReader(open(RAW / "manifest.csv", encoding="utf-8")))
    red = set((LOCAL / "red_sources.txt").read_text(encoding="utf-8").split(chr(10))) - {""}
    by_code = defaultdict(list)
    for r in allm:
        if r["file"] not in red: by_code[r["farm_code"]].append(r["file"])
    (OUT / "img").mkdir(parents=True, exist_ok=True); rows = []
    for n, r in enumerate(hard):
        src = r["file"]; lr = random.Random(zlib.crc32(f"{PFX}|{src}".encode())); sid = f"{PFX}{zlib.crc32(src.encode()):010d}"
        raw = Image.open(RAW / src).convert("RGB"); st = lr.getstate(); base = dist_cover(raw, lr); a = to_np(base)
        (OUT / "img" / f"{sid}_real.jpg").write_bytes(jpeg_bytes(base))
        kind = KINDS[lr.randrange(len(KINDS))]
        if kind.startswith("filter"):
            fr, info = raw_filter(raw, kind, lr); ls = random.Random(); ls.setstate(st); fa = to_np(dist_cover(fr, ls))
        else:
            pool = [f for f in by_code[r["farm_code"]] if f != src] or [src]; oth = lr.choice(pool)
            other = to_np(dist_cover(Image.open(RAW / oth).convert("RGB"), lr))
            if other.shape != a.shape: other = cv2.resize(other, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
            if kind == "composite": fa, info = compose(a, lr, other)
            else:
                fa, info = v2.make_fake(a, kind, lr, other)
                if kind == "full_style": kind = "pstyle_color"
        (OUT / "img" / f"{sid}_fake.jpg").write_bytes(jpeg_bytes(to_img(fa)))
        for lab, fn, k, inf in [(0, f"{sid}_real.jpg", "real", ""), (1, f"{sid}_fake.jpg", kind, info)]:
            rows.append(dict(file=fn, label=lab, kind=k, info=inf, source=src, group=r["group"], farm_code=r["farm_code"], tar=r["tar"], other="", hard=r.get("hard", ""), farm_id=r.get("farm_id", ""), scale=r.get("scale", "")))
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("rows", len(rows))

if __name__ == "__main__":
    main()
