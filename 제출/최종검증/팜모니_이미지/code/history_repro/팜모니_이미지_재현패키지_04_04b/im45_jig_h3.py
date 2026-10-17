"""IM45: 후보 H2용 — htrain 원본 1800장에 처리 흔들기(im36b jiggle: 덮게 축소 커널 4종 → 가운데 크롭 → JPEG q70~95, 저장 q95 = jig500과 같은 방식) → im_set_h3j(진짜, kind real_jig).
서버(nadan). 사용: python im45_jig_h3.py"""
import sys, csv, random, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from im36b_finefp_eval_v2 import jiggle  # noqa
from im36_finefp_eval import L  # noqa
RAWH = L / "im_h_raw"; OUT = L / "im_set_h3j"; (OUT / "img").mkdir(parents=True, exist_ok=True)
rows = [r for r in csv.DictReader(open(RAWH / "manifest.csv", encoding="utf-8")) if r["role"] == "htrain"]
with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["file", "label", "kind", "info", "source", "group", "farm_code", "tar", "other"]); w.writeheader()
    for i, r in enumerate(rows):
        fn = f"h3j_{i:05d}.jpg"; im = jiggle(RAWH / r["tar"] / r["file"], random.Random(zlib.crc32(f"h3j|{r['file']}".encode()))); im.save(OUT / "img" / fn, quality=95)
        w.writerow(dict(file=fn, label=0, kind="real_jig", info="", source=r["file"], group=r["group"], farm_code=r["farm_code"], tar=r["tar"], other=""))
print("H3J_DONE", len(rows))
