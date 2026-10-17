"""IM57: R4(im_i_raw hold2) 묶음이 다른 자료(θ 진짜 풀·학습 세트)와 겹치는지 기록(비평 지적). 서버."""
import csv, glob
from pathlib import Path
L = Path.home() / "storage/nadan/farmai_img/data/local"
r4g = {r["group"] for r in csv.DictReader(open(L / "im_i_raw/manifest.csv", encoding="utf-8")) if r["role"] == "hold2"}
print("R4 묶음", sorted(r4g))
for m in sorted(glob.glob(str(L / "im_set_*/manifest.csv"))):
    gs = {r.get("group", "") for r in csv.DictReader(open(m, encoding="utf-8"))}; ov = r4g & gs
    if ov: print(f"  겹침 {Path(m).parent.name}: {len(ov)}묶음 {sorted(ov)}")
print("OVERLAP_DONE")
