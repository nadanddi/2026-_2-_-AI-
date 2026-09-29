# -*- coding: utf-8 -*-
"""Where do 49 / 48 / 32 come from?  Count greenhouses by what they can support."""
import numpy as np
import pandas as pd
from common import load_raw

tX, ty, sX = load_raw()
d = tX.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
lab = d[d.sub_temp.notna()]

others = sorted(f for f in d.farm.unique() if f not in ("F13", "F47"))
no32 = [f for f in others if f != "F32"]
print("전체 온실           : %d곳" % d.farm.nunique())
print("F13·F47 제외        : %d곳" % len(others))
print("F32까지 제외        : %d곳" % len(no32))

rows = []
for f in no32:
    g = lab[lab.farm == f]
    rows.append(dict(farm=f, n=len(g),
                     cold6=int((g.in_temp <= 6).sum()),
                     cold7=int((g.in_temp <= 7).sum()),
                     warm8=int((g.in_temp > 8).sum())))
t = pd.DataFrame(rows)

print("\n=== 한랭 외삽 실험(따뜻한 구간만 학습 → 추운 구간 예측)에 쓸 수 있는 온실 ===")
print("  조건: 예측할 추운 행이 충분하고, 학습할 따뜻한 행도 충분해야 함\n")
print("  %-28s %8s" % ("기준", "해당 온실"))
for name, m in [
        ("≤6℃ 라벨이 1행 이상", t.cold6 >= 1),
        ("≤6℃ 라벨이 10행 이상", t.cold6 >= 10),
        ("≤6℃ 라벨이 20행 이상", t.cold6 >= 20),
        ("≤6℃ 20행 이상 + >8℃ 1000행 이상", (t.cold6 >= 20) & (t.warm8 >= 1000)),
        ("≤6℃ 30행 이상 + >8℃ 1000행 이상", (t.cold6 >= 30) & (t.warm8 >= 1000)),
        ("≤7℃ 50행 이상 + >8℃ 1000행 이상", (t.cold7 >= 50) & (t.warm8 >= 1000)),
]:
    print("  %-28s %8d곳" % (name, int(m.sum())))

print("\n=== ≤6℃ 라벨 행 수 분포 (48곳) ===")
q = t.cold6.describe(percentiles=[.25, .5, .75]).round(1)
print("  0행인 온실 %d곳 / 중앙값 %.0f행 / 최대 %d행"
      % (int((t.cold6 == 0).sum()), t.cold6.median(), t.cold6.max()))
print("  분포:", {k: int(v) for k, v in q.items() if k in ("25%", "50%", "75%", "max")})
print("\n  상위 5곳:", t.nlargest(5, "cold6")[["farm", "cold6"]].to_dict("records"))
print("  0행 온실:", t[t.cold6 == 0].farm.tolist())
