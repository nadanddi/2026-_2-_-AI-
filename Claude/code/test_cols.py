# -*- coding: utf-8 -*-
"""What does test_X actually contain, column by column, per greenhouse?"""
import pandas as pd
from common import load_raw

tX, ty, sX = load_raw()
cols = [c for c in sX.columns if c not in ("row_id", "farm", "day", "hour", "t")]
GROUP = {"out_temp": "외부기상", "out_hum": "외부기상", "out_rad": "외부기상", "out_wspd": "외부기상",
         "in_temp": "내부환경", "in_hum": "내부환경", "in_co2": "내부환경", "in_rad": "내부환경"}

print("%-12s %-8s %10s %10s %12s %12s" % ("컬럼", "분류", "F13 결측%", "F47 결측%", "F13 고유값", "F47 고유값"))
for c in cols:
    g = GROUP.get(c, "구동기")
    a, b = sX[sX.farm == "F13"], sX[sX.farm == "F47"]
    print("%-12s %-8s %9.1f%% %9.1f%% %12s %12s"
          % (c, g, 100 * a[c].isna().mean(), 100 * b[c].isna().mean(),
             a[c].nunique() or "-", b[c].nunique() or "-"))

usable = [c for c in cols if sX[c].notna().any()]
print("\n사용 가능 %d개 / 전부 결측 %d개" % (len(usable), len(cols) - len(usable)))
print("전부 결측:", ", ".join(c for c in cols if c not in usable))

print("\n=== 학습(F13·F47)과 평가의 결측률 비교 ===")
t = tX[tX.farm.isin(["F13", "F47"])]
print("%-12s %12s %12s" % ("컬럼", "학습 결측%", "평가 결측%"))
for c in usable:
    print("%-12s %11.2f%% %11.2f%%" % (c, 100 * t[c].isna().mean(), 100 * sX[c].isna().mean()))
