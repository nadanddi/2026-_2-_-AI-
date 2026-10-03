# -*- coding: utf-8 -*-
"""X0: tidy hourly tables from public smart-farm open datasets (user-provided ZIPs,
스마트팜 개방 데이터셋; kept outside the repo).  2022_9 이레아이에스 (strawberry only),
2022_1 골든플래닛 (strawberry).  Output: scratch CSVs, wide per farm x hour."""
import os, sys, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import env  # noqa: F401
import pandas as pd
DL = r"C:\Users\aozks\Downloads"
OUT = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\2ea435a4-3dc2-434b-ad01-aff27f988a60\scratchpad"
os.makedirs(OUT, exist_ok=True)
for zf, tag, crop in (("2022_9_ds (1).zip", "ire", "딸기"), ("2022_1_ds (1).zip", "golden", "딸기")):
    z = zipfile.ZipFile(os.path.join(DL, zf))
    x = pd.read_csv(z.open(z.infolist()[0]), encoding="cp949", usecols=["수집일자", "농가id", "품목명", "장비명", "측정값"])
    x = x[x["품목명"] == crop]
    x["t"] = pd.to_datetime(x["수집일자"]); x["v"] = pd.to_numeric(x["측정값"], errors="coerce")
    W = x.pivot_table(index=["농가id", "t"], columns="장비명", values="v", aggfunc="mean").reset_index()
    W = W.rename(columns={"농가id": "farm"})
    W.to_csv(os.path.join(OUT, "x0_%s_hourly.csv" % tag), index=False, encoding="utf-8")
    print(tag, W.shape, W.farm.nunique(), W.t.min(), W.t.max()); print(W.columns.tolist())
