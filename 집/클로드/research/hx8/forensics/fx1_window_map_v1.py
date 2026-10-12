# -*- coding: utf-8 -*-
# hx8/forensics fx1: 순수 2차 창(달력 70~100일) 기록 지도 — 서술용.
# 평가 기록의 날짜 = 그 기록 자신의 24시 외기와 학습 기록 외기 완전 일치(학습 기록만 상대, 규정 안).
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3

OUT = os.path.dirname(os.path.abspath(__file__))
rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)])
cal = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
day = np.full(N, np.nan)
for i in range(N):
    if rid[i] in cal.index: day[i] = cal.loc[rid[i], "day"]
# eval dates via exact outdoor match to TRAIN records (>= 22 of 24 hours x 4 vars equal)
Xf = X.reshape(N, -1)
dsrc = np.array([""] * N, dtype=object)
for i in np.where(~TRAIN)[0]:
    eq = (np.abs(Xf[TRAIN] - Xf[i]) < 1e-6).mean(1)
    tr = np.where(TRAIN)[0]
    m = tr[eq >= 22 / 24]
    ds = [day[j] for j in m if np.isfinite(day[j])]
    if ds:
        day[i] = np.median(ds); dsrc[i] = ",".join(rid[m])
sub = pd.read_csv(r"C:\work\farmai\제출\13회차_2026-10-12(나)\submission_22.csv")
sub["rec"] = sub.row_id.str[:7]
pm = sub.groupby("rec").sub_ec.mean()
p = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv")
p = p[(p.validator == "DIAG10") & (p.tag == "CUR")]
p["rec"] = p.row_id.str[:7]
oof = p.groupby("rec").sg.mean()
rows = []
for i in range(N):
    rows.append(dict(rec=rid[i], split="train" if TRAIN[i] else "EVAL", day=day[i],
                     ec_mean=np.nanmean(EC[i]) if TRAIN[i] else np.nan,
                     ec0=EC[i, 0] if TRAIN[i] else np.nan, ec23=EC[i, 23] if TRAIN[i] else np.nan,
                     pred13=pm.get(rid[i], np.nan), oof=oof.get(rid[i], np.nan),
                     fan=np.nanmean(AC[i, :, 4]) if AC.shape[2] > 4 else np.nan,
                     co2=np.nanmean(IN_[i, :, 2]), match=dsrc[i]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "fx1_records_v1.csv"), index=False)
w = df[(df.day >= 70) & (df.day <= 100)].sort_values(["day", "rec"])
pd.set_option("display.width", 250)
print(w[["rec", "split", "day", "ec_mean", "ec0", "ec23", "pred13", "oof", "fan", "co2"]].round(3).to_string(index=False))
print("AC vars order check:", AC.shape)
