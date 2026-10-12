# -*- coding: utf-8 -*-
# hx8/forensics fx1 v2: 순수 2차 창(달력 70~100일) 기록 지도 — 서술용.
# 날짜 = DM11 달력(평가 입력 사용 → 서술 대조 전용, 특징·선택에 쓰지 않음).
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3

OUT = os.path.dirname(os.path.abspath(__file__))
rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)])
cal = pd.read_csv(os.path.join(env.ROOT, u"집", u"클로드", "research", "ec0h", "rise", "dm11_calendar_v1.csv"))
cal = cal.dropna(subset=["day"]).drop_duplicates("rec").set_index("rec")
day = np.array([cal.day.get(r, np.nan) for r in rid])
sub = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv"))
sub["rec"] = sub.row_id.str[:7]
pm = sub.groupby("rec").sub_ec.mean()
p = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
p = p[(p.validator == "DIAG10") & (p.tag == "CUR")]
p["rec"] = p.row_id.str[:7]
oof = p.groupby("rec").sg.mean()
rows = []
for i in range(N):
    rows.append(dict(rec=rid[i], split="train" if TRAIN[i] else "EVAL", day=day[i],
                     ec_mean=np.nanmean(EC[i]) if TRAIN[i] else np.nan,
                     ec0=EC[i, 0] if TRAIN[i] else np.nan, ec23=EC[i, 23] if TRAIN[i] else np.nan,
                     pred13=pm.get(rid[i], np.nan), oof=oof.get(rid[i], np.nan),
                     ac=" ".join(f"{np.nanmean(AC[i, :, k]):.0f}" for k in range(AC.shape[2])),
                     tin=np.nanmean(IN_[i, :, 0]), hum=np.nanmean(IN_[i, :, 1]), co2=np.nanmean(IN_[i, :, 2])))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "fx1_records_v2.csv"), index=False)
w = df[(df.day >= 70) & (df.day <= 100)].sort_values(["day", "rec"])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(w.round(3).to_string(index=False))
