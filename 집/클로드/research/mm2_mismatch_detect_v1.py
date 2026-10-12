# -*- coding: utf-8 -*-
"""MM2 (2026-10-12 집 클로드; 자율 — 비평 6.750: 차단 해제 앵커는 MM1형(입력·정답 어긋남) 기록에서만 큰 이득 → MM1형을 정답 없이 입력만으로 알아낼 수 있나).  서술(신호 확인).
가설 H-SWAP: MM1 기록의 실내 입력은 같은 날짜 다른 동의 실내 입력과 거의 같다(입력이 다른 동 것과 섞임). F47_191 의 실내온도가 F47_190 과 가까웠던 관찰(6.702)에서 출발.
지표(입력만, 정답 안 씀): 같은 날짜(DM11) 다른 기록과의 실내 3변수(in_temp, in_hum, in_co2) 0~5시 / 하루 전체 평균 절대차(표준화)의 최솟값 dmin, 그리고
  같은 날짜 기록들끼리의 일반적 거리 대비 비율.  자정 실내 점프 |dT|(전날 진짜 짝 기준, 진단용)도 함께.
비교 집단: MM1 확인 기록(F47_191, F47_178, F47_189, F13_107) + 비평의 차단 해제 이득 기록(F47_241) vs 차단 해제 손해 기록(F13_149, F47_057, F47_016, F13_019) vs 나머지 학습 기록.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
X = X[X.row_id.str[:3].isin(["F13", "F47"])]; X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
V = ["in_temp", "in_hum", "in_co2"]
recs = sorted(X.rec.unique()); A = np.stack([X.pivot(index="rec", columns="h", values=v).reindex(index=recs, columns=range(24)).values for v in V], 2)
A = (A - np.nanmean(A, (0, 1))) / (np.nanstd(A, (0, 1)) + 1e-9); ri = {r: i for i, r in enumerate(recs)}
C = pd.read_csv(os.path.join(H, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(C.rec, C.day)); SPL = dict(zip(C.rec, C.split))
byday = C.dropna(subset=["day"]).groupby("day").rec.apply(list).to_dict()
def dist(a, b, hrs): return float(np.nanmean(np.abs(A[ri[a], hrs] - A[ri[b], hrs])))
rows = []
for r in recs:
    if SPL.get(r) != "train" or not np.isfinite(DAY.get(r, np.nan)): continue
    sib = [x for x in byday.get(DAY[r], []) if x != r and x in ri]
    if not sib: continue
    d05 = [dist(r, x, list(range(6))) for x in sib]; dall = [dist(r, x, list(range(24))) for x in sib]
    k = int(np.argmin(dall)); others = [dist(x, y, list(range(24))) for i, x in enumerate(sib) for y in sib[i + 1:]]
    rows.append(dict(rec=r, n_sib=len(sib), dmin05=min(d05), dmin=min(dall), nearest=sib[k], typical=np.mean(others) if others else np.nan))
T = pd.DataFrame(rows); T["ratio"] = T.dmin / T.typical
GAIN = {"F47_191", "F47_178", "F47_189", "F13_107", "F47_241"}; LOSS = {"F13_149", "F47_057", "F47_016", "F13_019"}
T["grp"] = np.where(T.rec.isin(GAIN), "MM1 / gate-release gain", np.where(T.rec.isin(LOSS), "gate-release loss", "other"))
pd.set_option("display.width", 200)
print(T.groupby("grp")[["dmin05", "dmin", "ratio"]].describe().loc[:, (slice(None), ["count", "mean", "50%"])].round(3).to_string())
print(T[T.grp != "other"].sort_values("grp")[["rec", "grp", "n_sib", "dmin05", "dmin", "nearest", "typical", "ratio"]].round(3).to_string(index=False))
print("\nrank of MM1/gain records by dmin among all training records (low = indoor nearly identical to a same-date record):")
T["rank"] = T.dmin.rank(); print(T[T.grp != "other"][["rec", "grp", "rank"]].to_string(index=False), f"| of {len(T)}")
