# -*- coding: utf-8 -*-
"""DLV1e (2026-11-04 집 클로드; 발표 1.2 찾은 점 ③ '추울 때 생기는 오차' 그림 자료). 서술용, 판정 없음.
DLV1과 같은 기준 모델(선형 회귀 + LightGBM 잔차, 날짜 연속 5덩어리 OOF)로 F13·F47 학습 기록의 시각별 예측을 만들고
아래를 잰다. 구간과 기준은 실행 전에 고정:
  실내 온도 구간: <8, 8~10, 10~12, 12~15, 15~20, >=20 (℃)
  (a) 구간별 오차 크기 RMSE(F13·F47 합침)
  (b) 구간별 '정답 − 예측' 평균을, 고EC 날(그 기록의 하루 평균 실제 EC >= 1.0)과 보통 날로 나눠서
출력: 집/클로드/발표/최종/data/cold_bins_v1.json
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dlv1e_cold_bins_v1.py
"""
import os, json
import env  # noqa: F401
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "dlv1_day_level_offsets_all_farms_v1.py"), encoding="utf-8").read()
src = src[: src.index("from sklearn.linear_model import LinearRegression")]
ns = {"__file__": os.path.join(HERE, "dlv1_day_level_offsets_all_farms_v1.py")}
exec(compile(src, "dlv1_head", "exec"), ns)
D, fit_lr = ns["D"], ns["fit_lr"]
YE = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_ec"])

BINS = [-99, 8, 10, 12, 15, 20, 99]
LAB = ["8℃ 미만", "8~10", "10~12", "12~15", "15~20", "20℃ 이상"]
parts = []
for T in ("F13", "F47"):
    R = D[D.farm == T].copy(); days = np.array(sorted(R.day.unique()))
    blk = {d: min(i * 5 // len(days), 4) for i, d in enumerate(days)}; R["fold"] = R.day.map(blk); R["pred"] = np.nan
    for k in range(5):
        tr = R[(R.fold != k) & R.sub_temp.notna()]; te = R[R.fold == k]; _, p = fit_lr(tr, np.ones(len(tr)), te); R.loc[te.index, "pred"] = p
    parts.append(R[R.sub_temp.notna() & R.pred.notna()])
A = pd.concat(parts).merge(YE, on="row_id", how="left")
A["r"] = A.sub_temp - A.pred                      # 정답 − 예측
A["ecday"] = A.groupby(["farm", "day"]).sub_ec.transform("mean")
A["hiec"] = A.ecday >= 1.0
A = A[A.in_temp.notna()]
A["bin"] = pd.cut(A.in_temp, BINS, right=False, labels=LAB)
g = A.groupby("bin", observed=False)
rmse = g.r.apply(lambda s: float(np.sqrt(np.mean(s ** 2))) if len(s) else np.nan)
n = g.size()
mh = A[A.hiec].groupby("bin", observed=False).r.agg(["mean", "size"])
ml = A[~A.hiec].groupby("bin", observed=False).r.agg(["mean", "size"])
print("전체 RMSE", round(float(np.sqrt(np.mean(A.r ** 2))), 3), "행", len(A), "고EC 날 행", int(A.hiec.sum()))
for b in LAB:
    print(f"{b:>8}: n {int(n[b]):5d}  RMSE {rmse[b]:.3f} | 고EC 정답−예측 {mh.loc[b,'mean']:+.3f} (n {int(mh.loc[b,'size'])}) | 보통 {ml.loc[b,'mean']:+.3f} (n {int(ml.loc[b,'size'])})")
out = dict(bins=LAB, n=[int(n[b]) for b in LAB], rmse=[round(float(rmse[b]), 3) for b in LAB],
           hiec_mean=[None if np.isnan(mh.loc[b, "mean"]) else round(float(mh.loc[b, "mean"]), 3) for b in LAB],
           hiec_n=[int(mh.loc[b, "size"]) for b in LAB],
           low_mean=[None if np.isnan(ml.loc[b, "mean"]) else round(float(ml.loc[b, "mean"]), 3) for b in LAB],
           low_n=[int(ml.loc[b, "size"]) for b in LAB])
fn = os.path.join(HERE, "..", "발표", "최종", "data", "cold_bins_v1.json")
json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", os.path.abspath(fn))
