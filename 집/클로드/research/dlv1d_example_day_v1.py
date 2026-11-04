# -*- coding: utf-8 -*-
"""DLV1d (2026-11-04 집 클로드; 발표 1.2 (1/3) 찾은 점 ①의 '실제 예시 하루' 그림 자료). 서술용, 판정 없음.
DLV1(dlv1_day_level_offsets_all_farms_v1.py)과 같은 기준 모델(선형 회귀 + LightGBM 잔차, 날짜 연속 5덩어리 OOF)로
F13·F47의 시각별 예측을 다시 만들고, 예시 하루를 아래 규칙(실행 전 고정)으로 고른다.
  규칙: 하루 제곱 오차 합이 그 온실 상위 10%인 날 중에서, 그날 안의 '하루 밀림 몫'(24·m^2 / sum e^2)이 가장 큰 날 하나.
        F13·F47 둘 다 계산하고 하루 밀림 몫이 더 큰 쪽을 쓴다.
출력: 집/클로드/발표/최종/data/example_day_v1.json (시각별 정답·예측·실내 온도·외기 온도), 표의 F13·F47 수치 재현 확인.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dlv1d_example_day_v1.py
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
XO = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "out_temp"])

best = None
for T in ("F13", "F47"):
    R = D[D.farm == T].copy(); days = np.array(sorted(R.day.unique()))
    blk = {d: min(i * 5 // len(days), 4) for i, d in enumerate(days)}; R["fold"] = R.day.map(blk); R["pred"] = np.nan
    for k in range(5):
        tr = R[(R.fold != k) & R.sub_temp.notna()]; te = R[R.fold == k]; _, p = fit_lr(tr, np.ones(len(tr)), te); R.loc[te.index, "pred"] = p
    R = R[R.sub_temp.notna() & R.pred.notna()]; R["e"] = R.pred - R.sub_temp
    g = R.groupby("day"); m = g.e.mean(); n = g.size(); sse = (R.e ** 2).sum()
    lvl = float((n * m ** 2).sum() / sse); dse = g.e.apply(lambda s: (s ** 2).sum()).sort_values(ascending=False)
    top = float(dse.head(max(1, len(dse) // 10)).sum() / sse)
    print(f"{T}: 하루 밀림 몫 {lvl:.3f}, 상위 10% 날 몫 {top:.3f} (DLV1 표: F13 .486/.588, F47 .529/.487)")
    cand = dse.head(max(1, len(dse) // 10)).index
    within = (n[cand] * m[cand] ** 2) / dse[cand]
    within = within[n[cand] == 24]
    d = within.idxmax(); sc = float(within.max())
    print(f"   예시 후보 {T}_{d:03d}: 그날 안 하루 밀림 몫 {sc:.3f}, 하루 평균 오차 {m[d]:+.2f}℃")
    if best is None or sc > best[2]:
        best = (T, d, sc, R[R.day == d].sort_values("hour"))

T, d, sc, E = best
E = E.merge(XO, on="row_id", how="left")
out = dict(record=f"{T}_{d:03d}", within_day_level_share=round(sc, 3), mean_error=round(float(E.e.mean()), 3),
           hour=E.hour.tolist(), truth=E.sub_temp.round(2).tolist(), pred=E.pred.round(2).tolist(),
           in_temp=E.in_temp.round(2).tolist(), out_temp=E.out_temp.round(2).tolist())
os.makedirs(os.path.join(HERE, "..", "발표", "최종", "data"), exist_ok=True)
fn = os.path.join(HERE, "..", "발표", "최종", "data", "example_day_v1.json")
json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("예시:", out["record"], "그날 안 하루 밀림 몫", out["within_day_level_share"], "평균 오차(예측−정답)", out["mean_error"])
print("wrote", os.path.abspath(fn))
