# -*- coding: utf-8 -*-
"""PHY1 (2026-10-16 집 클로드; 사용자 '2번 진행' — 숨은 배지 상태를 공기의 열 반응으로 읽을 수 있나).  서술(비인과 상한), 실행 전 커밋.  온도.
가설: 배지가 따뜻한 날은 같은 바깥 조건·운영에서 실내 공기가 덜 식는다(배지가 열 저장고) -> '공기 열수지 모형의 잔차'가 하루 수준 오차를 설명.
공기 모형(학습 입력만, 농장별 선형): d in_temp(t) = in_temp(t) - in_temp(t-1) ~ (out - in)(t-1), 난방, 환기 x (out - in), 보온커튼, 차광, 외기 일사, sin/cos 시각.
기록 특징: 공기 잔차 평균(0~6시 / 전체), 0~6시 냉각률(실내 하락 / 평균 실내-외기 차), 0~6시 실내-외기 차, 0~6시 난방 평균, 0시 실내 - 그날 실내 평균.
비교용 기존 표현: 그날 실내·외기 평균(입력 요약; 6.997 에서 R² 음수).
목표: 기록 하루 수준 오차 = 평균(기본 셀 예측 - 정답) (dgx5_rows: CV8·N·CK2·EL1 의 cells, 배치 평균).
상한(비인과, 하루 전체 입력 사용): Ridge(표준화, alpha 10), 5일 블록 단위 5겹 교차 R² (농장별·합침) x 특징 묶음 {요약, 물리, 요약+물리}.
다음 단계 기준(고정): 물리 또는 요약+물리 합침 교차 R² >= .10 이고 두 농장 모두 > 0 이면 인과판(그 시각까지)으로 진행, 아니면 2번 방향 닫음.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u phy1_air_response_state_bound_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
HERE = os.path.dirname(os.path.abspath(__file__))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["rec"] = X.row_id.str[:7]; X["farm"] = X.row_id.str[:3]; X["h"] = X.row_id.str[8:10].astype(int); X = X.sort_values(["rec", "h"]).reset_index(drop=True)
g = X.groupby("rec")
X["din"] = g.in_temp.diff(); X["gap1"] = (X.out_temp - X.in_temp).groupby(X.rec).shift(1); X["vgap"] = X.act_vent.fillna(0) * X.gap1
X["hs"], X["hc"] = np.sin(2 * np.pi * X.h / 24), np.cos(2 * np.pi * X.h / 24)
AIR = ["gap1", "act_heating", "vgap", "act_thermal", "act_shade", "out_rad", "hs", "hc"]
X["res"] = np.nan
for f in ("F13", "F47"):
    m = (X.farm == f) & X[AIR + ["din"]].notna().all(1); lr = LinearRegression().fit(X.loc[m, AIR], X.loc[m, "din"])
    X.loc[m, "res"] = X.loc[m, "din"] - lr.predict(X.loc[m, AIR]); print(f, "air model R2 %.3f" % lr.score(X.loc[m, AIR], X.loc[m, "din"]), "coef", dict(zip(AIR, np.round(lr.coef_, 3))))
n = X[X.h <= 6]
F = pd.DataFrame(index=sorted(X.rec.unique()))
F["res_night"] = n.groupby("rec").res.mean(); F["res_all"] = X.groupby("rec").res.mean()
dd = n.groupby("rec").apply(lambda z: (z.in_temp.iloc[0] - z.in_temp.iloc[-1]) / max(abs((z.in_temp - z.out_temp).mean()), .5))
F["cool_rate"] = dd; F["gap_night"] = (n.in_temp - n.out_temp).groupby(n.rec).mean(); F["heat_night"] = n.groupby("rec").act_heating.mean()
F["in0_minus_mean"] = X[X.h == 0].set_index("rec").in_temp - X.groupby("rec").in_temp.mean()
F["in_mean"] = X.groupby("rec").in_temp.mean(); F["out_mean"] = X.groupby("rec").out_temp.mean()
PHYS = ["res_night", "res_all", "cool_rate", "gap_night", "heat_night", "in0_minus_mean"]; SUM = ["in_mean", "out_mean"]
R = pd.read_csv(os.path.join(HERE, "dgx5_rows_v1.csv.gz")); R["e"] = R.cells - R.y
lv = R.groupby(["layout", "rec"]).e.mean().groupby("rec").mean().rename("lvl")
D = F.join(lv, how="inner").dropna(); D["farm"] = D.index.str[:3]; D["blk"] = D.farm + "_" + (D.index.str[4:7].astype(int) // 5).astype(str)
print("records", len(D), D.farm.value_counts().to_dict(), "| level offset sd %.3f" % D.lvl.std())
print("\nSpearman with day-level offset (pooled / F13 / F47):")
for c in PHYS + SUM: print(f"  {c:15s} {D[c].corr(D.lvl, method='spearman'):+.3f} / {D[D.farm == 'F13'][c].corr(D[D.farm == 'F13'].lvl, method='spearman'):+.3f} / {D[D.farm == 'F47'][c].corr(D[D.farm == 'F47'].lvl, method='spearman'):+.3f}")
def cvr2(cols, sub):
    Z = D if sub == "pooled" else D[D.farm == sub]; blks = sorted(Z.blk.unique()); fold = {b: i % 5 for i, b in enumerate(blks)}; k = Z.blk.map(fold).values
    Xm = Z[cols].to_numpy(float); y = Z.lvl.values
    if sub == "pooled": Xm = np.column_stack([Xm, (Z.farm == "F47").astype(float)])
    pr = np.zeros(len(Z))
    for i in range(5):
        tr = k != i; m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xm[tr], y[tr]); pr[~tr] = m.predict(Xm[~tr])
    return 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("\ncross-fitted R2 (5-day blocks, 5 folds):")
OUT = {}
for nm, cols in (("summary", SUM), ("physics", PHYS), ("summary+physics", SUM + PHYS)):
    OUT[nm] = {s: cvr2(cols, s) for s in ("pooled", "F13", "F47")}; print(f"  {nm:16s} " + " | ".join(f"{s} {v:+.3f}" for s, v in OUT[nm].items()))
ok = any(OUT[nm]["pooled"] >= .10 and OUT[nm]["F13"] > 0 and OUT[nm]["F47"] > 0 for nm in ("physics", "summary+physics"))
print("\nPHY1 gate (pooled R2 >= .10 and both farms > 0):", "PROCEED to causal version" if ok else "CLOSE direction 2")
