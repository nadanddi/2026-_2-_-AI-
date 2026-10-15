# -*- coding: utf-8 -*-
"""DTY1 (2026-10-16 집 클로드; 사용자 '유형별로 비슷한 부분 + 모델이 자동으로 빌려오는 형태(5점)').  서술(판정 없음), 실행 전 커밋.  온도.
1) 날 유형 자동 나누기: 모든 온실 날(24시간 다 있는 날)의 공통 입력 요약 12개(실내온도 0~5시 평균·최저·최고·7~11시 상승·12~16시 평균·18~23시 평균,
   습도 밤/낮 평균, CO2 밤/낮 평균, 실내온도 하루 표준편차, 0시 실내온도)를 표준화 -> GaussianMixture(대각, 시드 0), 유형 수 k = 2..10 중 BIC 최소(사람이 안 고름).
2) 유형별 닮음: 유형 k 에서 온실마다 시각별 (배지 − 실내) 중앙값 곡선(그 유형 날 >= 5일), F13·F47 곡선과의 RMS 거리 -> 유형별 가장 닮은 온실 3곳·거리,
   기준선: 같은 유형에서 F13 vs F47 거리.  F32(강한 가온) 포함(자동이므로 빼지 않음).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dty1_auto_day_types_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
HERE = os.path.dirname(os.path.abspath(__file__))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp", "in_hum", "in_co2"]); Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
D = X.merge(Y, on="row_id"); D["farm"] = D.row_id.str[:3]; D["day"] = D.row_id.str[4:7].astype(int); D["h"] = D.row_id.str[8:10].astype(int)
D.loc[~D.in_temp.between(-5, 50), "in_temp"] = np.nan; D.loc[D.in_hum <= 0, "in_hum"] = np.nan; D.loc[D.in_co2 <= 0, "in_co2"] = np.nan; D.loc[~D.sub_temp.between(-5, 45), "sub_temp"] = np.nan
P = D.pivot_table(index=["farm", "day"], columns="h", values=["in_temp", "in_hum", "in_co2", "sub_temp"])
ok = P["in_temp"].notna().sum(1).eq(24) & P["sub_temp"].notna().sum(1).ge(20); P = P[ok]
T_, H_, C_ = P["in_temp"], P["in_hum"], P["in_co2"]
F = pd.DataFrame({"t_night": T_.loc[:, 0:5].mean(1), "t_min": T_.min(1), "t_max": T_.max(1), "t_rise": T_.loc[:, 11] - T_.loc[:, 7], "t_aft": T_.loc[:, 12:16].mean(1),
                  "t_eve": T_.loc[:, 18:23].mean(1), "h_night": H_.loc[:, 0:5].mean(1), "h_day": H_.loc[:, 10:16].mean(1), "c_night": C_.loc[:, 0:5].mean(1),
                  "c_day": C_.loc[:, 10:16].mean(1), "t_sd": T_.std(1), "t0": T_.loc[:, 0]})
F = F.fillna(F.median()); Z = (F - F.mean()) / F.std()
bic = {k: GaussianMixture(k, covariance_type="diag", random_state=0, n_init=3).fit(Z).bic(Z) for k in range(2, 11)}
K = min(bic, key=bic.get); gm = GaussianMixture(K, covariance_type="diag", random_state=0, n_init=3).fit(Z); typ = pd.Series(gm.predict(Z), index=Z.index)
print("BIC by k:", {k: round(v) for k, v in bic.items()}, "-> k", K)
pd.set_option("display.width", 220)
desc = F.groupby(typ.values).median()[["t_min", "t_max", "t_rise", "t_night", "t_eve", "h_night", "c_day"]].round(1); desc["days_all"] = typ.value_counts().sort_index()
desc["F13"] = typ[typ.index.get_level_values(0) == "F13"].value_counts().reindex(desc.index).fillna(0).astype(int); desc["F47"] = typ[typ.index.get_level_values(0) == "F47"].value_counts().reindex(desc.index).fillna(0).astype(int)
print(desc.to_string())
GAP = (P["sub_temp"] - P["in_temp"])
rows = []
for k in range(K):
    idx = typ.index[typ.values == k]; G = GAP.loc[idx]; farm = G.index.get_level_values(0)
    cnt = pd.Series(farm).value_counts(); prof = G.groupby(farm).median()[cnt.reindex(G.groupby(farm).median().index) >= 5]
    for tgt in ("F13", "F47"):
        if tgt not in prof.index: rows.append(dict(type=k, target=tgt, note="target <5 days")); continue
        dist = np.sqrt(((prof - prof.loc[tgt]) ** 2).mean(1)).drop(tgt).sort_values()
        ref = dist.get("F47" if tgt == "F13" else "F13", np.nan)
        rows.append(dict(type=k, target=tgt, n_farms=len(dist), twin_dist=round(ref, 2), top3=", ".join(f"{f}:{v:.2f}" for f, v in dist.drop(["F13", "F47"], errors="ignore").head(3).items()),
                         median_dist=round(dist.median(), 2), within_twin=int((dist.drop(["F13", "F47"], errors="ignore") <= ref).sum()) if np.isfinite(ref) else -1))
print(pd.DataFrame(rows).to_string(index=False))
typ.rename("type").reset_index().to_csv(os.path.join(HERE, "dty1_types_v1.csv"), index=False)
