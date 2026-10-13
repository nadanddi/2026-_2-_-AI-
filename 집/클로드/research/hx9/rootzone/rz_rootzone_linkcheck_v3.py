# -*- coding: utf-8 -*-
"""HX9 근권 온도 가족 — 링크 점검 v3 (2026-10-13 집 클로드, 사후 점검).
v2 에서 Y3(Δ EC ~ Δ배지온) 관계가 다른 온실 id 사이 링크(n 10, r .86)에만 있고 같은 id 링크(n 17)에서는 r ≈ 0 이었다.
다른 id 링크가 '같은 동의 다음 날'이 아니라 '다른 동'일 수 있는지(그럼 Δ는 날짜 변화가 아니라 동 간 차이) 서술로 본다.
보는 것: 링크 비용 c1·cert, 자정 EC/배지온 점프, Δ배지온 크기, 같은 날짜 쌍둥이 중 더 비슷한 후보 존재 여부.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import env  # noqa: F401
import numpy as np, pandas as pd
pd.set_option("display.width", 250)
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
V = pd.read_csv(os.path.join(HERE, "vars_v1.csv"), index_col="rec")
R = pd.read_csv(os.path.join(HX9, "data", "records.csv"), index_col="rec")
L = R[(R.hi10 == 1) & (R.prev_hi10 == 1)].copy()
L["prev_st_h23"] = [R.at[p, "st_h23"] for p in L.prev_rec]
L["jump_ec"] = L.ec_h0 - L.prev_ec_h23
L["jump_st"] = L.st_h0 - L.prev_st_h23
L["d_st_mean"] = V.loc[L.index, "st_mean"].values - V.loc[L.prev_rec, "st_mean"].values
L["dlnEC"] = np.log(L.ec_mean) - np.log(L.prev_ec_mean)
L["d_st_in_night"] = V.loc[L.index, "st_in_night"].values - V.loc[L.prev_rec, "st_in_night"].values
L["d_heat_night"] = V.loc[L.index, "heat_night"].values - V.loc[L.prev_rec, "heat_night"].values
cols = ["prev_rec", "prev_same_farm", "a1_c1", "a1_cert", "ep_id", "prev_ec_mean", "ec_mean", "dlnEC", "d_st_mean", "jump_ec", "jump_st",
        "d_st_in_night", "d_heat_night", "pass2"]
print(L[cols].sort_values(["prev_same_farm", "ep_id"]).round(3).to_string())
for sf in (1, 0):
    S = L[L.prev_same_farm == sf]
    print(f"same_farm={sf}: n {len(S)}, median c1 {S.a1_c1.median():.2f}, cert {S.a1_cert.mean():.2f}, |jump_ec| med {S.jump_ec.abs().median():.3f}, "
          f"|jump_st| med {S.jump_st.abs().median():.2f}, |d_st_mean| med {S.d_st_mean.abs().median():.2f}, |dlnEC| med {S.dlnEC.abs().median():.3f}, "
          f"r(dlnEC,d_st) {np.corrcoef(S.dlnEC, S.d_st_mean)[0,1]:+.2f}, r(dlnEC,d_st_in_night) {np.corrcoef(S.dlnEC, S.d_st_in_night)[0,1]:+.2f}, "
          f"r(dlnEC,d_heat_night) {np.corrcoef(S.dlnEC, S.d_heat_night)[0,1]:+.2f}")
# all clean links: same vs cross farm slope
A = R[R.prev_rec.notna()].copy()
A["d_st"] = V.loc[A.index, "st_mean"].values - V.loc[A.prev_rec, "st_mean"].values
A["dln"] = np.log(A.ec_mean) - np.log(A.prev_ec_mean)
for sf in (1, 0):
    S = A[A.prev_same_farm == sf]
    print(f"all clean links same_farm={sf}: n {len(S)}, r(dlnEC, d_st) {np.corrcoef(S.dln, S.d_st)[0,1]:+.3f}, slope {np.polyfit(S.d_st, S.dln, 1)[0]:+.4f}, "
          f"median c1 {S.a1_c1.median():.2f}")
