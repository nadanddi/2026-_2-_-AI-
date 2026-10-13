# -*- coding: utf-8 -*-
"""EP1R (2026-10-14 집 클로드; 사용자: '2 → 1 순서' — 1 = 운영 방식이 바뀌면 고EC 가 끝나나/시작하나).  서술.  EC.
같은 동 쌍: (가) cert 1일 쌍(끊은 것 포함, 1:1 정리 전 전체), (나) CP2 빈 날 연결.  쌍마다 하루 운영 지문 거리(구동기·CO2 8열, 학습 표준화)와 상태 변화:
  유지(고→고, 보통→보통), 끝(고→보통), 시작(보통→고)  [고 = 하루 평균 >= 1.0].  운영 지문 유형(CP2 KMeans 4; 유형 1 = 고EC 운영) 의 변화도.
보고: 상태 변화별 지문 거리 분포, 끝/시작 판별 AUC(거리), 유형 전환과 상태 전환의 교차표.  F13_225→227 의 거리(.58 계열)를 학습 분포에 놓아 봄."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
T = pd.read_csv(os.path.join(HERE, "cp2_tracks_v1.csv")).set_index("rec")
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
FP = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FP = (FP - np.nanmean(FP[TRAIN], (0, 1))) / (np.nanstd(FP[TRAIN], (0, 1)) + 1e-9); fpd = lambda a, b: float(np.nanmean(np.minimum(np.abs(FP[idx[a]] - FP[idx[b]]), 4)))
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); pairs = [(rid(int(a)), rid(int(b)), "cert1") for a, b, c in zip(TL.a1, TL.b, TL.cert) if c]
pairs += [(p, r, "gap") for r, p, lt in zip(T.index, T.prev, T.link_type) if lt == "gap" and isinstance(p, str)]
rows = []
for a, b, k in pairs:
    if not (np.isfinite(DM[idx[a]]) and np.isfinite(DM[idx[b]])): continue
    ha, hb = DM[idx[a]] >= 1.0, DM[idx[b]] >= 1.0
    rows.append(dict(a=a, b=b, kind=k, fp=fpd(a, b), change=("keep_high" if ha and hb else "keep_normal" if (not ha and not hb) else "END" if ha else "START"),
                     type_a=int(T.at[a, "fp_type"]), type_b=int(T.at[b, "fp_type"]), dEC=DM[idx[b]] - DM[idx[a]]))
P = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(P.groupby(["kind", "change"]).fp.describe()[["count", "50%", "75%", "max"]].round(3).to_string())
for ch in ("END", "START"):
    m = P.change.isin([ch, "keep_high" if ch == "END" else "keep_normal"])
    G = P[m]
    if G.change.nunique() == 2: print(f"AUC fingerprint distance for {ch} vs keep: {roc_auc_score(G.change == ch, G.fp):.3f} (n {ch} {int((G.change == ch).sum())}, keep {int((G.change != ch).sum())})")
P["type_switch"] = np.select([(P.type_a == 1) & (P.type_b != 1), (P.type_a != 1) & (P.type_b == 1)], ["left type1", "entered type1"], "same side")
print("\noperation-type switch (type1 = high-EC operation) x state change:"); print(pd.crosstab(P.type_switch, P.change).to_string())
print("\nEND / START cases:"); print(P[P.change.isin(["END", "START"])].sort_values("change")[["a", "b", "kind", "fp", "type_a", "type_b", "dEC"]].round(3).to_string(index=False))
ev = [("F13_225", "F13_227"), ("F13_223", "F13_225"), ("F13_221", "F13_223"), ("F47_219", "F47_221")]
print("\neval consecutive same-house (number-2) distances + types:", {f"{a}>{b}": (round(fpd(a, b), 3), int(T.at[a, 'fp_type']), int(T.at[b, 'fp_type'])) for a, b in ev})
P.to_csv(os.path.join(HERE, "ep1r_pairs_v1.csv"), index=False)
