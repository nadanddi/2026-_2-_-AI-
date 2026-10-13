# -*- coding: utf-8 -*-
"""LK3 (2026-10-14 집 클로드; IV1 후속).  서술·탐색(판정 아님).  LK2 짝 정답 확률 모형 + 동 불변값(IV1) 차이 특징.
불변값은 규정을 지키려 '대상 기록 b 의 0~5시(밤)만' 으로 계산(짝 a 도 같은 정의로 0~5시 — a 는 앞 기록이라 전체를 써도 되지만 정의를 맞춤):
  fan_n = 유동팬 가동 비율, heat_n = 난방 켜진 시각 실내온 최소, co2_n = in_co2 최소, insul_n = 난방 꺼진 시각 (실내−외기) 중앙, hum_n = in_hum 최대.
  특징 = |f(b) − f(a)| (결측이면 중앙값 대치 + 결측 표시).
비교: LK2(10특징) vs LK3(10 + 불변값 5 + 결측 표시) — 둘 다 거울 표 cert 짝으로 적합, EV 표 cert 짝으로 평가(AUC, 문턱 .8 정확도·개수).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int); _X = _X[_X.h < 6]
g = _X.groupby("rec")
INV = pd.DataFrame(dict(fan_n=g.act_circfan.apply(lambda s: float((s > 0).mean())),
                        heat_n=g.apply(lambda d: d.in_temp[d.act_heating > 0].min() if (d.act_heating > 0).any() else np.nan),
                        co2_n=g.in_co2.min(), insul_n=g.apply(lambda d: (d.in_temp - d.out_temp)[~(d.act_heating > 0)].median() if (~(d.act_heating > 0)).any() else np.nan),
                        hum_n=g.in_hum.max()))
F0 = ["lp0", "margin", "p_null", "cross", "dTin", "dTout", "date_yes", "date_no", "same_id_earlier", "hp2_agree"]
def add(T):
    for c in INV.columns:
        d = np.abs(INV.reindex(T.rec)[c].values - INV.reindex(T.prev)[c].values); T[f"d_{c}"] = d
    return T
MT = add(pd.read_csv(os.path.join(HERE, "lk2_mirror_hops_q_v1.csv"))); ET = add(pd.read_csv(os.path.join(HERE, "lk2_ev_hops_q_v1.csv")))
DF = [f"d_{c}" for c in INV.columns]
for T in (MT, ET):
    for c in DF: T[c + "_na"] = T[c].isna().astype(float); T[c] = T[c].fillna(MT[c].median())
F1 = F0 + DF + [c + "_na" for c in DF]
fit, ev = MT[MT.ok.notna()], ET[ET.ok.notna()]
res = {}
for nm, feats in (("LK2", F0), ("LK3", F1)):
    sc = StandardScaler().fit(fit[feats]); LR = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(fit[feats]), fit.ok.astype(int)); q = LR.predict_proba(sc.transform(ev[feats]))[:, 1]
    res[nm] = q; print(f"{nm}: EV AUC {roc_auc_score(ev.ok, q):.3f}", " | ".join(f"q>={t}: n {int((q >= t).sum())} acc {ev.ok[q >= t].mean():.3f}" for t in (.6, .8, .9)))
    if nm == "LK3": print("   LK3 coefficients (invariant diffs):", dict(zip(DF, np.round(LR.coef_[0][len(F0):len(F0) + len(DF)], 3))))
print("single-feature AUC on EV cert hops (lower diff = correct, so AUC of -diff):", {c: round(roc_auc_score(ev.ok, -ev[c]), 3) for c in DF})
ev = ev.assign(q2=res["LK2"], q3=res["LK3"]); print("DS8 culprit hops:", ET[ET.rec.isin(["F47_126", "F13_160"])][["rec", "prev"]].to_dict("records"))
print(ev[ev.rec.isin(["F47_126", "F13_160"])][["rec", "prev", "ok", "q2", "q3"]].round(3).to_string(index=False))
