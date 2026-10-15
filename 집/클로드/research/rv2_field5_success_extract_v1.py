# -*- coding: utf-8 -*-
"""RV2 (2026-10-15 집 클로드; 사용자: '5점 발표에 쓸 만한 모델별로 안 된 부분·왜·잘된 부분만 따로 추출할 수 없는지').  서술(사후), 실행 전 커밋.  EC.
SZ2 독립 기본 예측 위 저장 결과로, 각 후보가 '기준' 대비 어느 상황에서 이득/손해였는지 같은 틀로 분해:
  상황 = 2차 보통 / 2차 고EC / 링크(26번 규칙 MASK 거울 p0 ≥ .5) 있음·없음 / 1차(DIAG10 만) / 시각대 0-7·8-15·16-23.
  후보(기준): HR6(HR5, ec_rv1)·HR6R(HR5, SZ2)·EV3LH(HR5, ec_rv1)·CRG(F5E, SZ2)·GT1 A(F5E, SZ2)·GT1 고친 판 A'(링크 있으면 .5NH+.5EVY, 없으면 F5E; 사후)·
  E4(F5E + v7 확신 링커 EC 앵커, A3)·EA3 AN8(F5E + MASK 거울 링커 앵커).
보고: 상황별 상대 RMSE 변화 %, 상황별 기록 수.  '추출 후보' = 세 검증기 모두 개선된 상황 칸.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u rv2_field5_success_extract_v1.py
"""
import os, sys, pickle, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC as ECA, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); SEEDS = (1357, 2468, 3579)
Lm = pd.read_csv(os.path.join(env.LOCAL, "v15r_mirror_masked_pkg25.csv")); LINK = {r: p for r, p, q in zip(Lm.rec, Lm.prev0, Lm.p0) if isinstance(p, str) and q >= .5 and p != r}
def segs(D):
    p2 = D.p2.values; hi = D.hi.values; lk = D.rec.isin(LINK.keys()).values; h = D.hour.values
    return {"p2 normal": p2 & ~hi, "p2 high": p2 & hi, "p2 linked": p2 & lk, "p2 unlinked": p2 & ~lk, "p1": ~p2, "h0-7": p2 & (h <= 7), "h8-15": p2 & (h >= 8) & (h <= 15), "h16-23": p2 & (h >= 16)}
def table(name, D, cand, base, seeds):
    out = []
    for v, g in D.groupby("validator"):
        for sn, m in segs(g).items():
            if not m.any(): continue
            y = g.y.values; ch = np.mean([rm((g[cand.format(s)].values - y)[m]) / rm((g[base.format(s)].values - y)[m]) - 1 for s in seeds])
            out.append(dict(model=name, val=v, seg=sn, pct=100 * ch, n_rec=int(g[m].rec.nunique())))
    return out
rows = []
# ---- SZ2-based: HR6R, CRG, GT1, E4, EA3 ----
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); MTp = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); EP = pickle.load(open(os.path.join(env.LOCAL, "nh2_ev3l_v1.pkl"), "rb"))
CP = pickle.load(open(os.path.join(env.LOCAL, "crg_preds_v1.pkl"), "rb")); EY = pickle.load(open(os.path.join(env.LOCAL, "gt1_evy_v1.pkl"), "rb"))
Z = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(env.LOCAL, "sz2_ckpt", "*.csv"))], ignore_index=True).rename(columns={"validation_fold": "fold"}); Z["rec"] = Z.farm + "_" + Z.day.map("{:03d}".format)
R = R.merge(Z[["validator", "fold", "rec", "hour"] + [f"pre_SEA_{s}" for s in SEEDS]], on=["validator", "fold", "rec", "hour"], how="left")
mt = np.array([MTp[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); ev = np.array([EP[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
ey = np.array([EY[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); lk = R.rec.isin(LINK.keys()).values
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
CURE = {r: p for r, p, q, l, dT in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if isinstance(p, str) and q >= .8 and l and abs(dT) <= 1.5 and r in v7set}
MASK8 = {r: p for r, p, q in zip(Lm.rec, Lm.prev0, Lm.p0) if isinstance(p, str) and q >= .8 and p in idx and TRAIN[idx[p]]}
def ea(col, links):
    out = R[col].values.copy()
    for (v, k, r), g in R.groupby(["validator", "fold", "rec"]):
        p = links.get(r)
        if p is None or p not in idx or not np.isfinite(ECA[idx[p], 23]) or p in set(R.rec[(R.validator == v) & (R.fold == k)]): continue
        ii = g.sort_values("hour").index.values; q = R.loc[ii, col].values; pm = np.cumsum(q) / np.arange(1, len(q) + 1); out[ii] = q + (ECA[idx[p], 23] - pm)
    return out
hr6r = pd.read_csv(os.path.join(HERE, "hr6r_rows_v1.csv.gz"))
for s in SEEDS:
    nh = (R[f"A1_{s}"] - .1 * mt) / .9; R[f"F5E_{s}"] = .9 * nh + .1 * ev
    cr = np.array([CP[(v, k)][(r, s)][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); R[f"CRG_{s}"] = .9 * (R[f"pre_SEA_{s}"] + .5 * cr) + .1 * ev
    R[f"GA_{s}"] = np.where(lk, .5 * nh + .5 * ey, nh); R[f"GAF_{s}"] = np.where(lk, .5 * nh + .5 * ey, R[f"F5E_{s}"])
    R[f"E4_{s}"] = ea(f"F5E_{s}", CURE); R[f"EA8_{s}"] = ea(f"F5E_{s}", MASK8)
for name, c in (("CRG", "CRG_{}"), ("GT1 A", "GA_{}"), ("GT1 A' (post-hoc fix)", "GAF_{}"), ("E4 v7-anchor", "E4_{}"), ("EA3 mask-anchor", "EA8_{}")):
    rows += table(name, R, c, "F5E_{}", SEEDS)
H = hr6r.copy()
for v, g in H.groupby("validator"):
    for sn, m in segs(g).items():
        if not m.any(): continue
        y = g.y.values; ch = np.mean([rm((g[g.seed == s].hr6.values - g[g.seed == s].y.values)[m[g.seed.values == s]]) / rm((g[g.seed == s].hr5.values - g[g.seed == s].y.values)[m[g.seed.values == s]]) - 1 for s in SEEDS])
        rows.append(dict(model="HR6R", val=v, seg=sn, pct=100 * ch, n_rec=int(g[m].rec.nunique())))
T = pd.DataFrame(rows); pd.set_option("display.width", 260)
P = T.pivot_table(index=["model", "seg"], columns="val", values="pct").round(2); NREC = T.pivot_table(index=["model", "seg"], columns="val", values="n_rec")
P["all3_better"] = (P[["DIAG10", "EL1", "P2LOO"]] < 0).all(axis=1)
print(P.to_string()); print("\nrecords per cell (pass-2 segs):"); print(NREC.loc[NREC.index.get_level_values(0) == "E4 v7-anchor"].to_string())
P.to_csv(os.path.join(HERE, "rv2_extract_table_v1.csv"))
