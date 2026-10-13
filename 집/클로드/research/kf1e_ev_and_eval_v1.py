# -*- coding: utf-8 -*-
"""KF1E (2026-10-13 집 클로드; 사용자: 'KF1 의 1) EV 교차 확인, 2) 평가 적용량').  서술(새 판정 아님 — KF1 규칙·측정 분산 그대로, EV 는 처음 보는 하네스).
(1) EV 온도 하네스(8폴드, 4칸 = BASE 7/101 × PFN A/B): 기준 = v22(13회차형) + TNA8(v15R EV 짝 = HP2 짝 또는 p0 ≥ .8), KFa = 기준 + 앵커 없는 기록에 KF 추정.
    짝 = v15r15_ev(폴드별), 사슬 중간 기록은 같은 폴드의 val 기록(짝이 있는 경우)만, 정답 보임 = 학습 기록 & 폴드 val∪del 아님.  q = v18(앵커 전) 칸 값.
(2) 평가: 14회차 온도(submission_23) + KFa(평가 짝 v15r15_eval, 사슬 중간 = 같은 ID 앞 번호 평가 기록, 정답 보임 = 학습 기록, q = 11회차 온도(앵커 전)).
    바뀌는 기록·행·RMS·점수 산수.
"""
import os, sys, glob
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tb1_temp_null_agree_anchor_v1.py")
exec(src[:src.index("PE15 = ")])   # hp2, v15_new, ST, idx, TRAIN, FARM, NUM, N, A7(BETA), CLEAN, rid
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261126)
S2, L2, H2, BIAS = .549, .342, .239, .36; B23 = float(A7.BETA[23])          # measured in KF1 (6.810)
def kf(r, picks, Qm, visible, inq):
    chain = []; cur = r; P = 1.0
    for k in range(3):
        if cur not in picks: break
        pv, p0 = picks[cur]
        if not isinstance(pv, str): break
        P *= float(p0)
        if P < .3: break
        vis = visible(pv) and np.isfinite(ST[idx[pv], 23]); chain.append(pv)
        if vis: break
        if pv not in inq: return None, 0
        cur = pv
    if not chain or not (visible(chain[-1]) and np.isfinite(ST[idx[chain[-1]], 23])): return None, 0
    est23 = ST[idx[chain[-1]], 23]; v = L2
    for pv2 in reversed(chain[:-1]):
        g = (est23 - BIAS) - Qm[pv2][0]; est23 = Qm[pv2][23] + B23 * g; v = B23 ** 2 * v + H2
    g0 = (est23 - BIAS) - Qm[r][0]; return float(np.clip(S2 / (S2 + v) * P * g0, -2, 2)), len(chain)
# ---------------- (1) EV ----------------
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hid = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
foldof = {r: k for k, g in FO[FO.role == "val"].groupby("fold") for r in g.rec}
lab_ev = lambda r: TRAIN & ~np.array([rid(i) in hid[foldof[r]] for i in range(N)])
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7set = {r for r, q, l, d in zip(L7.rec, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv"))
TNA8E = {**v15_new(PE, lab_ev, v7set, 0.0, True), **v15_new(PE, lab_ev, v7set, .8, False)}
pk_fold = {k: {r: (pv, p0) for r, pv, p0 in zip(g.rec, g.prev0, g.p0)} for k, g in PE.groupby("fold")}
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]; DD = DD.reset_index(drop=True)
y = DD.sub_temp.values; p2 = DD.day.values >= 179; hh = DD.hour.values; rows = []; ds_all = []
def per_rec(col):
    return {r: g.sort_values("hour")[col].values for r, g in DD.groupby("rec")}
for c in ["7A", "7B", "101A", "101B"]:
    v22 = DD[f"v22_a_{c}"].values; q18 = DD[f"v18_a_{c}"].values; base = v22.copy()
    for rec, G in DD.groupby("rec"):
        if rec in TNA8E and TNA8E[rec] not in hid[int(G.fold.iloc[0])]:
            ix = G.index.values; q0 = q18[ix[G.hour.values == 0]][0]; base[ix] = q18[ix] + A7.BETA[G.hour.values] * np.clip(ST[idx[TNA8E[rec]], 23] - .36 - q0, -2, 2)
    anchored = set(DD.rec[np.abs(v22 - q18) > 1e-9]) | set(TNA8E)
    Qm = {r: g.sort_values("hour").set_index("hour")[f"v18_a_{c}"].reindex(range(24)).values for r, g in DD.groupby("rec")}
    new = base.copy(); fired = {}
    for rec, G in DD.groupby("rec"):
        if rec in anchored: continue
        k = int(G.fold.iloc[0]); vis = lambda pv, k=k: bool(TRAIN[idx[pv]]) and pv not in hid[k]
        inq = {r for r in pk_fold[k] if r in Qm}
        g, st = kf(rec, pk_fold[k], Qm, vis, inq)
        if g is None: continue
        ix = G.index.values; new[ix] = base[ix] + A7.BETA[G.hour.values] * g; fired[rec] = st
    rows.append(dict(cell=c, all_pct=100 * (rm(new - y) / rm(base - y) - 1), p2_pct=100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1), fired=len(fired), steps=pd.Series(fired).value_counts().to_dict()))
    ds_all.append(pd.Series((new - y) ** 2 - (base - y) ** 2).groupby((DD.farm + "_" + (DD.day // 5).astype(str)).values).mean())
pd.set_option("display.width", 200); print("== EV harness, KFa vs 14th-type (v22 + TNA8):"); print(pd.DataFrame(rows).round(3).to_string(index=False))
v = pd.concat(ds_all, axis=1).mean(axis=1).values; ix = rng.integers(0, len(v), (20000, len(v))); print(f"   block bootstrap P(worse) {float((v[ix].mean(1) >= 0).mean()):.4f}")
# ---------------- (2) evaluation application ----------------
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P11 = pd.read_csv(os.path.join(env.ROOT, "제출", "11회차_2026-10-11(나)", "submission_18.csv")).set_index("row_id").reindex(P14.row_id).reset_index()
P14["rec"] = P14.row_id.str[:7]; P14["h"] = P14.row_id.str[8:].astype(int)
E15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_eval.csv")); pk = {r: (pv, p0) for r, pv, p0 in zip(E15.rec, E15.prev0, E15.p0)}
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].row)
tna8e = {**v15_new(E15, lambda r: TRAIN.copy(), v7e, 0.0, True), **v15_new(E15, lambda r: TRAIN.copy(), v7e, .8, False)}
anch = v7e | set(tna8e)
Qe = {r: g.sort_values("h").sub_temp.values for r, g in P11.assign(rec=P11.row_id.str[:7], h=P11.row_id.str[8:].astype(int)).groupby("rec")}
t0 = P14.sub_temp.values; tn = t0.copy(); fired = {}
for rec, G in P14.groupby("rec"):
    if rec in anch: continue
    g, st = kf(rec, pk, Qe, lambda pv: bool(TRAIN[idx[pv]]), set(Qe))
    if g is None: continue
    ix = G.index.values; tn[ix] = t0[ix] + A7.BETA[G.h.values] * g; fired[rec] = (st, round(g, 3))
d = tn - t0; r_ = rm(d); s0 = .4864
print(f"\n== evaluation: KFa on top of round 14: records {len(fired)} {fired} | rows changed {int((np.abs(d) > 1e-9).sum())} | RMS {r_:.4f}")
for rho in (-.3, -.2, -.15, -.1, 0): print(f"   corr {rho:+.2f} -> temp {np.sqrt(s0 ** 2 + r_ ** 2 + 2 * rho * r_ * s0):.4f}")
print(f"   breakeven corr {-r_ / (2 * s0):+.3f}")
