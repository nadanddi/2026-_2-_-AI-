# -*- coding: utf-8 -*-
"""submission_23 temperature (2026-10-13 집 클로드, LEADERBOARD HYPOTHESIS TEST; catalog 6.789, 6.800).
pred = round-13 temperature (temp_candidate_v22.csv) + extra anchors TNA8 with the round-13 anchor formula:
    pred_h = q_h + BETA[h] * clip(A - .36 - q_0, -2, +2),   A = 23 h substrate label of the chosen previous record (a TRAINING record)
TNA8 = records NOT already anchored by round 13 (v7 confident 1-step anchors with |dT_air| <= 1.5) whose v15R first pick is a labelled
  training record AND (a) the pick equals the HP2 pick (date-first: training-only DMT2 calendar date d from records with identical hour-0
  outdoor -> labelled records of date d-1 in the same segment whose certain next day is not already a visible labelled record -> nearest
  hour-0 operation fingerprint, standardised on TRAINING records only)  OR  (b) v15R confidence p0 >= .8.
Local evidence (6.800): all-rows improvement on every layout (DIAG10 -1.8 %, EL1 -3.8 %, EXT12 -2.5 %, EXT10 -0.3..-0.4 %), pass-2 EV -3.4 %,
  but one or two harmed records (correct link, substrate jumped at midnight) -> strict local rule FAIL; submitted as a hypothesis test.
Run (from 온도/):  LINKER_WORK=<dir with base.pkl/truth.pkl> LINKER_DATA=<folder with train_X.csv,test_X.csv>
       python code/make_temp_v23_tna8.py ../temp_candidate_v22.csv <v15r eval picks csv> <eval60_confident_1step_anchor_v7.csv> ../temp_candidate_v23.csv
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PKG_T = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG_T, "linker_v7"))
from feat4 import ST, X as OUTX, FARM, NUM, N, TRAIN, S3  # noqa: E402
C, CLIP = 0.36, 2.0
BETA = np.array([0.7648, 0.6232, 0.5525, 0.4844, 0.4091, 0.3453, 0.3025, 0.2576, 0.1866, 0.1264, 0.1570, 0.1661, 0.1655, 0.1566,
                 0.1999, 0.2032, 0.2233, 0.2263, 0.2145, 0.2008, 0.1897, 0.1721, 0.1991, 0.2355])   # same as make_temp_v22_anchor.py
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}


def main():
    t_path, pk_path, a7_path, out = sys.argv[1:5]
    assert not os.path.exists(out), "refusing to overwrite " + out
    DT = pd.read_csv(os.path.join(PKG_T, "linker_v15r", "data", "dmt2_days.csv"))
    DAYT = np.array([dict(zip(DT.rec, DT.day)).get(rid(i), np.nan) for i in range(N)]); SEGT = np.array([dict(zip(DT.rec, DT.seg)).get(rid(i), np.nan) for i in range(N)])
    TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); TLc = TL[TL.cert]; SUCC = {}
    for a_, b_ in zip(TLc.a1, TLc.b): SUCC.setdefault(int(a_), []).append(int(b_))
    V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
    D = os.environ["LINKER_DATA"]; XX = pd.concat([pd.read_csv(os.path.join(D, f)) for f in ("train_X.csv", "test_X.csv")]); XX = XX[XX.row_id.str[:3].isin(["F13", "F47"])]
    XX["rec"] = XX.row_id.str[:7]; XX["h"] = XX.row_id.str[8:].astype(int)
    FPA = np.stack([XX.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in V], 2)
    FPA = (FPA - np.nanmean(FPA[TRAIN], (0, 1))) / (np.nanstd(FPA[TRAIN], (0, 1)) + 1e-9)       # training records only
    def hp2(b, lab):
        vi = np.where(lab)[0]; vi = vi[vi != b]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1)]; m = m[np.isfinite(DAYT[m])]
        if len(m) == 0: return None
        d = pd.Series(DAYT[m]).mode().iloc[0]; sg = pd.Series(SEGT[m][DAYT[m] == d]).mode().iloc[0]
        cand = [a for a in vi if DAYT[a] == d - 1 and SEGT[a] == sg and not any(lab[s] and DAYT[s] == d for s in SUCC.get(a, []))]
        if not cand: return None
        return rid(min(cand, key=lambda a: float(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0]), 4)))))
    A7 = pd.read_csv(a7_path, encoding="utf-8-sig"); v7e = set(A7[A7.dT_air.abs() <= 1.5].row)
    P = pd.read_csv(pk_path); lab = TRAIN.copy(); AN = {}
    for r, pv, p0, lb in zip(P.rec, P.prev0, P.p0, P.lab0):
        if r in v7e or not isinstance(pv, str) or not bool(lb): continue
        if p0 >= .8 or hp2(idx[r], lab) == pv: AN[r] = float(ST[idx[pv], 23])
    T = pd.read_csv(t_path); k = T.row_id.str[:7]; h = T.row_id.str[8:10].astype(int)
    q0 = T[h == 0].set_index(k[h == 0]).sub_temp; new = T.sub_temp.values.copy()
    for i, (kk, hh) in enumerate(zip(k, h)):
        if kk in AN: new[i] = T.sub_temp.values[i] + BETA[hh] * np.clip(AN[kk] - C - q0[kk], -CLIP, CLIP)
    S = T[["row_id"]].copy(); S["sub_temp"] = new
    assert len(S) == 1440 and np.isfinite(S.sub_temp).all()
    S.to_csv(out, index=False, float_format="%.6f"); d = new - T.sub_temp.values
    print("TNA8 anchors:", {r: round(v, 2) for r, v in sorted(AN.items())})
    print("wrote %s | rows changed %d, RMS %.4f, mean %+.4f" % (out, int((np.abs(d) > 1e-9).sum()), np.sqrt((d ** 2).mean()), d.mean()))


if __name__ == "__main__":
    main()
