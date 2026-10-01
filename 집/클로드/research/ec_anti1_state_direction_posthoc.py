# -*- coding: utf-8 -*-
"""ANTI1 (POST-HOC exploration, motivated by round 7's score): does the public
validation also prefer the OPPOSITE of Codex's past-public-EC state change?
q = v2_s + lam * (state_s - v2_s); state_s = Codex ec_v2_state candidate OOF.
lam = +1 is the submitted state model (round 7), lam < 0 is 'anti-state'.
Reports the per-validator optimal lam and RMSE at fixed lam in {-1,-0.5,0,+1}.
Exploratory only; any candidate needs a fresh pre-registered confirmation
(e.g. Codex's unscored final-lock 40 days).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_anti1_state_direction_posthoc.py"""
import env, os, glob, numpy as np, pandas as pd
R = env.ROOT
o = pd.read_csv(os.path.join(R, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1", "oof_predictions.csv"), encoding="utf-8-sig")
rows = []
for f in glob.glob(os.path.join(R, u"집", u"코덱스", "local", "ec_v2_state_20261001_v1", "*_state.npz")):
    nm = os.path.basename(f).replace("_state.npz", "")
    val, fold = nm.rsplit("_", 1)
    z = np.load(f, allow_pickle=True)
    rows.append(pd.DataFrame({"row_id": z["row_id"], "validator": val, "validation_fold": int(fold),
                              **{"st_%d" % s: z["candidate_%d" % s] for s in (7, 101, 2024)}}))
S = pd.concat(rows)
m = o.merge(S, on=["row_id", "validator", "validation_fold"], how="inner")
print("merged rows", len(m), "of", len(o))
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
for v in ("DIAG10", "A", "B", "EXT10", "EXT12"):
    g = m[m.validator == v]
    out = []
    for s in (7, 101, 2024):
        e0 = g["v2_%d" % s] - g.sub_ec; d = g["st_%d" % s] - g["v2_%d" % s]
        lam = -(e0 * d).sum() / (d * d).sum()
        out.append("s%d lam* %+.2f | " % (s, lam) + " ".join("%+.1f:%.4f" % (L, r(e0 + L * d)) for L in (-1, -0.5, 0, 1)))
    print("%-6s %s" % (v, "\n       ".join(out)))
g = m[(m.validator == "DIAG10") & (m.day >= 179)]
for s in (7,):
    e0 = g["v2_%d" % s] - g.sub_ec; d = g["st_%d" % s] - g["v2_%d" % s]
    print("DIAG10 late>=179 s7: lam* %+.2f | " % (-(e0 * d).sum() / (d * d).sum()) + " ".join("%+.1f:%.4f" % (L, r(e0 + L * d)) for L in (-1, -0.5, 0, 1)))
