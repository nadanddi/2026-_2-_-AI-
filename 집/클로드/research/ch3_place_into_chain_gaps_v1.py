# -*- coding: utf-8 -*-
"""CH3 (diagnostic, fixed before running; 2026-10-06 집 클로드).  Can a held-out (pseudo-evaluation)
record be placed after its true predecessor using training label chains (CH2)?
Per DIAG10 fold: reference = labelled records outside the fold; CH2 chains rebuilt on the reference
only (same cost / C0 = 12).  For each held-out pass-2 day q:
  candidates = reference records p with outdoor date_cost(p -> q) <= 3 (uses q's hours 0-1 only);
  OPEN = candidates that are chain ENDS in the reference chains (no successor) - the other 동's record of
         that date normally already continues into its own next-day record;
  pick:  among OPEN candidates (else all candidates) the one with the lowest indoor+actuator continuity
         cost (q hours 0-1 vs p hours 22-23);
  level: at hour h, pred = EC_p(23) + (m_h - m_0)  (label level, model's within-day change; m = R3S)
Variants reported: OPEN-first (main), ALL (no chain information), and ORACLE (candidate whose EC_p(23)
is nearest the true EC_q(0)).  Also a guarded version: use the chain level only when exactly one OPEN
candidate exists, else the model.  Metrics on pass-2 rows: share of days whose chosen EC_p(23) is within
.05 of the true EC_q(0); row RMSE vs model (R3S seed mean); normal / high.
Clue (fixed): main or guarded version beats the model on pass-2 rows AND normal days, with
correct-placement share >= .7."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch2_label_chains_training_v1.py"), encoding="utf-8").read()
exec(src.split("rows = []")[0])
XT = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
XT = XT[XT.row_id.str[:3].isin(["F13", "F47"])].copy()
XT["farm"], XT["day"], XT["hour"] = XT.row_id.str[:3], XT.row_id.str[4:7].astype(int), XT.row_id.str[8:10].astype(int)
PA = XT.pivot_table(index=["farm", "day"], columns="hour", values=W + list(IN) + ACT)


def jmp(f, a, b, v):
    A23, A22, B0, B1 = PA.loc[(f, a), (v, 23)], PA.loc[(f, a), (v, 22)], PA.loc[(f, b), (v, 0)], PA.loc[(f, b), (v, 1)]
    return (B0 - A23) - .5 * ((A23 - A22) + (B1 - B0))


def dcost(f, a, b):
    return sum(WT[v] * ((jmp(f, a, b, v) / SC[(f, v)]) ** 2) for v in W)


def icost(f, a, b):
    c = sum(((jmp(f, a, b, v) / s) ** 2) for v, s in IN.items())
    return c + sum(abs(PA.loc[(f, b), (v, 0)] - PA.loc[(f, a), (v, 23)]) / 50 for v in ACT)


# full pair-cost matrices among labelled records (CH2 cost), computed once
FULLM = {}
for f in ("F13", "F47"):
    D = sorted(P.loc[f].index); n = len(D); M = np.full((n, n), 1e6)
    for i, a in enumerate(D):
        for j, b in enumerate(D):
            if a != b:
                ce = (jump(f, a, b, "sub_ec") / .02) ** 2
                c = ce + dcost(f, a, b) + icost(f, a, b) / 4
                M[i, j] = c if np.isfinite(c) else 1e6
    FULLM[f] = (D, M)


def chains(f, keep):
    D, M = FULLM[f]; ix = [i for i, d in enumerate(D) if d in keep]; Dk = [D[i] for i in ix]; Mk = M[np.ix_(ix, ix)]; n = len(Dk)
    big = np.full((2 * n, 2 * n), 1e6); big[:n, :n] = Mk
    big[:n, n:] = np.where(np.eye(n) == 1, C0, 1e6); big[n:, :n] = np.where(np.eye(n) == 1, C0, 1e6); big[n:, n:] = 0
    r_, c_ = linear_sum_assignment(big)
    succ = {Dk[i]: Dk[j] for i, j in zip(r_, c_) if i < n and j < n}
    return set(Dk), succ


O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"))
Fd = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"), usecols=["farm", "day", "validator", "validation_fold"])
Fd = Fd[Fd.validator == "DIAG10"].drop_duplicates(["farm", "day"])
O = O[O.day >= 179].sort_values(["farm", "day", "hour"]).copy()
O["m0"] = O.groupby(["farm", "day"]).p.transform("first")
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
res = []
for k in sorted(O.validation_fold.unique()):
    vd = set(zip(Fd[Fd.validation_fold == k].farm, Fd[Fd.validation_fold == k].day))
    for f in ("F13", "F47"):
        keep = {d for d in P.loc[f].index if (f, d) not in vd}
        ref, succ = chains(f, keep)
        for d in sorted({dd for ff, dd in vd if ff == f and dd >= 179}):
            cands = [p for p in ref if np.isfinite(dcost(f, p, d)) and dcost(f, p, d) <= 3]
            y0 = P.loc[(f, d), ("sub_ec", 0)]
            row = dict(farm=f, day=d, n_cand=len(cands))
            if cands:
                openc = [p for p in cands if p not in succ]
                row["n_open"] = len(openc)
                pick_all = min(cands, key=lambda p: icost(f, p, d))
                pick_open = min(openc, key=lambda p: icost(f, p, d)) if openc else pick_all
                pick_orc = min(cands, key=lambda p: abs(P.loc[(f, p), ("sub_ec", 23)] - y0))
                for tag, p in (("open", pick_open), ("all", pick_all), ("orc", pick_orc)):
                    row[tag] = P.loc[(f, p), ("sub_ec", 23)]
                row["guard"] = row["open"] if len(openc) == 1 else np.nan
            res.append(row)
Rz = pd.DataFrame(res)
O = O.merge(Rz, on=["farm", "day"], how="left")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
yz = O[O.hour == 0].set_index(["farm", "day"]).sub_ec
Rz["y0"] = [yz.get((f, d)) for f, d in zip(Rz.farm, Rz.day)]
print("pass-2 held-out days %d; with candidates %d (median %d), with OPEN candidates %d, exactly one OPEN %d" % (
    len(Rz), Rz.n_cand.gt(0).sum(), Rz.n_cand.median(), Rz.n_open.gt(0).sum(), (Rz.n_open == 1).sum()))
for tag in ("open", "all", "orc", "guard"):
    ok = (Rz[tag] - Rz.y0).abs() <= .05
    print("  %-6s level within .05 of true 0h EC: %d of %d days with a value (%.0f%%)" % (tag, ok.sum(), Rz[tag].notna().sum(), 100 * ok.sum() / max(1, Rz[tag].notna().sum())))
out = {}
for tag in ("open", "all", "orc", "guard"):
    O["pred_" + tag] = np.where(O[tag].notna(), O[tag] + (O.p - O.m0), O.p)
for nm, m in (("pass-2", O.dm.notna()), ("normal", O.dm < 1), ("high", O.dm >= 1)):
    g = O[m]
    print("  %-7s model %.4f | open %.4f | all %.4f | guard %.4f | oracle %.4f" % (nm, r(g.p - g.sub_ec), r(g.pred_open - g.sub_ec),
          r(g.pred_all - g.sub_ec), r(g.pred_guard - g.sub_ec), r(g.pred_orc - g.sub_ec)))
O.to_csv(os.path.join(env.LOCAL, "ch3_rows_v1.csv"), index=False); Rz.to_csv(os.path.join(env.LOCAL, "ch3_days_v1.csv"), index=False)
share = ((Rz.open - Rz.y0).abs() <= .05).mean()
clue = share >= .7 and any(r(O["pred_" + t] - O.sub_ec) < r(O.p - O.sub_ec) and r((O["pred_" + t] - O.sub_ec)[O.dm < 1]) < r((O.p - O.sub_ec)[O.dm < 1]) for t in ("open", "guard"))
print("CH3 clue:", clue)
