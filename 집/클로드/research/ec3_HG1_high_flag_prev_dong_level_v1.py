# -*- coding: utf-8 -*-
"""EC stage-3 HG1: on rows the model already flags as high-EC, adjust the level by
the same 동's recent EC level (fixed before running; 2026-10-04 집 클로드).
Basis: HC3/HC5 (6.242/6.244) - within the 31 high-EC days the day size correlates
with the minimum EC of the same 동's labelled records in the previous 7 dates
(rho +.41), while supply history (public 이레) is unrelated.  ST8 (6.209) carried
previous labels on ALL rows and failed; this restricts to high-flagged rows.
Flag (causal): cm_h = expanding mean of R3S (seed s) over hours 0..h of the record
  >= 0.9.
Information: x = pB*minB + (1-pB)*minA, minA/minB = minimum label day-mean EC of the
  AVAILABLE (not in validation fold +-1, not lock +-1) labelled records of 동 A/B in
  the previous 7 calendar dates (C6.205 date index, same pass); pB as ST8 (causal
  exact-pair flag, else per-hour input classifier); a 동's min missing -> use the
  other; both missing -> no correction.
Correction: Ridge(alpha 1) of (y - R3S) on [x, cm_h] fitted on DIAG10 OOF flagged
  rows whose days are NOT in the evaluated fold's days +-1 (features w.r.t. their own
  DIAG10 fold), clipped to [-0.6, 0.6]; applied to flagged rows only.
Baseline R3S = stored DC5 / EL1 OOF.  Rule (user's): all 3 seeds x 5 validators
better and DIAG10 P(worse) < .025; EL1 better for all seeds also required.
"""
import env  # noqa: F401
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("se3", os.path.join(HERE, "ec3_SE3_same_dong_residual_carry_v1.py"))
se3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(se3)
p3 = se3.p3
SEEDS = (7, 101, 2024)
THR, WIN, CLIP = 0.9, 7, 0.6


def main():
    pB, rec = se3.structure()                      # rec: index (farm, day) -> role, dong, date
    lock = {(z["farm"], int(z["day"])) for z in json.loads(open(p3.LOCK, encoding="utf-8").read())["selected"]}
    Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
    Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
    Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
    ECd = Y.groupby(["farm", "day"]).sub_ec.mean()
    ECd = ECd[[k not in lock for k in ECd.index]]
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "farm", "day", "hour", "sub_ec", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    O = pd.concat([d5, e1[d5.columns]], ignore_index=True)
    O["pB"] = pB.reindex(O.row_id).values
    assert O.pB.notna().all()
    O = O.sort_values(["validator", "validation_fold", "farm", "day", "hour"]).reset_index(drop=True)
    xs = np.full(len(O), np.nan)
    for (v, k), G in O.groupby(["validator", "validation_fold"]):
        vd = set(zip(G.farm, G.day))
        fb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        for (f, d), idx in G.groupby(["farm", "day"]).groups.items():
            dt, ps = rec.loc[(f, d), "date"], d >= 179
            mins = {}
            for dg in ("A", "B"):
                vals = [ECd[(f, q)] for (ff, q) in rec.index if ff == f and (q >= 179) == ps and q < d
                        and 1 <= dt - rec.loc[(f, q), "date"] <= WIN and rec.loc[(f, q), "dong"] == dg
                        and (f, q) in ECd.index and (f, q) not in fb]
                mins[dg] = min(vals) if vals else np.nan
            if np.isnan(mins["A"]) and np.isnan(mins["B"]):
                continue
            mA = mins["A"] if not np.isnan(mins["A"]) else mins["B"]
            mB = mins["B"] if not np.isnan(mins["B"]) else mins["A"]
            p = O.loc[idx, "pB"].values
            xs[np.asarray(idx)] = p * mB + (1 - p) * mA
    O["x"] = xs
    grp = O.groupby(["validator", "validation_fold", "farm", "day"])
    D = None
    for s in SEEDS:
        O["cm_%d" % s] = grp["r3s_%d" % s].transform(lambda z: z.expanding().mean())
        O["flag_%d" % s] = (O["cm_%d" % s] >= THR) & O.x.notna()
    D = O[O.validator == "DIAG10"]
    for s in SEEDS:
        corr = np.zeros(len(O))
        for (v, k), G in O.groupby(["validator", "validation_fold"]):
            vd = {(f, d + j) for f, d in set(zip(G.farm, G.day)) for j in (-1, 0, 1)}
            T = D[D["flag_%d" % s] & np.array([(f, d) not in vd for f, d in zip(D.farm, D.day)])]
            if len(T) < 50:
                continue
            m = Ridge(alpha=1.0).fit(T[["x", "cm_%d" % s]].values, (T.sub_ec - T["r3s_%d" % s]).values)
            fl = G["flag_%d" % s].values
            if fl.any():
                corr[G.index[fl]] = np.clip(m.predict(G.loc[fl, ["x", "cm_%d" % s]].values), -CLIP, CLIP)
            if v == "DIAG10" and k == 0:
                print("seed %d fold0 coef x %.3f cm %.3f intercept %.3f (fit rows %d)" % (s, m.coef_[0], m.coef_[1], m.intercept_, len(T)))
        O["hg_%d" % s] = np.maximum(O["r3s_%d" % s] + corr, 0)
        O["corr_%d" % s] = corr
    O.to_csv(os.path.join(env.LOCAL, "ec3_HG1_all.csv"), index=False)
    print("flagged share by validator (seed 7):", O.groupby("validator").flag_7.mean().round(3).to_dict())
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261004)
    allb = True
    print("\npooled RMSE R3S -> R3S + high-flag same-dong level correction")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["hg_%d" % s] - G.sub_ec)
            if v != "EL1":
                allb &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["hg_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    E = O[O.validator == "EL1"]
    elok = all(r(E["hg_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    L = D.day >= 179
    hi = D.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1
    print("  DIAG10 late R3S %.4f -> HG1 %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["hg_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 high-EC days (rows %d): label %.3f  R3S %.3f  HG1 %.3f" % (hi.sum(), D.sub_ec[hi].mean(), D.r3s_7[hi].mean(), D.hg_7[hi].mean()))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nHG1 decision:", "PASS" if allb and all(p < 0.025 for p in ps) and elok else "FAIL",
          "(rule %s, EL1 %s)" % (allb and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
