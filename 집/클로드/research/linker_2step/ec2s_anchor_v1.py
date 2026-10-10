# -*- coding: utf-8 -*-
"""EC 2-step anchor value test (2026-10-12 집 클로드). Rule fixed BEFORE running:
For a record WITHOUT a confident 1-step anchor (critic mirror CV v7m: p>=.8 & previous labelled) whose direct 2-step linker
pick (mirror CV, hour 0) has p >= .8:   pred_h = p_h + 0.5 * (A - pm_h)
    A = picked 2-dates-earlier training record's 23 h EC, pm_h = running mean of p over hours 0..h.
Base p = 9th-round-like EC OOF (RV1 file, tag CUR, column sg), seeds 47 / 1414 / 6464.
PASS iff on pass-2 rows (day >= 179) all 3 seeds better on DIAG10 AND P2LOO AND EL1 AND DIAG10 record-level bootstrap
P(worse) < .025 (seed-mean squared-error change per anchored record).
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
S2S = r"C:\work\farmai\집\클로드\research\local\linker_2step"
C1 = pd.read_csv(r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\critic\cv_all400_choice_v7m.csv")
one = set(C1[(C1.p >= .8) & C1.prev_is_labelled.astype(bool)].rec)
L = pd.read_csv(S2S + r"\cv_mirror_choice_2step_v1.csv")
L = L[(L.p >= .8) & L.pick.notna() & L.anchor_ec23.notna() & ~L.rec.isin(one)]
AN = dict(zip(L.rec, L.anchor_ec23))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy()
O["rec"] = O.row_id.str[:7]; O = O.sort_values(["validator", "fold", "seed", "rec", "hour"])
p2recs = set(O[O.day >= 179].rec)
LL = L[L.rec.isin(p2recs)]
print("confident 2-step anchors (no 1-step anchor): %d, on pass-2 OOF records: %d" % (len(L), len(LL)))
print(LL[["rec", "pick", "p", "xf", "anchor_ec23", "truth_a2", "hit"]].to_string(index=False))
cm = O.groupby(["validator", "fold", "seed", "rec"]).sg.transform(lambda s: s.expanding().mean())
a = O.rec.map(AN); O["ea"] = np.where(a.notna(), O.sg + 0.5 * (a - cm), O.sg)
rng = np.random.default_rng(20261012); ok = True
for v in ("DIAG10", "P2LOO", "EL1"):
    Xv = O[(O.validator == v) & (O.day >= 179)]
    cells = [rm(Z.ea - Z.sub_ec) / rm(Z.sg - Z.sub_ec) - 1 for _, Z in Xv.groupby("seed")]; ok &= all(c < 0 for c in cells)
    per = ((Xv.ea - Xv.sub_ec) ** 2 - (Xv.sg - Xv.sub_ec) ** 2).groupby([Xv.seed, Xv.rec]).sum().groupby(level=1).mean(); per = per[per.abs() > 1e-12]
    sub = Xv[Xv.rec.isin(per.index)]
    line = "  %-6s pass-2 cells %s | anchored recs %d improved %d | on anchored rows RMSE %.4f -> %.4f" % (
        v, " ".join("%+.2f%%" % (100 * c) for c in cells), len(per), int((per < 0).sum()), rm(sub.sg - sub.sub_ec), rm(sub.ea - sub.sub_ec))
    if len(per):
        idx = rng.integers(0, len(per), (20000, len(per))); pw = float((per.values[idx].sum(1) >= 0).mean()); line += " | P(worse) %.4f" % pw
        if v == "DIAG10": ok &= pw < .025
    else:
        ok = False
    print(line)
    if v == "DIAG10":
        dd = Xv[Xv.rec.isin(per.index)].groupby("rec").apply(lambda Z: pd.Series(dict(rmse_base=rm(Z.sg - Z.sub_ec), rmse_new=rm(Z.ea - Z.sub_ec))))
        print(dd.round(4).to_string())
print("VERDICT EC 2-step anchor (beta .5, p>=.8): %s" % ("PASS" if ok else "FAIL"))
