# -*- coding: utf-8 -*-
"""EA1 (2026-10-11 집 클로드; rule review: replace W2's 'deleted record' branch, which was designed after seeing the evaluation
composition, with the rule-clean linker v5).  EC anchor from linker v5: for a record whose same-house previous date is found
with confidence (1 step, previous record labelled, p >= .8, linker v5 CV with the record's 10-record chunk hidden and both
neighbours deleted), move the prediction to the previous date's 23 h EC exactly like W2 (alpha 1):
    pred_h = p_h + (A - pm_h),   pm_h = running mean of p over hours 0..h,   A = previous record's 23 h EC (training label)
Fixed before running (alpha 1 = the W2 form, threshold .8 = linker standard; no tuning).
Base p = 9th-round-like EC OOF (RV1 file, tag CUR, column sg = 0.8 R3 + 0.2 PFN + SG2), seeds 47 / 1414 / 6464.
Judgement on pass-2 rows: PASS iff all 3 seeds better on DIAG10 AND P2LOO AND EL1 AND record-level (anchored pass-2 records,
seed-mean squared-error change) bootstrap P(worse) < .025.  Normal / high days and the anchored-record list reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ea1_ec_linker_anchor_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy.stats import binomtest
LK = os.path.join(env.LOCAL, "linker_v5", "cv_all400_choice_v5.csv")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


def main():
    L = pd.read_csv(LK); L = L[L.prev_is_labelled.astype(bool) & (L.p >= .8) & (L.steps_to_anchor == 1) & L.anchor_ec23.notna()]
    AN = dict(zip(L.rec, L.anchor_ec23)); print("confident 1-step EC anchors on labelled records: %d" % len(AN))
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy()
    O["rec"] = O.row_id.str[:7]; O = O.sort_values(["validator", "fold", "seed", "rec", "hour"])
    cm = O.groupby(["validator", "fold", "seed", "rec"]).sg.transform(lambda s: s.expanding().mean())
    a = O.rec.map(AN); O["ea"] = np.where(a.notna(), O.sg + (a - cm), O.sg)
    O["dm"] = O.groupby(["validator", "fold", "seed", "rec"]).sub_ec.transform("mean")
    rng = np.random.default_rng(20261128); ok = True
    for v in ("DIAG10", "P2LOO", "EL1"):
        X = O[(O.validator == v) & (O.day >= 179)]
        cells = [rm(Z.ea - Z.sub_ec) / rm(Z.sg - Z.sub_ec) - 1 for _, Z in X.groupby("seed")]; ok &= all(c < 0 for c in cells)
        per = ((X.ea - X.sub_ec) ** 2 - (X.sg - X.sub_ec) ** 2).groupby([X.seed, X.rec]).sum().groupby(level=1).mean(); per = per[per.abs() > 1e-12]
        nm, hi = X.dm < 1, X.dm >= 1
        line = "  %-6s pass-2 cells %s | normal %.4f -> %.4f | high %.4f -> %.4f | anchored records %d, improved %d" % (
            v, " ".join("%+.2f%%" % (100 * c) for c in cells), rm((X.sg - X.sub_ec)[nm]), rm((X.ea - X.sub_ec)[nm]), rm((X.sg - X.sub_ec)[hi]), rm((X.ea - X.sub_ec)[hi]), len(per), int((per < 0).sum()))
        if len(per):
            idx = rng.integers(0, len(per), (20000, len(per))); pw = float((per.values[idx].sum(1) >= 0).mean()); line += " | record P(worse) %.4f" % pw
            if v == "DIAG10":
                ok &= pw < .025; line += ", sign p %.3f" % binomtest(int((per < 0).sum()), len(per), .5, alternative="greater").pvalue
        print(line)
        if v == "DIAG10":
            print("    per-record change (DIAG10):", per.sort_values().round(3).to_dict())
    print("VERDICT EA1: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
