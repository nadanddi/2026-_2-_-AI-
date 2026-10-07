# -*- coding: utf-8 -*-
"""PS1 (premise / upper bound, 2026-10-07 집 클로드; user: "첫날 말고 그 이후부터 적용" -> use EARLIER records'
full-day fingerprints to fix the source, and take the level only from LABELLED same-source records).
Question before building any identifier: even with the source chain known perfectly (oracle), how many hidden
pass-2 records have a labelled same-source ancestor within k = 1 / 2 steps, and how much better than SG2 is the
level taken from it?  If this ceiling is small, the identification work is not worth doing.
Truth chains: CH2 reliable label links a -> b (|EC jump| < .05, 6.346).
Hidden sets (validation analogues of the evaluation blocks): EL1 (5 consecutive pass-2 labelled records per fold)
and P2LOO (one record).  Hidden = fold records + record numbers +-1 (as the evaluation blocks have a missing
record on each side).  Walk b -> pred(b) -> ... until a record outside the hidden set: k steps; none if the chain
starts before that.
Level (oracle source, legal level): pred_h = EC_anc(23) + (p_h - p_0)   (p = CUR base, model within-day change);
compared with SG2 on the same rows (saved SG3/SG2 OOF on the CUR base, seeds 47/1414/6464 mean).
Also: same numbers for DIAG10 pass-2 days (TM subset not needed).
Reading rule (fixed before running): continue to the identification step only if, on EL1, records with k <= 2
are >= 30 % of hidden pass-2 records AND the oracle level beats SG2 on those rows by >= 20 % (RMSE); else stop.
Run:  PYTHONPATH="" py -3.12 -u ps1_prior_record_source_upper_bound_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd

SEEDS = (47, 1414, 6464)


def main():
    L = pd.read_csv(os.path.join(env.LOCAL, "ch2_label_links_v1.csv")); L = L[L.ec_jump.abs() < .05]
    pred = {(f, int(b)): (f, int(a)) for f, a, b in zip(L.farm, L.a, L.b)}
    Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
    Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
    E23 = Y[Y.hour == 23].set_index(["farm", "day"]).sub_ec
    O = pd.read_csv(os.path.join(env.LOCAL, "ec3_SG3_all.csv"))
    O["p"] = O[["core_%d" % s for s in SEEDS]].mean(1); O["sg"] = O[["sg2_%d" % s for s in SEEDS]].mean(1)
    O = O[O.day >= 179].sort_values(["validator", "validation_fold", "farm", "day", "hour"]).copy()
    O["p0"] = O.groupby(["validator", "validation_fold", "farm", "day"]).p.transform("first")
    O["dm"] = O.groupby(["validator", "validation_fold", "farm", "day"]).sub_ec.transform("mean")
    r = lambda e: float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")
    out = []
    for (v, fo), G in O.groupby(["validator", "validation_fold"]):
        vd = set(zip(G.farm, G.day))
        hidden = {(f, d + j) for f, d in vd for j in (-1, 0, 1)}
        for (f, d) in vd:
            k, a = 0, (f, d)
            while True:
                a = pred.get(a); k += 1
                if a is None:
                    k = None; break
                if a not in hidden:
                    break
            out.append(dict(validator=v, validation_fold=fo, farm=f, day=d, k=k, anc=a, lvl=(E23.get(a) if k else np.nan)))
    K = pd.DataFrame(out)
    O = O.merge(K, on=["validator", "validation_fold", "farm", "day"], how="left")
    O["orc"] = np.where(O.lvl.notna(), O.lvl + (O.p - O.p0), O.sg)
    for v in ("EL1", "P2LOO", "DIAG10"):
        G = O[O.validator == v]; D = K[K.validator == v]
        if not len(G):
            continue
        n = len(D)
        kc = D.k.value_counts(dropna=False).sort_index()
        print("\n[%s] hidden pass-2 records %d | k=1 %d, k=2 %d, k=3 %d, k>=4 %d, no labelled ancestor %d" % (
            v, n, (D.k == 1).sum(), (D.k == 2).sum(), (D.k == 3).sum(), (D.k >= 4).sum(), D.k.isna().sum()))
        for nm, m in (("k=1", G.k == 1), ("k=2", G.k == 2), ("k<=2", G.k <= 2), ("k=3", G.k == 3)):
            g = G[m]
            if not len(g):
                continue
            nn = g.dm < 1
            print("   %-5s days %2d: SG2 %.3f -> oracle-source level %.3f (%+.0f%%) | normal SG2 %.3f -> %.3f | high SG2 %.3f -> %.3f" % (
                nm, g.groupby(["farm", "day"]).ngroups, r(g.sg - g.sub_ec), r(g.orc - g.sub_ec), 100 * (r(g.orc - g.sub_ec) / r(g.sg - g.sub_ec) - 1),
                r((g.sg - g.sub_ec)[nn]), r((g.orc - g.sub_ec)[nn]), r((g.sg - g.sub_ec)[~nn]), r((g.orc - g.sub_ec)[~nn])))
        print("   all hidden rows: SG2 %.4f -> oracle on k<=2 only %.4f" % (r(G.sg - G.sub_ec), r(np.where(G.k <= 2, G.orc, G.sg) - G.sub_ec)))
    D = K[K.validator == "EL1"]; G = O[(O.validator == "EL1") & (O.k <= 2)]
    cov = float((D.k <= 2).mean()); gain = r(G.orc - G.sub_ec) / r(G.sg - G.sub_ec) - 1 if len(G) else 0
    print("\nPS1 reading:", "continue to identification" if cov >= .3 and gain <= -.2 else "stop", "(EL1 coverage %.2f, gain %+.0f%%)" % (cov, 100 * gain))
    K.to_csv(os.path.join(env.LOCAL, "ps1_ancestors_v1.csv"), index=False)


if __name__ == "__main__":
    main()
