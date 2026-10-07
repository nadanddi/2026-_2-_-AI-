# -*- coding: utf-8 -*-
"""PS1b (premise, 2026-10-07 집 클로드).  PS1 hid only the fold's own records (one farm ID); the evaluation
blocks hide the SAME DATES in both IDs (sources cross IDs, 6.407), so PS1's coverage can be optimistic.
Strict hidden set: fold records + every record sharing an outdoor date group with a fold record (both IDs,
full-day outdoor twin <= .05 as FP1) + record numbers +-1 of all of those.  Also report the evaluation records'
own structure: per evaluation record, does its date group contain a LABELLED record (C1/C2-like) and does the
previous calendar date group (by SG2-style calendar of labelled records) contain labelled records.
Same oracle level and comparison as PS1.  Reading rule (fixed): continue only if on EL1-strict coverage(k<=2)
>= 30 % and oracle gain <= -20 %.
Run:  PYTHONPATH="" py -3.12 -u ps1b_prior_record_source_upper_bound_strict_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd

SEEDS = (47, 1414, 6464)
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]


def date_groups():
    X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)).assign(t=i) for i, f in enumerate(("train_X.csv", "test_X.csv"))])
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    isT = X.groupby(["farm", "day"]).t.first()
    WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W); WZ = (WV - WV[isT == 0].mean()) / WV[isT == 0].std()
    keys = list(WZ.index); A = WZ.values; Dm = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
    par = list(range(len(keys)))

    def fd(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(len(keys)):
        for j in np.where(Dm[i] <= .05)[0]:
            par[fd(i)] = fd(j)
    return pd.Series([fd(i) for i in range(len(keys))], index=pd.MultiIndex.from_tuples(keys)), isT


def main():
    grp, isT = date_groups()
    members = grp.groupby(grp).groups
    L = pd.read_csv(os.path.join(env.LOCAL, "ch2_label_links_v1.csv")); L = L[L.ec_jump.abs() < .05]
    pred = {(f, int(b)): (f, int(a)) for f, a, b in zip(L.farm, L.a, L.b)}
    Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
    Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
    E23 = Y[Y.hour == 23].set_index(["farm", "day"]).sub_ec
    O = pd.read_csv(os.path.join(env.LOCAL, "ec3_SG3_all.csv"))
    O["p"] = O[["core_%d" % s for s in SEEDS]].mean(axis=1); O["sg"] = O[["sg2_%d" % s for s in SEEDS]].mean(axis=1)
    O = O[O.day >= 179].sort_values(["validator", "validation_fold", "farm", "day", "hour"]).copy()
    key = ["validator", "validation_fold", "farm", "day"]
    O["p0"] = O.groupby(key).p.transform("first"); O["dm"] = O.groupby(key).sub_ec.transform("mean")
    r = lambda e: float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")
    out = []
    for (v, fo), G in O.groupby(["validator", "validation_fold"]):
        vd = set(zip(G.farm, G.day))
        same_date = set().union(*[set(members[grp[x]]) for x in vd])
        hidden = {(f, d + j) for f, d in same_date for j in (-1, 0, 1)}
        test = {x for x in isT.index if isT[x] == 1}
        hidden |= test                                    # evaluation records are never labelled
        for (f, d) in vd:
            k, a = 0, (f, d)
            while True:
                a = pred.get(a); k += 1
                if a is None:
                    k = None; break
                if a not in hidden:
                    break
            out.append(dict(validator=v, validation_fold=fo, farm=f, day=d, k=k, lvl=(E23.get(a) if k else np.nan),
                            n_same_date=len(same_date)))
    K = pd.DataFrame(out)
    O = O.merge(K, on=key, how="left")
    O["orc"] = np.where(O.lvl.notna(), O.lvl + (O.p - O.p0), O.sg)
    for v in ("EL1", "P2LOO", "DIAG10"):
        G = O[O.validator == v]; D = K[K.validator == v]
        print("\n[%s strict] hidden pass-2 records %d (hidden same-date records per fold median %d) | k=1 %d, k=2 %d, k=3 %d, k>=4 %d, none %d" % (
            v, len(D), int(D.n_same_date.median()), (D.k == 1).sum(), (D.k == 2).sum(), (D.k == 3).sum(), (D.k >= 4).sum(), D.k.isna().sum()))
        for nm, m in (("k=1", G.k == 1), ("k=2", G.k == 2), ("k<=2", G.k <= 2)):
            g = G[m]
            if not len(g):
                continue
            nn = g.dm < 1
            print("   %-5s days %2d: SG2 %.3f -> oracle-source level %.3f (%+.0f%%) | normal %.3f -> %.3f | high %.3f -> %.3f" % (
                nm, g.groupby(["farm", "day"]).ngroups, r(g.sg - g.sub_ec), r(g.orc - g.sub_ec), 100 * (r(g.orc - g.sub_ec) / r(g.sg - g.sub_ec) - 1),
                r((g.sg - g.sub_ec)[nn]), r((g.orc - g.sub_ec)[nn]), r((g.sg - g.sub_ec)[~nn]), r((g.orc - g.sub_ec)[~nn])))
        print("   all hidden rows: SG2 %.4f -> oracle on k<=2 only %.4f" % (r(G.sg - G.sub_ec), r(np.where(G.k <= 2, G.orc, G.sg) - G.sub_ec)))
    # evaluation records: labelled records in own date group / any labelled record with a labelled predecessor-date candidate
    T = [x for x in isT.index if isT[x] == 1]
    own = sum(any(isT[m] == 0 for m in members[grp[x]]) for x in T)
    print("\nevaluation records %d: date group contains a labelled record %d (the rest are evaluation-only dates)" % (len(T), own))
    D = K[K.validator == "EL1"]; G = O[(O.validator == "EL1") & (O.k <= 2)]
    cov = float((D.k <= 2).mean()); gain = r(G.orc - G.sub_ec) / r(G.sg - G.sub_ec) - 1 if len(G) else 0
    print("PS1b reading:", "continue to identification" if cov >= .3 and gain <= -.2 else "stop", "(EL1-strict coverage %.2f, gain %+.0f%%)" % (cov, 100 * gain))
    K.to_csv(os.path.join(env.LOCAL, "ps1b_ancestors_v1.csv"), index=False)


if __name__ == "__main__":
    main()
