# -*- coding: utf-8 -*-
"""HK0 (diagnostic, fixed before running; 2026-10-05 집 클로드).
High-EC size again, now that ALL training labels may be referenced (6.309) and SG2's
hour-causal signature kNN passed (6.310).  SG2's guard (|a1 - pm_h| <= .30) blocks the
large upward moves high days need (high days barely changed).  Earlier the size of a
high day was tied to the same 동's nearby EC (HC5 +.41) but labels were too sparse; the
kNN may now reach same-source high neighbours.
Gate idea: when the model already says HIGH (detection AUC .989) and the anchor is also
high, a wrong-동 pick is less likely, so allow the correction without the .30 cap.
Variants (k = 2):
  HGA  rows with pm_h >= 0.9 and a1 >= 1.0: pred = p + 0.5 (a1 - pm_h); else SG2 rule.
  HGB  as HGA but needs the best TWO anchors both >= 1.0, level = their mean.
Base for comparison = SG2 rule (guard .30) on the same rows.
Data: DIAG10 R3S seed-mean OOF (local/ec2_DC5_oof.csv), ALL days (pass 1 too, to have
the 31 high days); reference = labelled records outside the fold; SG2 functions with
the pass-2-only restriction lifted for this diagnostic.
Also descriptive: Spearman(a1 - pm_23, y - pm_23) on high days (does the anchor know size?).
Clue (fixed): a variant lowers high-day RMSE (label day mean >= 1) vs SG2 AND row RMSE
on all days and on pass-2 days, with normal-day RMSE not worse by > 1 %, and farm x
5-day cluster bootstrap P(worse) on all rows < .0125."""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)


def anchors(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal):
    """Per row: a1, a2 (best two anchors), pm_h.  SG2 logic, all days."""
    out = pd.DataFrame(index=frame.index, columns=["a1", "a2", "pm"], dtype=float)
    roles = R.set_index(["farm", "day"]).role
    cache = {}

    def twin_date(f, d, h):
        E = [e for e in R[R.farm == f].day if (f, e) in ref]
        cols = np.asarray(hrs <= h)
        A = WV.loc[[(f, e) for e in E]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1)); ok = dist <= .05
        return float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o])) if ok.any() else None

    def full_date(f, d):
        if (f, d) in ref:
            return cal[(f, d)]
        if (f, d) in cache:
            return cache[(f, d)]
        t = twin_date(f, d, 23)
        if t is None:
            days = sorted(R[R.farm == f].day); i = days.index(d)
            t = (full_date(f, days[i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        cache[(f, d)] = t
        return t

    for (f, d), idx in frame.groupby(["farm", "day"]).groups.items():
        G = R[R.farm == f]
        G = G[[(f, e) in ref and (f, e) in ec.index and (f, e) not in lock for e in G.day]]
        calc = np.array([cal[(f, e)] for e in G.day])
        days = sorted(R[R.farm == f].day); i = days.index(d)
        rows = frame.loc[idx].sort_values("hour")
        cum = rows[pcol].expanding().mean().values
        for k, (ii, rr) in enumerate(rows.iterrows()):
            h = int(rr.hour)
            out.loc[ii, "pm"] = cum[k]
            cq = twin_date(f, d, h) if h >= 5 else None
            if cq is None:
                cq = (full_date(f, days[i - 1]) + 0.1) if i else 0.0
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            if not m.any():
                continue
            Gm = G[m]; S = SIG[h]
            RS = S.loc[[(f, e) for e in R[R.farm == f].day if (f, e) in ref]].astype(float)
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            q = (S.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
            C = np.nan_to_num(((S.loc[list(zip(Gm.farm, Gm.day))].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            o = np.argsort(dist)
            out.loc[ii, "a1"] = ec[(f, Gm.iloc[o[0]].day)]
            if len(o) > 1:
                out.loc[ii, "a2"] = ec[(f, Gm.iloc[o[1]].day)]
    return out


def main():
    raw, full, lab, lock, signatures, fds = sg2.p3.prepare()
    R, WV, hrs, SIG = sg2.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].reset_index(drop=True)
    O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
    parts = []
    for k, G in O.groupby("validation_fold"):
        vd = set(zip(G.farm, G.day)); ref = {x for x in labset if x not in vd}
        cal = sg2.ref_calendar(R, WV, ref)
        parts.append(anchors(G, "p", vd, lockd, ec, R, WV, hrs, SIG, ref, cal))
        print("fold %s done" % k, flush=True)
    A = pd.concat(parts); O = O.join(A)
    O.to_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"), index=False)
    d = O.a1 - O.pm
    sg = np.where(O.a1.notna() & (d.abs() <= .30), O.p + .5 * d, O.p)
    gA = O.a1.notna() & (O.pm >= .9) & (O.a1 >= 1.0)
    hga = np.where(gA, O.p + .5 * d, sg)
    gB = gA & O.a2.notna() & (O.a2 >= 1.0)
    hgb = np.where(gB, O.p + .5 * ((O.a1 + O.a2) / 2 - O.pm), sg)
    O["sg"], O["hga"], O["hgb"] = sg, hga, hgb
    O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    print("\nrows gated: HGA %.1f%%, HGB %.1f%% (of all rows); on high days %.0f%% / %.0f%%" % (
        100 * gA.mean(), 100 * gB.mean(), 100 * gA[O.dm >= 1].mean(), 100 * gB[O.dm >= 1].mean()))
    D23 = O[O.hour == 23]; H = D23[D23.dm >= 1].dropna(subset=["a1"])
    print("high days with anchor at 23h: %d of %d; Spearman(a1 - pm, y - pm) %.2f; anchor mean %.3f vs label %.3f vs pm %.3f" % (
        len(H), (D23.dm >= 1).sum(), spearmanr(H.a1 - H.pm, H.dm - H.pm).correlation, H.a1.mean(), H.dm.mean(), H.pm.mean()))
    print("\n%-14s %6s %8s %8s %8s %8s" % ("segment", "rows", "R3S", "SG2", "HGA", "HGB"))
    for nm, m in (("all", O.dm.notna()), ("pass-2", O.day >= 179), ("high days", O.dm >= 1), ("normal days", O.dm < 1),
                  ("pass-2 high", (O.day >= 179) & (O.dm >= 1)), ("pass-2 normal", (O.day >= 179) & (O.dm < 1))):
        G = O[m]
        print("%-14s %6d %8.4f %8.4f %8.4f %8.4f" % (nm, len(G), r(G.p - G.sub_ec), r(G.sg - G.sub_ec), r(G.hga - G.sub_ec), r(G.hgb - G.sub_ec)))
    O["cl"] = O.farm + "_" + (O.day // 5).astype(str)
    rng = np.random.default_rng(20261005); clue = False
    for c in ("hga", "hgb"):
        dd = ((O[c] - O.sub_ec) ** 2 - (O.sg - O.sub_ec) ** 2).groupby(O.cl).agg(["sum", "count"])
        sm, n = dd["sum"].values, dd["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
        seg = lambda m: (r(O[c][m] - O.sub_ec[m]), r(O.sg[m] - O.sub_ec[m]))
        hi, al, p2, no = seg(O.dm >= 1), seg(O.dm.notna()), seg(O.day >= 179), seg(O.dm < 1)
        ok = hi[0] < hi[1] and al[0] < al[1] and p2[0] < p2[1] and no[0] <= 1.01 * no[1] and p < .0125
        clue |= ok
        print("%s vs SG2: P(worse) %.4f, high %+.1f%%, all %+.1f%%, pass-2 %+.1f%%, normal %+.1f%% -> %s" % (
            c.upper(), p, 100 * (hi[0] / hi[1] - 1), 100 * (al[0] / al[1] - 1), 100 * (p2[0] / p2[1] - 1), 100 * (no[0] / no[1] - 1), "clue" if ok else "no"))
    print("\nHK0 clue:", clue)


if __name__ == "__main__":
    main()
