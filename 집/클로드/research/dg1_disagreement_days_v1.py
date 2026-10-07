# -*- coding: utf-8 -*-
"""DG1 (diagnostic, 2026-10-07 집 클로드; user "2 가보자").  CV1 (6.425) found evaluation records where the
model and SG2's best reference anchor a1 disagree by more than the SG2 guard (.30), so SG2 is off there
(F47 221/227 1.40-1.66 vs 2.71, F47 223 1.43 vs .47, F13 236 .61 vs 1.45 ...).  Questions on labelled days:
  Q1  on disagreement rows (|a1 - pm_h| > .30), is the label closer to a1 or to the model?  oracle gain of
      picking the right side; fixed rules 'move halfway / fully to a1' and 'to mean(a1, a2)'
  Q2  can evaluation-legal signals tell beforehand which side is right?  signals (hour-causal, per row):
      d1 signature distance of a1, |a1 - a2| anchor agreement, dated (outdoor-twin date found at h), n_cand,
      S_low, pm level, direction (a1 above / below pm)
Data: base = CUR = clip(.8 R3_DP1 + .2 PFN) from the saved LH1/WT1 OOF (seeds 47/1414/6464); anchors recomputed
with the SUBMISSION's sg2post (prepare / ref_calendar / same search), reference = labelled records outside the
validation fold (as SG2's own validation).  Sets: DIAG10 pass-2, P2LOO, EL1 (main); DIAG10 pass-1 (auxiliary,
SG2 normally off there; anchors computed the same way to enlarge n).
Unit of evidence = day (rows of a day are not independent): a day is a disagreement day if >= 6 of its rows
disagree; label side judged on those rows.
Reading rule (fixed before running): a signal is a lead only if, on disagreement DAYS, its AUC for 'a1 closer
than the model' is >= .75 with the same direction on each main set that has >= 8 such days, AND >= .70 on the
auxiliary pass-1 set; otherwise this direction stops (report the side statistics as description).
Run:  PYTHONPATH="" py -3.12 -u dg1_disagreement_days_v1.py
"""
import env  # noqa: F401
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"클로드", "submission14_ec_sg2"))
import sg2post  # noqa: E402

DV = ["in_hum", "act_shade", "act_thermal", "act_fog", "in_temp", "in_co2"]
SGN = np.array([1, 1, 1, 1, -1, -1], float)
SEEDS = (47, 1414, 6464)


def auc(score, lab):
    s, l = np.asarray(score, float), np.asarray(lab, bool); ok = np.isfinite(s); s, l = s[ok], l[ok]
    if l.sum() == 0 or (~l).sum() == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return float((r[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum()))


def anchors(F, S, ec, ref, DM):
    """F: rows farm/day/hour/p (one validation fold).  Returns per-row a1, a2, d1, n, dated, slow, pm."""
    cal = sg2post.ref_calendar(S, ref)
    WV, hrs, SIG, days = S["WV"], S["hrs"], S["SIG"], S["days"]
    ST = {}
    for f in ("F13", "F47"):
        for h in range(24):
            M = DM.xs(h, level="hour").loc[f]; M = M[[(f, d) in ref for d in M.index]]
            ST[(f, h)] = (M.mean().values, M.std().replace(0, 1).values)
    cache = {}

    def twin_date(f, d, h):
        E = [e for e in days[f] if (f, e) in ref]; cols = np.asarray(hrs <= h)
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
            i = days[f].index(d)
            t = (full_date(f, days[f][i - 1]) + (0.0 if S["second"][(f, d)] else 0.1)) if i else 0.0
        cache[(f, d)] = t
        return t

    out = []
    for (f, d), idx in F.groupby(["farm", "day"]).groups.items():
        E = [e for e in days[f] if (f, e) in ref and (f, e) in ec.index]
        calc = np.array([cal[(f, e)] for e in E]); i = days[f].index(d)
        rows = F.loc[idx].sort_values("hour")
        cums = {s: rows["p%d" % s].expanding().mean().values for s in SEEDS}
        for k, (ii, rr) in enumerate(rows.iterrows()):
            h = int(rr.hour)
            cq = twin_date(f, d, h) if h >= 5 else None
            dated = cq is not None
            if cq is None:
                cq = full_date(f, days[f][i - 1]) + 0.1 if i else 0.0
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            a1 = a2 = d1 = np.nan; n = int(m.sum())
            if m.any():
                Em = [e for e, z in zip(E, m) if z]; Sh = SIG[h]
                RS = Sh.loc[[(f, e) for e in days[f] if (f, e) in ref]].astype(float)
                mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
                q = (Sh.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
                C = np.nan_to_num(((Sh.loc[[(f, e) for e in Em]].values.astype(float) - mu) / sd)[:, use])
                dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq); o = np.argsort(dist)
                a1 = float(ec[(f, Em[o[0]])]); d1 = float(dist[o[0]])
                a2 = float(ec[(f, Em[o[1]])]) if len(o) > 1 else np.nan
            mu_, sd_ = ST[(f, h)]
            slow = float(np.nansum(SGN * (DM.loc[(f, d, h)].values - mu_) / sd_))
            row = dict(idx=ii, a1=a1, a2=a2, d1=d1, n_cand=n, dated=dated, slow=slow)
            for s in SEEDS:
                row["pm%d" % s] = cums[s][k]
            out.append(row)
    return pd.DataFrame(out).set_index("idx")


def main():
    x = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); tx = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
    y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
    keep = lambda d: d[d.row_id.str[:3].isin(["F13", "F47"])].copy()
    x, tx, y = keep(x), keep(tx), keep(y)
    yi = sg2post._ident(y); ec = yi.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    X = sg2post._ident(pd.concat([x, tx], ignore_index=True))
    S = sg2post.prepare(pd.concat([x, tx], ignore_index=True))
    XX = X.sort_values(["farm", "day", "hour"]).copy(); gg = XX.groupby(["farm", "day"])
    for v in DV:
        XX["cm_" + v] = gg[v].transform(lambda z: z.expanding().mean())
    DM = XX.set_index(["farm", "day", "hour"])[["cm_" + v for v in DV]]
    O = pd.read_csv(os.path.join(env.LOCAL, "ec3_LH1_all.csv"))
    for s in SEEDS:
        r3 = .6 * O["et_%d" % s] + .3 * O["lgb_%d" % s] + .1 * O["mlp_%d" % s]
        O["p%d" % s] = np.clip(.8 * r3 + .2 * O.pfn, O.lo, O.hi)
    O = O[["row_id", "farm", "day", "hour", "sub_ec", "validator", "validation_fold"] + ["p%d" % s for s in SEEDS]]
    parts = []
    for (v, fo), G in O.groupby(["validator", "validation_fold"]):
        vd = set(zip(G.farm, G.day)); ref = {z for z in labset if z not in vd}
        A = anchors(G, S, ec, ref, DM)
        parts.append(G.join(A))
        print("%s/%d done (%d rows)" % (v, fo, len(G)), flush=True)
    D = pd.concat(parts)
    D.to_csv(os.path.join(env.LOCAL, "dg1_rows_v1.csv"), index=False)
    D["set"] = np.where(D.validator == "DIAG10", np.where(D.day >= 179, "DIAG10-p2", "DIAG10-p1(aux)"), D.validator)
    D["ag"] = (D.a1 - D.a2).abs()
    r = lambda e: float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")
    print("\n" + "=" * 100)
    for st, G in D.groupby("set"):
        print("\n[%s] rows %d days %d" % (st, len(G), G.groupby(["validation_fold", "farm", "day"]).ngroups))
        for s in SEEDS:
            pm = G["pm%d" % s]; p = G["p%d" % s]; y = G.sub_ec
            dis = (G.a1 - pm).abs() > .30
            # day-level: disagreement day if >= 6 disagreeing rows; side on those rows
            H = G.assign(dis=dis, e_model=(p - y) ** 2, e_a1=(p + (G.a1 - pm) - y) ** 2,
                         e_half=(p + .5 * (G.a1 - pm) - y) ** 2,
                         e_mean=(p + (np.where(np.isfinite(G.a2), (G.a1 + G.a2) / 2, G.a1) - pm) - y) ** 2,
                         up=G.a1 > pm)
            H = H[H.dis]
            Dd = H.groupby(["validation_fold", "farm", "day"]).agg(n=("dis", "size"), e_model=("e_model", "sum"), e_a1=("e_a1", "sum"),
                                                                  e_half=("e_half", "sum"), e_mean=("e_mean", "sum"),
                                                                  y=("sub_ec", "mean"), pm=("pm%d" % s, "mean"), a1=("a1", "mean"),
                                                                  d1=("d1", "mean"), ag=("ag", "mean"), dated=("dated", "mean"),
                                                                  n_cand=("n_cand", "mean"), slow=("slow", "mean"), up=("up", "mean"))
            Dd = Dd[Dd.n >= 6]
            if not len(Dd):
                print("  seed %d: no disagreement days" % s); continue
            a1side = (Dd.e_a1 < Dd.e_model).values
            N = Dd.n.sum()
            print("  seed %4d: disagreement rows %d (%.0f%%), days %d | a1 closer on %d/%d days (up %d/%d, down %d/%d) | "
                  "RMSE on those rows: model %.3f, full->a1 %.3f, half %.3f, ->mean(a1,a2) %.3f, oracle side %.3f" % (
                      s, dis.sum(), 100 * dis.mean(), len(Dd), a1side.sum(), len(Dd),
                      (a1side & (Dd.up > .5)).sum(), (Dd.up > .5).sum(), (a1side & (Dd.up <= .5)).sum(), (Dd.up <= .5).sum(),
                      np.sqrt(Dd.e_model.sum() / N), np.sqrt(Dd.e_a1.sum() / N), np.sqrt(Dd.e_half.sum() / N),
                      np.sqrt(Dd.e_mean.sum() / N), np.sqrt(np.minimum(Dd.e_a1, Dd.e_model).sum() / N)))
            if s == SEEDS[0]:
                print("    AUC for 'a1 closer' (day level; >.5 = higher value -> a1 more often right):  " + "  ".join(
                    "%s %.2f" % (c, auc(Dd[c], a1side)) for c in ("d1", "ag", "dated", "n_cand", "slow", "pm", "up")))
                print("    days: " + "; ".join("%s%d y%.2f pm%.2f a1%.2f%s" % (f[1:], d, rr.y, rr.pm, rr.a1, "*" if a else "")
                                               for (fo, f, d), rr, a in zip(Dd.index, Dd.itertuples(), a1side)))
    # evaluation records' signal values for comparison
    T = pd.read_csv(os.path.join(env.LOCAL, "cv1_test_rows_v1.csv"))
    T["dis"] = (T.a1 - T.pm).abs() > .30
    E = T[T.dis].groupby(["farm", "day"]).agg(n=("dis", "size"), pm=("pm", "mean"), a1=("a1", "mean"), a2=("a2", "mean"),
                                               slow=("slow", "mean")).query("n >= 6")
    print("\nevaluation disagreement records (>= 6 rows): %d\n%s" % (len(E), E.round(3).to_string()))


if __name__ == "__main__":
    main()
