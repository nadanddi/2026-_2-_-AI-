# -*- coding: utf-8 -*-
"""CV1 (diagnostic, 2026-10-07 집 클로드; user: "1번 가보자" = widen precise-but-narrow tools by union).
Step 1 = COVERAGE ONLY, counted on the 60 evaluation records with evaluation-legal inputs (same farm, hours
0..h, earlier records; reference = the 400 labelled training records, as in submission_14).  No labels of
evaluation rows exist; nothing is fitted.  Tools:
  HG4  pm_h >= .9 and best two SG2 anchors >= 1.0 and S_low <= -1.0 (6.341), pm = submission_14 pre-SG2
       prediction (sg2_details.pred_submission13_config), computed with the submission's sg2post.prepare.
  GX3  lab Claude's grid-exclusion decisions on the evaluation records (연구실/클로드/results/ec_gx3_test_v1.csv,
       st == '결정'), read as given.
  DATE lab Claude DT0 classes (ec_dt0_date_class_v1.csv): C1/C2 = date fixed by a pass-1 outdoor twin
       (SG2 already uses these), C3/C4 = undated (6.404/6.405, the larger error share).
Also: the same HG4 trigger counted on validation pass-2 days (HG4 checkpoints, rows where sg != sgb).
Reading rule (fixed before running): the union HG4 or GX3 is worth an effect test only if it acts on
>= 10 of the 60 evaluation records, or on >= 5 undated (C3/C4) records; otherwise stop this direction.
Run:  PYTHONPATH="" py -3.12 -u cv1_precise_signal_coverage_v1.py
"""
import env  # noqa: F401
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"클로드", "submission14_ec_sg2"))
import sg2post  # noqa: E402

DV = ["in_hum", "act_shade", "act_thermal", "act_fog", "in_temp", "in_co2"]
SGN = np.array([1, 1, 1, 1, -1, -1], float)
LAB = os.path.join(env.ROOT, u"연구실", u"클로드", "results")


def main():
    x = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); tx = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
    y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
    keep = lambda d: d[d.row_id.str[:3].isin(["F13", "F47"])].copy()
    x, tx, y = keep(x), keep(tx), keep(y)
    yi = sg2post._ident(y); ec = yi.groupby(["farm", "day"]).sub_ec.mean(); ref = set(ec.index)
    X = sg2post._ident(pd.concat([x, tx], ignore_index=True))
    S = sg2post.prepare(pd.concat([x, tx], ignore_index=True)); cal = sg2post.ref_calendar(S, ref)
    WV, hrs, SIG, days = S["WV"], S["hrs"], S["SIG"], S["days"]
    # S_low: cumulative 0..h means, standardised by reference records per (farm, h)
    XX = X.sort_values(["farm", "day", "hour"]).copy(); gg = XX.groupby(["farm", "day"])
    for v in DV:
        XX["cm_" + v] = gg[v].transform(lambda z: z.expanding().mean())
    DM = XX.set_index(["farm", "day", "hour"])[["cm_" + v for v in DV]]
    ST = {}
    for f in ("F13", "F47"):
        for h in range(24):
            M = DM.xs(h, level="hour").loc[f]; M = M[[(f, d) in ref for d in M.index]]
            ST[(f, h)] = (M.mean().values, M.std().replace(0, 1).values)
    z = np.load(os.path.join(env.ROOT, u"집", u"클로드", "local", "submission14_ec_sg2_20261005", "run1", "sg2_details.npz"), allow_pickle=True)
    F = sg2post._ident(pd.DataFrame({"row_id": z["row_id"]})); F["p"] = z["pred_submission13_config"]; F["fin"] = z["pred_final"]
    F = F[F.farm.isin(["F13", "F47"])].reset_index(drop=True)
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

    rows_out = []
    for (f, d), idx in F.groupby(["farm", "day"]).groups.items():
        E = [e for e in days[f] if (f, e) in ref and (f, e) in ec.index]
        calc = np.array([cal[(f, e)] for e in E]); i = days[f].index(d)
        rows = F.loc[idx].sort_values("hour"); cum = rows.p.expanding().mean().values
        for k, (ii, rr) in enumerate(rows.iterrows()):
            h = int(rr.hour)
            cq = twin_date(f, d, h) if h >= 5 else None
            dated = cq is not None
            if cq is None:
                cq = full_date(f, days[f][i - 1]) + 0.1
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            a1 = a2 = np.nan
            if m.any():
                Em = [e for e, zz in zip(E, m) if zz]; Sh = SIG[h]
                RS = Sh.loc[[(f, e) for e in days[f] if (f, e) in ref]].astype(float)
                mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
                q = (Sh.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
                C = np.nan_to_num(((Sh.loc[[(f, e) for e in Em]].values.astype(float) - mu) / sd)[:, use])
                o = np.argsort(np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq))
                a1 = float(ec[(f, Em[o[0]])]); a2 = float(ec[(f, Em[o[1]])]) if len(o) > 1 else np.nan
            mu_, sd_ = ST[(f, h)]
            slow = float(np.nansum(SGN * (DM.loc[(f, d, h)].values - mu_) / sd_))
            pm = cum[k]
            hg = bool(pm >= .9 and a1 >= 1.0 and np.isfinite(a2) and a2 >= 1.0 and slow <= -1.0)
            rows_out.append(dict(farm=f, day=d, hour=h, p=rr.p, fin=rr.fin, pm=pm, a1=a1, a2=a2, slow=slow,
                                 dated_h=dated, sg2_on=bool(np.isfinite(a1) and abs(a1 - pm) <= .30), hg4=hg,
                                 hg4_pred=(rr.p + .5 * (min((a1 + a2) / 2, 1.8) - pm)) if hg else np.nan))
    T = pd.DataFrame(rows_out)
    T.to_csv(os.path.join(env.LOCAL, "cv1_test_rows_v1.csv"), index=False)
    D = T.groupby(["farm", "day"]).agg(pred=("fin", "mean"), pm23=("pm", "last"), sg2_rows=("sg2_on", "sum"),
                                       hg4_rows=("hg4", "sum"), dated_h=("dated_h", "mean"), a1_23=("a1", "last")).reset_index()
    gx = pd.read_csv(os.path.join(LAB, "ec_gx3_test_v1.csv"), encoding="utf-8-sig")
    dt = pd.read_csv(os.path.join(LAB, "ec_dt0_date_class_v1.csv"), encoding="utf-8-sig")
    D = D.merge(gx[["farm", "day", "st", "v"]], on=["farm", "day"], how="left").merge(dt[["farm", "day", "cls"]], on=["farm", "day"], how="left")
    D["gx3"] = D.st.eq("결정"); D["hg4"] = D.hg4_rows > 0; D["undated"] = D.cls.isin(["C3", "C4"])
    D["union"] = D.gx3 | D.hg4
    pd.set_option("display.width", 220)
    print("Evaluation records: %d (F13 %d, F47 %d); classes %s" % (len(D), (D.farm == "F13").sum(), (D.farm == "F47").sum(),
          D.cls.value_counts().to_dict()))
    print("\nper-record table (records where any tool acts or undated):")
    print(D[D.union | D.undated][["farm", "day", "cls", "pred", "pm23", "a1_23", "sg2_rows", "hg4_rows", "gx3", "v"]].round(3).to_string(index=False))
    for nm, c in (("HG4", D.hg4), ("GX3", D.gx3), ("HG4 or GX3", D.union), ("HG4 and GX3", D.gx3 & D.hg4)):
        print("%-12s acts on %2d / 60 records; undated (C3/C4) %d / %d" % (nm, c.sum(), (c & D.undated).sum(), D.undated.sum()))
    print("SG2 active (any row) on %d / 60 records; HG4 rows %d / 1440" % ((D.sg2_rows > 0).sum(), int(T.hg4.sum())))
    hgd = D[D.hg4]
    if len(hgd):
        print("HG4 records: mean change of day prediction %.3f (rows triggered only)" % np.nanmean(T.hg4_pred - T.fin))
    # validation pass-2 days triggered by HG4 (saved checkpoints)
    V = pd.read_csv(os.path.join(env.LOCAL, "ec3_HG4_all.csv")); V = V[V.day >= 179]
    for s in (53, 1515, 7575):
        V["t%d" % s] = (V["sg_%d" % s] - V["sgb_%d" % s]).abs() > 1e-12
    print("\nvalidation pass-2 days with any HG4 trigger (seed 53/1515/7575):")
    for v, G in V.groupby("validator"):
        g = G.groupby(["validation_fold", "farm", "day"])[["t53", "t1515", "t7575"]].any()
        print("  %-7s days %3d  triggered %s" % (v, len(g), g.sum().to_dict()))
    ok = D.union.sum() >= 10 or (D.union & D.undated).sum() >= 5
    print("\nCV1 reading:", "worth an effect test" if ok else "coverage too small -> stop this direction",
          "(union %d / 60, undated %d)" % (D.union.sum(), (D.union & D.undated).sum()))


if __name__ == "__main__":
    main()
