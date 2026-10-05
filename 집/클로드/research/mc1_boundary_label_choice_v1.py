# -*- coding: utf-8 -*-
"""MC1 (diagnostic, fixed before running; 2026-10-05 집 클로드).
Finding MC0: EC is continuous across midnight within a source (correct links: |EC(0h) - EC_prev(23h)|
median .005; hour-to-hour .003).  The candidates for 'same source on the adjacent calendar date' are
usually two records (two 동) with very different EC, and the model separates high / low well.
Rule (hour-causal for the query; training data referenced in both directions, 6.309):
  calendar: SG2 reference calendar (training records only) and SG2 query date at hour h;
  P-candidates: labelled reference records with cal in [cq - 1.5, cq - 0.5]; boundary value = their
            hour-23 EC; N-candidates: cal in [cq + 0.5, cq + 1.5]; boundary value = their hour-0 EC;
  choice: in each set, the candidate whose DAY-MEAN label is closest to pm_h (mean of the model's
            predictions for hours 0..h of the query record);
  level:  both chosen -> linear interpolation over the 24 hours between P(23h) and N(0h);
          one chosen -> that value; none -> model.
  MC1 = model row prediction replaced by that level (no blending), pass-2 rows.
Also an ORACLE choice (candidate nearest to the TRUE day mean, upper bound) and the model alone.
Data: DIAG10 R3S seed mean (hk0_rows_v1.csv) with reference = labelled records outside the fold.
Reported: pass-2 / normal / high row RMSE; share of days where the chosen P candidate's boundary is
within .05 of the true 0h EC.  Clue (fixed): MC1 beats the model on pass-2 rows AND normal days."""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
R, WV, hrs, SIG = sg2.prepare_structure()
roles = R.set_index(["farm", "day"]).role
alld = {f: sorted(R[R.farm == f].day) for f in ("F13", "F47")}
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
E = Y.set_index(["farm", "day", "hour"]).sub_ec; ec = Y.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"))
F = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"), usecols=["farm", "day", "validator", "validation_fold"])
F = F[F.validator == "DIAG10"].drop_duplicates(["farm", "day"])
O = O[O.day >= 179].sort_values(["validation_fold", "farm", "day", "hour"]).reset_index(drop=True)
O["pm"] = O.groupby(["farm", "day"]).p.transform(lambda z: z.expanding().mean())
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
O["mc1"] = np.nan; O["orc"] = np.nan; O["pok"] = np.nan
for k, G in O.groupby("validation_fold"):
    vd = set(zip(F[F.validation_fold == k].farm, F[F.validation_fold == k].day))
    ref = {x for x in labset if x not in vd}; cal = sg2.ref_calendar(R, WV, ref); cache = {}

    def twin_date(f, d, h):
        Ee = [e for e in alld[f] if (f, e) in ref]; cols = np.asarray(hrs <= h)
        A = WV.loc[[(f, e) for e in Ee]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1)); ok = dist <= .05
        return float(np.mean([cal[(f, e)] for e, o in zip(Ee, ok) if o])) if ok.any() else None

    def full_date(f, d):
        if (f, d) in ref: return cal[(f, d)]
        if (f, d) in cache: return cache[(f, d)]
        t = twin_date(f, d, 23)
        if t is None:
            i = alld[f].index(d); t = (full_date(f, alld[f][i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        cache[(f, d)] = t; return t
    for (f, d), idx in G.groupby(["farm", "day"]).groups.items():
        cand = [e for e in alld[f] if (f, e) in ref]
        cc = np.array([cal[(f, e)] for e in cand]); dmv = np.array([ec[(f, e)] for e in cand])
        i = alld[f].index(d); y0 = E.get((f, d, 0), np.nan); ytrue = ec[(f, d)]
        for ii in idx:
            h = int(O.loc[ii, "hour"]); pm = O.loc[ii, "pm"]
            cq = twin_date(f, d, h) if h >= 5 else None
            if cq is None: cq = full_date(f, alld[f][i - 1]) + 0.1
            out = {}
            for tag, target in (("mc1", pm), ("orc", ytrue)):
                vals = []
                for lo, hi_, hb in ((-1.5, -.5, 23), (.5, 1.5, 0)):
                    m = (cc - cq >= lo) & (cc - cq <= hi_)
                    if m.any():
                        j = np.argmin(np.abs(dmv[m] - target)); e = np.array(cand)[m][j]
                        vals.append(E.get((f, e, hb), np.nan))
                    else:
                        vals.append(np.nan)
                a, b = vals
                if np.isfinite(a) and np.isfinite(b): out[tag] = a + (b - a) * (h + 1) / 25
                elif np.isfinite(a): out[tag] = a
                elif np.isfinite(b): out[tag] = b
                if tag == "mc1" and h == 23: O.loc[idx, "pok"] = float(abs(a - y0) <= .05) if np.isfinite(a) else np.nan
            O.loc[ii, "mc1"] = out.get("mc1", np.nan); O.loc[ii, "orc"] = out.get("orc", np.nan)
for c in ("mc1", "orc"):
    O[c + "_f"] = O[c].fillna(O.p)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
print("pass-2 rows %d; MC1 level available %.0f%% of rows; P-choice within .05 of true 0h on %.0f%% of days with a P candidate" % (
    len(O), 100 * O.mc1.notna().mean(), 100 * O.groupby(["farm", "day"]).pok.first().mean()))
for nm, m in (("pass-2", O.dm.notna()), ("normal", O.dm < 1), ("high", O.dm >= 1)):
    g = O[m]
    print("  %-7s model %.4f | MC1 %.4f | oracle choice %.4f" % (nm, r(g.p - g.sub_ec), r(g.mc1_f - g.sub_ec), r(g.orc_f - g.sub_ec)))
O.to_csv(os.path.join(env.LOCAL, "mc1_rows_v1.csv"), index=False)
print("MC1 clue:", r(O.mc1_f - O.sub_ec) < r(O.p - O.sub_ec) and r((O.mc1_f - O.sub_ec)[O.dm < 1]) < r((O.p - O.sub_ec)[O.dm < 1]))
