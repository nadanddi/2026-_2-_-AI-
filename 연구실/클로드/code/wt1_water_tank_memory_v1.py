# -*- coding: utf-8 -*-
"""WT1: does a water-tank thermal memory (CALENDAR-ordered past weather) explain the
G_C2 day offset?  (lab Claude, 2026-10-01)  -- pre-registered DIAGNOSTIC, not a model.

Source of the hypothesis (user, 2026-10-01, from a grower/domain person):
  * substrate temperature may be only weakly tied to greenhouse air; underground is cold
    even when the surface is warm;
  * substrate temperature is tied most to the irrigation WATER temperature;
  * an outdoor water tank warms with sunlight / outdoor temperature, an indoor tank
    follows indoor temperature.
Why this is not a repeat: 6b.36 P1 used the previous RECORD days' outdoor mean, but the
records splice several sources (1.1, 1.12), so record order is not time order.  A tank
(large water mass) integrates several REAL days -> use the weather-only calendar map
(deep_cal_9, built without labels) and look strictly at PAST calendar dates.

Target  e_day = day mean of (G_C2 DIAG10 OOF pred - y), rebuilt from saved members
        (+ means the substrate was colder than predicted).
Proxies (per farm, per calendar date c, EWM with half-life 3 days over dates c-1, c-2, ...;
         today excluded; a date's value = mean over that farm's records on that date)
  W1   outdoor tank: z(EWM past out_temp daily mean) + z(EWM past out_rad daily sum)
  W1r  outdoor tank, sunlight only: EWM past out_rad daily sum
  W2   indoor tank: EWM past in_temp daily mean
  Predicted sign of the partial Spearman rho with e_day: NEGATIVE for all three
  (warmer past water -> substrate warmer than predicted -> e_day lower).
Controls (rank-partialled) the record day's own in_temp mean, out_temp mean, out_rad sum.

PRE-SET RULE (fixed before running): a proxy PASSES only if in BOTH farms the partial rho
  has the predicted sign, |rho| >= 0.20 and p < 0.05/6 (3 proxies x 2 farms).
  A passing proxy goes to a separately pre-registered causal model hypothesis; a failing
  one is recorded as rejected.  Descriptive extras (not judged): the same numbers on the
  day-level label quantity (sub - in_temp) mean, and on the 2nd segment only.
Limitations stated up front: tank location per source is unknown, so a mixture of indoor
  and outdoor tanks dilutes both W1 and W2; W2 averages records of the same date that may
  come from different sources.

Output  logs/wt1_water_tank_memory_v1.log
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "집", "클로드", "research"))
import env  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, t as T  # noqa: E402
import common  # noqa: E402
from harness import load  # noqa: E402
from ls1_label_structure_recovery_v1 import gc2_diag  # noqa: E402

LOG = os.path.join(HERE, "..", "logs", "wt1_water_tank_memory_v1.log")
HL = 3.0
out = open(LOG, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")
    out.flush()


def ewm_past(series, hl=HL):
    """series indexed by consecutive calendar ints; value at c uses c-1, c-2, ... only."""
    idx = np.arange(series.index.min(), series.index.max() + 1)
    s = series.reindex(idx)
    lam = 0.5 ** (1.0 / hl)
    res, num, den = {}, 0.0, 0.0
    for c in idx:
        res[c] = num / den if den > 0 else np.nan
        v = s.loc[c]
        num, den = num * lam, den * lam
        if not np.isnan(v):
            num, den = num + v, den + 1.0
    return pd.Series(res)


def partial_rho(x, y, Z):
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(Z).any(1))
    rx, ry = rankdata(x[m]), rankdata(y[m])
    RZ = np.c_[np.ones(m.sum()), np.apply_along_axis(rankdata, 0, Z[m])]
    ex = rx - RZ @ np.linalg.lstsq(RZ, rx, rcond=None)[0]
    ey = ry - RZ @ np.linalg.lstsq(RZ, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    dof = m.sum() - 2 - Z.shape[1]
    tt = r * np.sqrt(dof / (1 - r ** 2))
    return r, 2 * T.sf(abs(tt), dof), int(m.sum())


def main():
    _, lab, _ = load()
    lab = lab.copy()
    lab["e"] = gc2_diag(lab) - lab.sub_temp
    lab["sub_in"] = lab.sub_temp - lab.in_temp
    Dl = lab.groupby(["farm", "day"]).agg(e_day=("e", "mean"), sub_in=("sub_in", "mean")).reset_index()

    tX, _, sX = common.load_raw()
    a = pd.concat([tX, sX], ignore_index=True)
    a = a[a.farm.isin(["F13", "F47"])]
    R = a.groupby(["farm", "day"]).agg(tin=("in_temp", "mean"), tout=("out_temp", "mean"),
                                       rad=("out_rad", "sum")).reset_index()
    k = pd.read_csv(env.LOCAL + "/deep_cal_9_days.csv")[["farm", "day", "cal"]]
    R = R.merge(k, on=["farm", "day"])
    rows = []
    for f, q in R.groupby("farm"):
        bc = q.groupby("cal")[["tin", "tout", "rad"]].mean()
        ew = pd.DataFrame({"tout_p": ewm_past(bc.tout), "rad_p": ewm_past(bc.rad), "tin_p": ewm_past(bc.tin)})
        q = q.merge(ew, left_on="cal", right_index=True, how="left")
        rows.append(q)
    R = pd.concat(rows)
    D = Dl.merge(R, on=["farm", "day"], how="left")
    for f in ("F13", "F47"):
        m = D.farm == f
        for c in ("tout_p", "rad_p"):
            D.loc[m, c + "_z"] = (D.loc[m, c] - D.loc[m, c].mean()) / D.loc[m, c].std()
    D["W1"] = D.tout_p_z + D.rad_p_z
    D["W1r"] = D.rad_p
    D["W2"] = D.tin_p
    p("days %d (labelled); EWM half-life %.0f calendar days, past dates only; missing proxy %d"
      % (len(D), HL, int(D[["W1", "W1r", "W2"]].isna().any(axis=1).sum())))
    p("check: corr(W1 proxy parts) tout_p vs rad_p %.2f; tout_p vs tin_p %.2f"
      % (D.tout_p.corr(D.rad_p), D.tout_p.corr(D.tin_p)))

    thr = 0.05 / 6
    hyp = [("W1", "outdoor tank: past out_temp + out_rad"), ("W1r", "outdoor tank: past sunlight"),
           ("W2", "indoor tank: past in_temp")]
    Zc = ["tin", "tout", "rad"]
    p("\nJUDGED: partial Spearman with e_day, controls %s; rule: both farms rho<0, |rho|>=0.20, p<%.4f" % (Zc, thr))
    for col, name in hyp:
        res, ok = [], True
        for f in ("F13", "F47"):
            q = D[D.farm == f]
            r, pv, n = partial_rho(q[col].values, q.e_day.values, q[Zc].values)
            res.append("%s rho %+.3f (p %.4f, n %d)" % (f, r, pv, n))
            ok &= (r < 0) and (abs(r) >= 0.20) and (pv < thr)
        p("  %-4s %-40s %s -> %s" % (col, name, " | ".join(res), "PASS" if ok else "fail"))

    p("\nDESCRIPTIVE (not judged)")
    for lbl, ycol, mask in (("e_day, 2nd segment", "e_day", D.day >= 179),
                            ("label sub-in mean, all", "sub_in", D.day > 0)):
        for col, name in hyp:
            res = []
            for f in ("F13", "F47"):
                q = D[(D.farm == f) & mask]
                r, pv, n = partial_rho(q[col].values, q[ycol].values, q[Zc].values)
                res.append("%s %+.3f (p %.3f, n %d)" % (f, r, pv, n))
            p("  %-24s %-4s %s" % (lbl, col, " | ".join(res)))
    G = pd.read_csv(os.path.join(HERE, "..", "..", "..", "집", "클로드", "research", "re17_day_offsets.csv"))
    D = D.merge(G[["farm", "day", "group"]], on=["farm", "day"], how="left")
    p("\n  group medians of proxies (over = substrate colder than predicted):")
    p(D.groupby("group")[["W1", "W1r", "W2", "tin", "tout"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
