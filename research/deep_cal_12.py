# -*- coding: utf-8 -*-
"""Diagnose causal twin / pred / cday against the analysis-only calendar and links."""
import env  # noqa
import numpy as np
import pandas as pd
import deep_cal_feats as DF

df = DF.build()
df.to_csv(env.LOCAL + "/deep_cal_feats_days.csv", index=False)
k = pd.read_csv(env.LOCAL + "/deep_cal_9_days.csv")
L = pd.read_csv(env.LOCAL + "/deep_cal_10_links.csv")
S = pd.read_csv(env.LOCAL + "/deep_cal_10_daysum.csv")
m = df.merge(k[["farm", "day", "date", "cal", "is_test"]], on=["farm", "day"])
dd = {(f, d): dt for f, d, dt in zip(k.farm, k.day, k.date)}
m["truth_twin"] = [any(dd.get((f, e)) == dt for e in range(0, d)) for f, d, dt in zip(m.farm, m.day, m.date)]
tw_ok = np.array([(not np.isnan(t)) and dd.get((f, int(t))) == dt for f, t, dt in zip(m.farm, m.twin, m.date)])
print("twin: truth-has %d, found %d, found&correct %d, false-positive %d, missed %d" % (
    m.truth_twin.sum(), (m.has_twin == 1).sum(), tw_ok.sum(), ((m.has_twin == 1) & ~tw_ok).sum(),
    (m.truth_twin & (m.has_twin == 0)).sum()))
it = m.is_test.values
print("  test days: truth-has %d/%d, found-correct %d" % (m.truth_twin[it].sum(), it.sum(), tw_ok[it].sum()))
truth = {(f, b): a for f, a, b in zip(L.farm, L.d_from, L.d_to) if a < b}
m["truth_pred"] = [truth.get((f, d), np.nan) for f, d in zip(m.farm, m.day)]
h = m.dropna(subset=["truth_pred"])
print("pred: days with causal-reachable same-source yesterday (analysis links) %d/%d; causal rule hits %.3f"
      % (len(h), len(m), (h.pred == h.truth_pred).mean()))
cal = {(f, d): c for f, d, c in zip(k.farm, k.day, k.cal)}
m["pred_cal_ok"] = [cal.get((f, p)) == c - 1 if not np.isnan(p) else False for f, p, c in zip(m.farm, m.pred, m.cal)]
print("pred is the calendar-previous date (any source): %.3f overall; test %.3f"
      % (m.pred_cal_ok.mean(), m[m.is_test].pred_cal_ok.mean()))
print("pred gap distribution:", m.pred_gap.value_counts().head(8).to_dict())
for lag in (1, 2):
    ok = [truth.get((f, d)) == d - lag for f, d in zip(m.farm, m.day)]
    print("  d-%d is the same-source yesterday: %.3f of all days (%.3f of days with a link)"
          % (lag, np.mean(ok), np.sum(ok) / len(h)))
S2 = S.set_index(["farm", "day"])


def jump(f, d, e):
    try:
        return S2.loc[(f, d), "sub_ec_0"] - S2.loc[(f, int(e)), "sub_ec_23"]
    except Exception:
        return np.nan


j = np.array([jump(f, d, e) if not np.isnan(e) else np.nan for f, d, e in zip(m.farm, m.day, m.pred)])
print("|EC(d,0)-EC(pred,23)| median %.3f mean %.3f  (n=%d)" % (np.nanmedian(np.abs(j)), np.nanmean(np.abs(j)), np.isfinite(j).sum()))
for lag in (1, 2):
    j = np.array([jump(f, d, d - lag) for f, d in zip(m.farm, m.day)])
    print("|EC(d,0)-EC(d-%d,23)| median %.3f mean %.3f" % (lag, np.nanmedian(np.abs(j)), np.nanmean(np.abs(j))))
for f in ["F13", "F47"]:
    x = m[m.farm == f]
    print(f, "corr(cal, day) %.3f  corr(cal, cday) %.3f" % (x.cal.corr(x.day), x.cal.corr(x.cday)))
    t = x[x.is_test]
    print("   test day  %s\n   test cday %s\n   test cal  %s" % (t.day.tolist(), t.cday.round(0).astype(int).tolist(), t.cal.tolist()))
m.to_csv(env.LOCAL + "/deep_cal_12_diag.csv", index=False)
