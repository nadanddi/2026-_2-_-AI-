# -*- coding: utf-8 -*-
"""Analysis Q1b: what shape is the within-day temperature error?

anal_q2b_daylevel.py: the day-level offset (58% of the remaining error) is
not explained by source, calendar date, neighbouring days or the day's own
input summaries (leave-date-out R2 about 0), so it looks like a floor for
input-only models.  The other 42% is within-day.  If the within-day error has
a systematic shape it can be fixed:

  amplitude  predicted daily swing narrower/wider than the true swing
             (std of the within-day profile, pred / true)
  timing     predicted profile leads or lags the true one (best
             cross-correlation lag in -3..+3 h)
  fit        correlation of the two within-day profiles

per clean greenhouse-day with all 24 hours scored, from the diagnostic OOF
(local/oof_temp_diag.npz).  Also: does the amplitude ratio depend on how big
the day's indoor-air swing is?

Run:  cd research && PYTHONPATH="" <python> anal_q1b_withinday.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

from harness import load
from cleanw_v6 import weights


def main():
    panel, lab, _ = load()
    z = np.load(env.LOCAL + "/oof_temp_diag.npz", allow_pickle=True)
    clean = weights(lab, 3, 0.0) >= 1
    rows = []
    for tag in ("r3", "w02"):
        d = pd.DataFrame({"farm": lab.farm.values, "day": lab.day.values, "hour": lab.hour.values,
                          "y": lab.sub_temp.values, "p": z[tag], "air": lab.in_temp.values, "ok": clean})
        d = d[~np.isnan(d.p)]
        full = d.groupby(["farm", "day"]).filter(lambda g: len(g) == 24 and g.ok.all())
        for (f, day), g in full.groupby(["farm", "day"]):
            g = g.sort_values("hour")
            yw = g.y.values - g.y.mean()
            pw = g.p.values - g.p.mean()
            aw = g.air.values - g.air.mean()
            best, blag = -2, 0
            for lag in range(-3, 4):
                if lag >= 0:
                    a, b = pw[lag:], yw[:len(yw) - lag]
                else:
                    a, b = pw[:lag], yw[-lag:]
                c = np.corrcoef(a, b)[0, 1]
                if c > best:
                    best, blag = c, lag
            rows.append(dict(tag=tag, farm=f, day=day, sec="2nd" if day >= 179 else "1st",
                             sd_true=yw.std(), sd_pred=pw.std(), air_sd=aw.std(),
                             pcorr=np.corrcoef(pw, yw)[0, 1], lag=blag,
                             within_rmse=np.sqrt(((pw - yw) ** 2).mean())))
    R = pd.DataFrame(rows)
    R["amp"] = R.sd_pred / R.sd_true
    for tag in ("r3", "w02"):
        r = R[R.tag == tag]
        print("\n######## %s: %d full clean days ########" % (tag, len(r)))
        print("amplitude ratio pred/true: median %.3f | IQR %.3f-%.3f"
              % (r.amp.median(), r.amp.quantile(.25), r.amp.quantile(.75)))
        print("within-day profile correlation: median %.3f" % r["pcorr"].median())
        print("best lag (h, + = prediction late): " +
              ", ".join("%+d:%d" % (k, v) for k, v in r.lag.value_counts().sort_index().items()))
        for sec, g in r.groupby("sec"):
            print("  %s pass: n=%d amp %.3f corr %.3f within-rmse %.3f"
                  % (sec, len(g), g.amp.median(), g["pcorr"].median(), g.within_rmse.mean()))
        r = r.assign(q=pd.qcut(r.air_sd, 4, labels=["air swing Q1 (small)", "Q2", "Q3", "Q4 (large)"]))
        for k, g in r.groupby("q", observed=True):
            print("  %-22s amp %.3f corr %.3f within-rmse %.3f"
                  % (k, g.amp.median(), g["pcorr"].median(), g.within_rmse.mean()))
    # what would a single global amplitude correction do (diagnostic upper bound)
    r = R[R.tag == "r3"]
    k = float((r.sd_true * r["pcorr"]).sum() / r.sd_pred.sum())
    print("\nleast-squares within-day scale for r3 (diagnostic): %.3f" % k)


if __name__ == "__main__":
    main()
