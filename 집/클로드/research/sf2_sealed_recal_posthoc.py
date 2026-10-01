# -*- coding: utf-8 -*-
"""SF2 (POST-HOC descriptive, not a candidate): on sealed days, v2 ranks
high-EC days well (AUC .875) but shrinks the level.  Leave-one-DIAG10-fold-out
linear recalibration of the daily level on sealed days: fit y_daymean = a +
b * v2_daymean on the sealed days of the other 9 folds, apply as a level shift
to the held-out fold's sealed rows (hourly shape kept).  Sealed flag here uses
the WHOLE day's inputs (non-causal) - an optimistic bound; a causal version
must use only hours up to t.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u sf2_sealed_recal_posthoc.py"""
import env  # noqa
import os
import numpy as np
import pandas as pd
o = pd.read_csv(os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1",
                             "oof_predictions.csv"), encoding="utf-8-sig")
o = o[o.validator == "DIAG10"].copy()
D = pd.read_csv(os.path.join(env.LOCAL, "sf1_daytable.csv"))[["farm", "day", "sealed", "ec"]]
o = o.merge(D, on=["farm", "day"], how="left")
k = ["farm", "day"]
o["pm"] = o.groupby(k).v2.transform("mean")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
new = o.v2.copy()
coef = []
for fo in sorted(o.validation_fold.unique()):
    tr = o[(o.validation_fold != fo) & o.sealed].groupby(k)[["pm", "ec"]].first()
    b, a = np.polyfit(tr.pm, tr.ec, 1)
    coef.append((fo, round(a, 3), round(b, 3)))
    m = (o.validation_fold == fo) & o.sealed
    new[m] = o.v2[m] - o.pm[m] + (a + b * o.pm[m])
o["new"] = new
print("fold coefs (a, b):", coef)
print("DIAG10 all: v2 %.4f -> recal %.4f (%+.1f%%)" % (r(o.v2 - o.sub_ec), r(o.new - o.sub_ec), 100 * (r(o.new - o.sub_ec) / r(o.v2 - o.sub_ec) - 1)))
for nm, m in (("sealed", o.sealed), ("sealed high", o.sealed & (o.ec >= 1.2)), ("sealed normal", o.sealed & (o.ec < 1.2)),
              ("late sealed", o.sealed & (o.day >= 179))):
    print("  %-14s v2 %.4f recal %.4f" % (nm, r((o.v2 - o.sub_ec)[m]), r((o.new - o.sub_ec)[m])))
f = o.groupby("validation_fold").apply(lambda g: (r(g.new - g.sub_ec) - r(g.v2 - g.sub_ec)) / r(g.v2 - g.sub_ec) * 100, include_groups=False)
print("per-fold change %:", f.round(2).to_dict())
