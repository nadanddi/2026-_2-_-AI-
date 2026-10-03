# -*- coding: utf-8 -*-
"""HC2 (descriptive; 2026-10-04 집 클로드).  On high-EC days (label day mean >= 1,
DIAG10, R3S seed mean): split the day-level squared error into a common shift
(mean under-prediction) and the remaining day-to-day part; rank agreement of
predicted vs true size within high days; also how much of the remaining part a
perfect within-high ranking with the same predicted spread would remove."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy()
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
D = O.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean")).reset_index()
H = D[D.y >= 1].copy(); e = H.p - H.y
mse = np.mean(e ** 2); bias = e.mean()
print("high-EC days %d | day-level MSE %.4f = shift^2 %.4f (%.0f%%) + day-to-day %.4f (%.0f%%)" % (
    len(H), mse, bias ** 2, 100 * bias ** 2 / mse, e.var(ddof=0), 100 * e.var(ddof=0) / mse))
print("mean true %.3f  mean pred %.3f  | sd true %.3f  sd pred %.3f" % (H.y.mean(), H.p.mean(), H.y.std(), H.p.std()))
print("within-high rank agreement: Spearman %.2f, Pearson %.2f" % (spearmanr(H.p, H.y).correlation, pearsonr(H.p, H.y)[0]))
# best linear rescale of predictions within high days (oracle, in-sample) for scale
b = np.polyfit(H.p, H.y, 1); print("in-sample best line y = %.2f*p + %.2f -> residual MSE %.4f" % (b[0], b[1], np.mean((np.polyval(b, H.p) - H.y) ** 2)))
tot = np.mean((D.p - D.y) ** 2)
print("share of ALL-day day-level MSE on high days: %.0f%% (shift part %.0f%%, day-to-day part %.0f%%)" % (
    100 * np.sum(e ** 2) / np.sum((D.p - D.y) ** 2), 100 * len(H) * bias ** 2 / np.sum((D.p - D.y) ** 2), 100 * len(H) * e.var(ddof=0) / np.sum((D.p - D.y) ** 2)))
