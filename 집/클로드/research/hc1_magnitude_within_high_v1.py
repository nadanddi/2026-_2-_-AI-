# -*- coding: utf-8 -*-
"""HC1 (descriptive; 2026-10-04 집 클로드).  Within high-EC days (n=31) and within
sealed days (vent zero >= .8, n=85): which inputs track the SIZE of the day EC?
Spearman with label day mean; plus previous same-동 record's EC (label, if labelled)
and days into the sealed run.  Small n -> descriptive; |rho| >= .45 (n=31) ~ p<.01."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
D = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv"))
R = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).sort_values(["farm", "day"])
R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
D = D.merge(R[["farm", "day", "date"]], on=["farm", "day"])
EC = D.set_index(["farm", "day"]).ec
prev = []
for _, r in D.iterrows():
    G = D[(D.farm == r.farm) & (D.dong == r.dong) & (D.date < r.date) & (D.date >= r.date - 3)]
    prev.append(G.sort_values("date").ec.iloc[-1] if len(G) else np.nan)
D["prev_same_dong_ec"] = prev
cols = [c for c in D.columns if c not in ("farm", "day", "ec", "hi", "role", "dong", "date")]
for name, S in (("HIGH-EC days (ec>=1)", D[D.hi == 1]), ("SEALED days (vent zero>=.8)", D[D.act_vent_zero >= .8])):
    rows = []
    for c in cols:
        z = S[[c, "ec"]].dropna()
        if len(z) < 10 or z[c].nunique() < 3: continue
        rows.append((c, spearmanr(z[c], z.ec).correlation, len(z)))
    T = pd.DataFrame(rows, columns=["feature", "rho", "n"]).assign(a=lambda x: x.rho.abs()).sort_values("a", ascending=False)
    print("\n%s: n %d, EC q10 %.2f med %.2f q90 %.2f" % (name, len(S), S.ec.quantile(.1), S.ec.median(), S.ec.quantile(.9)))
    print(T.head(12).drop(columns="a").round(2).to_string(index=False))
H = D[D.hi == 1].sort_values("ec")
print("\nHIGH-EC days listed (low -> high):")
print(H[["farm", "day", "dong", "ec", "act_vent_zero", "out_temp_mean", "in_out_diff", "act_co2_day", "act_heating_mean", "prev_same_dong_ec"]].round(2).to_string(index=False))
