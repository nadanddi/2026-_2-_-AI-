# -*- coding: utf-8 -*-
"""Forensic 8: season-matched comparison (train days >= 175 vs test days) of day signatures."""
import env
import numpy as np, pandas as pd
from scipy import stats
D = pd.read_csv(env.LOCAL+"/eda_forensic_7_days.csv")
print("test day range", D[D.set=="test"].groupby("farm").day.agg(["min","max"]).to_dict())
feats=["rough_T_night","rough_H_night","rough_C_night","rough_AH","maxjump_res","min_in_minus_out_night","max_in_minus_out","corr_in_out"]
for lo in [0,120,175]:
    tr=D[(D.set=="train")&(D.day>=lo)]; te=D[D.set=="test"]
    print(f"\n== train days >= {lo}: n={len(tr)} vs test {len(te)}")
    for c in feats:
        ks=stats.mannwhitneyu(tr[c].dropna(),te[c].dropna())
        print(f"{c:24s} train med {tr[c].median():7.2f} p90 {tr[c].quantile(.9):7.2f} | test med {te[c].median():7.2f} p90 {te[c].quantile(.9):7.2f} | MWU p={ks.pvalue:.3g}")
