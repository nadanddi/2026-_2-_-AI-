# -*- coding: utf-8 -*-
"""verify_history v1b: ep_pos (days into episode) confounding for trans_p1 / ttrend5_dm11 / lowrad_run_dm11, plus dsince-onset usable proxy idea.
Fixed before running: if partial r (x, y | ep_pos) within episodes shrinks below half of the raw within r, call it confounded by episode progression."""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import numpy as np, pandas as pd
HX9 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
V = pd.read_csv(os.path.join(HX9, "history", "vars_v1.csv")).set_index("rec")
D = R.join(V, rsuffix="_v"); D["Y3"] = D.ec_mean - D.prev_ec_mean
hiS = D[(D.hi10 == 1) & (D.has_sg == 1)]; hi3 = D[(D.hi10 == 1) & D.prev_ec_mean.notna()]
def part(a, b, c):
    A = np.c_[np.ones(len(c)), c]; ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]; rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    return np.corrcoef(ra, rb)[0, 1]
for x, y, df in (("trans_p1", "Y3", hi3), ("trans_p1", "resid_mean", hiS), ("ttrend5_dm11", "Y3", hi3), ("lowrad_run_dm11", "resid_mean", hiS),
                 ("vpd_p3_dm11", "resid_mean", hiS), ("vpd_p3", "ec_mean", hiS)):
    sub = df[[x, y, "ep_id", "ep_pos"]].dropna()
    w = sub.groupby("ep_id")[[x, y, "ep_pos"]].transform(lambda s: s - s.mean())
    print(f"{x:16s} vs {y:10s} n {len(sub)} | within r {np.corrcoef(w[x], w[y])[0,1]:+.3f} | r(x,pos) {np.corrcoef(w[x], w.ep_pos)[0,1]:+.3f} "
          f"r(y,pos) {np.corrcoef(w[y], w.ep_pos)[0,1]:+.3f} | within partial | pos {part(w[x].values, w[y].values, w.ep_pos.values):+.3f} "
          f"| raw partial | pos (no demean) {part(sub[x].values, sub[y].values, sub.ep_pos.values):+.3f}")
# season-wide: does vpd_p3 rise with calendar over DMT3e days among ALL training records? (date-level trend)
a = D[["vpd_p3", "dm11_day"]].dropna(); print("all records r(vpd_p3, dm11_day)", round(np.corrcoef(a.vpd_p3, a.dm11_day)[0, 1], 3), "n", len(a))
b = D[(D.dm11_day >= 74) & (D.dm11_day <= 126)][["vpd_p3", "dm11_day"]].dropna(); print("dm11 74-126 r(vpd_p3, day)", round(np.corrcoef(b.vpd_p3, b.dm11_day)[0, 1], 3), "n", len(b))
