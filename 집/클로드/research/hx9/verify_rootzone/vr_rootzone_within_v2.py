# -*- coding: utf-8 -*-
"""verify_rootzone v2: per-chain within correlations for heating / cold vars (chains 0,1,6 only have variance). 서술."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import env  # noqa
import numpy as np, pandas as pd
HX9 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = pd.read_csv(os.path.join(HX9, "rootzone", "vars_v1.csv"), index_col="rec")
R = pd.read_csv(os.path.join(HX9, "data", "records.csv"), index_col="rec")
hi = V[(V.hi10 == 1) & V.Y2.notna()].copy(); hi["ep"] = hi.ep_id.astype(int)
hi["dm11"] = R.loc[hi.index, "dm11_day"]; hi["otd"] = R.loc[hi.index, "out_temp_d"]
for e in (0, 1, 6):
    s = hi[hi.ep == e].sort_values("dm11")
    print(f"chain {e} n {len(s)}")
    print(s[["dm11", "farm", "heat_mean", "heat_night", "in_cold10", "otd", "ec_mean", "sg_mean", "Y2"]].round(3).to_string())
    for v in ("heat_mean", "heat_hours", "heat_night", "in_cold10", "otd"):
        print(f"   r({v}, Y2) {np.corrcoef(s[v], s.Y2)[0,1]:+.3f}  r({v}, ec_mean) {np.corrcoef(s[v], s.ec_mean)[0,1]:+.3f}  r({v}, sg_mean) {np.corrcoef(s[v], s.sg_mean)[0,1]:+.3f}")
