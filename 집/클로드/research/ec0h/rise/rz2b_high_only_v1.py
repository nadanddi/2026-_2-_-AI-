# -*- coding: utf-8 -*-
"""RZ2b (사후 점검): RZ2 의 '상승일 = 밀폐·흐림' 이 고EC 날 자체의 특징인지 가리기 — 0시 EC25 >= 1.0 인 기록만 놓고 상승(+.15) vs 비상승 비교."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np
A = pd.read_csv("rz2_records_v1.csv", index_col=0)
H = A[A.lvl0 >= 1.0]; R = H.up >= .15
print("high-EC records (0h EC25>=1):", len(H), "| rise", int(R.sum()), "| pass-2", int(H.p2.sum()))
print("up distribution on high days:", np.round(H.up.quantile([.1, .25, .5, .75, .9]).values, 2).tolist(), "| fall<=-.15:", int((H.up <= -.15).sum()))
for k in ["lvl0", "rad", "otemp", "tin", "hum", "co2", "vent", "shade", "thermal", "heat", "fan", "st"]:
    a, b = H[R][k].values, H[~R][k].values
    auc = (a[:, None] > b[None, :]).mean() + .5 * (a[:, None] == b[None, :]).mean()
    print(f"  {k:8s} rise {np.median(a):9.2f} | no-rise high {np.median(b):9.2f} | AUC {auc:.2f}")
print("all records: fraction high-EC by farm/pass:", A.assign(h=A.lvl0 >= 1).groupby(["farm", "p2"]).h.mean().round(2).to_dict())
