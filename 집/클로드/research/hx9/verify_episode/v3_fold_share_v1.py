# -*- coding: utf-8 -*-
"""V3: d_prev_resid 잔차 지속이 DIAG10 같은 폴드(함께 가려짐) 때문인지 점검. 예측: 같은 폴드 비율이 높고, 같은 폴드 쌍에서만 상관이 큼."""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import numpy as np, pandas as pd
from scipy import stats
RES = r"C:\work\farmai\집\클로드\research"
R = pd.read_csv(os.path.join(RES, "hx9", "data", "records.csv")).set_index("rec")
m = R.prev_rec.notna() & (R.has_sg == 1)
pf = R.fold.reindex(R.prev_rec[m]).values; bf = R.fold[m].values
pr = R.resid_mean.reindex(R.prev_rec[m]).values; br = R.resid_mean[m].values
same = pf == bf; fin = np.isfinite(pr) & np.isfinite(br)
hi = (R.ec_mean[m] >= 1.0).values
for nm, mk in (("all", np.ones(len(br), bool)), ("hi10", hi)):
    for s_nm, s in (("same fold", same), ("diff fold", ~same)):
        k = mk & s & fin
        r = stats.pearsonr(pr[k], br[k])[0] if k.sum() > 4 else np.nan
        print(f"[{nm}] {s_nm}: n {k.sum()} resid-resid r {r:+.3f}")
    print(f"[{nm}] same-fold share {np.mean(same[mk & fin]):.2f}")
