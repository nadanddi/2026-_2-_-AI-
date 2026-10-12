# -*- coding: utf-8 -*-
"""VL4: are same-date training twins of Sq targets hidden in the same DIAG10 fold? (eval pure window has no same-date training)."""
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import numpy as np, pandas as pd
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
FOLD = O.groupby("rec").fold.first(); Q = O.groupby("rec").sg.mean()
Xf = X.reshape(N, -1); TR = np.where(TRAIN)[0]
rows = []
for r in FOLD.index:
    i = idx[r]; eq = (np.abs(Xf[TR] - Xf[i]) < 1e-6).mean(1); tw = [rid(j) for j in TR[eq > 0.7] if j != i]
    seen = [t for t in tw if t in FOLD.index and FOLD[t] != FOLD[r]]
    rows.append(dict(rec=r, ntw=len(tw), nseen=len(seen), y=float(np.nanmean(EC[i])), q=float(Q[r])))
R = pd.DataFrame(rows)
print("all OOF recs:", len(R), "| with twins", (R.ntw > 0).sum(), "| any twin seen by model (other fold)", (R.nseen > 0).sum())
S = R[R.q >= 1.0]
print("Sq:", len(S), "| with twins", (S.ntw > 0).sum(), "| twin seen", (S.nseen > 0).sum(),
      f"| q RMSE twin-seen {rm((S.q-S.y)[S.nseen>0]) if (S.nseen>0).any() else float('nan'):.3f} (n {(S.nseen>0).sum()}) vs no-seen-twin {rm((S.q-S.y)[S.nseen==0]):.3f} (n {(S.nseen==0).sum()})")
print("DIAG10 fold sizes:", FOLD.value_counts().sort_index().to_dict())
