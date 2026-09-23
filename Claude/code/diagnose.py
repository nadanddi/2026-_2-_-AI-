# -*- coding: utf-8 -*-
"""Sanity checks on the submission and a test-period-restricted CV estimate."""
import os
import numpy as np
import pandas as pd

from common import make_folds, rmse, TARGET_FARMS
from cv import prepare, N_FOLDS
from final_model import cv_score

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(os.path.dirname(HERE), "submission.csv")


def main():
    panel, tX, ty, sX = prepare(use_aux=True)
    sub = pd.read_csv(SUB)
    sub["farm"] = sub.row_id.str[:3]
    sub["day"] = sub.row_id.str[4:7].astype(int)

    print("=== submission format ===")
    print("  rows           :", len(sub), "(need 1440)")
    print("  columns        :", list(sub.columns[:3]))
    print("  row_id matches :", sub.row_id.tolist() == sX.row_id.tolist())
    print("  duplicates     :", int(sub.row_id.duplicated().sum()))
    print("  NaN / inf      :", int(sub[["sub_temp", "sub_ec"]].isna().sum().sum()),
          "/", int((~np.isfinite(sub[["sub_temp", "sub_ec"]].values)).sum()))

    print("\n=== predicted vs observed ranges ===")
    obs = ty[ty.row_id.str[:3].isin(TARGET_FARMS)]
    for t in ["sub_temp", "sub_ec"]:
        o = obs[t].dropna()
        print("  %-9s pred [%.3f, %.3f] mean %.3f | train [%.3f, %.3f] mean %.3f"
              % (t, sub[t].min(), sub[t].max(), sub[t].mean(),
                 o.min(), o.max(), o.mean()))

    print("\n=== test-block daily level vs neighbouring train days ===")
    lab = ty.copy()
    lab["farm"] = lab.row_id.str[:3]
    lab["day"] = lab.row_id.str[4:7].astype(int)
    for farm in TARGET_FARMS:
        s = sub[sub.farm == farm]
        L = lab[(lab.farm == farm) & lab.sub_ec.notna()]
        dl = L.groupby("day").sub_ec.mean()
        days = np.sort(s.day.unique())
        blocks = np.split(days, np.where(np.diff(days) > 1)[0] + 1)
        print("  [%s]" % farm)
        for b in blocks:
            pm = s[s.day.isin(b)].sub_ec.mean()
            near = dl[(dl.index >= b[0] - 6) & (dl.index <= b[-1] + 6)]
            tm = near.mean() if len(near) else float("nan")
            pt = s[s.day.isin(b)].sub_temp.mean()
            Lt = lab[(lab.farm == farm)]
            dlt = Lt.groupby("day").sub_temp.mean()
            neart = dlt[(dlt.index >= b[0] - 6) & (dlt.index <= b[-1] + 6)]
            print("    day %3d-%3d  EC pred %.3f vs train-nbr %.3f (n=%d) | "
                  "temp pred %.2f vs nbr %.2f"
                  % (b[0], b[-1], pm, tm, len(near), pt,
                     neart.mean() if len(neart) else float("nan")))

    print("\n=== CV restricted to the test day range (183-245) ===")
    days = {f: sorted(panel[(panel.farm == f) & (~panel.is_test)].day.unique())
            for f in TARGET_FARMS}
    folds = make_folds(days, n_folds=N_FOLDS, seed=0)
    for target in ["sub_temp", "sub_ec"]:
        r, per, oof, labp, _ = cv_score(panel, folds, target, verbose=False)
        got = ~np.isnan(oof)
        late = got & (labp.day.values >= 183)
        early = got & (labp.day.values < 183)
        print("  %-9s all %.4f | late(>=183) %.4f n=%d | early %.4f n=%d"
              % (target, r,
                 rmse(oof[late], labp[target].values[late]), int(late.sum()),
                 rmse(oof[early], labp[target].values[early]), int(early.sum())))


if __name__ == "__main__":
    main()
