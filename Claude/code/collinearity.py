# -*- coding: utf-8 -*-
"""Multicollinearity audit of the training features, and does pruning matter?

1. pairwise |r| > 0.95 / 0.98 pairs, 2. VIF, 3. greedy correlation pruning
re-scored on the test-geometry CV for both targets.
"""
import numpy as np
import pandas as pd

from common import USABLE, TARGET_FARMS
from model_v2 import get_panel
from make_submission import T_HUB, E_HUB, ET, V5_ET, DOMAIN_MARKS
from geometry_cv import geometry_folds, score, lgb_fp, et_fp, blend
import features_v2 as F2


def vif_table(X):
    """VIF via R^2 of each column on the others (ridge-free, on standardised data)."""
    Z = (X - X.mean()) / X.std(ddof=0)
    Z = Z.fillna(0.0).values
    n, p = Z.shape
    out = {}
    for j in range(p):
        y = Z[:, j]
        A = np.delete(Z, j, axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        r2 = 1 - resid.var() / (y.var() + 1e-12)
        out[X.columns[j]] = 1.0 / max(1 - r2, 1e-9)
    return pd.Series(out).sort_values(ascending=False)


def greedy_prune(X, thr):
    """Drop the later of any pair with |r| > thr (keeps earlier = more basic columns)."""
    c = X.corr().abs()
    keep = []
    for col in X.columns:
        if all(c.loc[col, k] <= thr for k in keep):
            keep.append(col)
    return keep


def audit(name, X):
    c = X.corr().abs()
    iu = np.triu_indices_from(c, k=1)
    pairs = pd.Series(c.values[iu], index=pd.MultiIndex.from_arrays([c.index[iu[0]], c.columns[iu[1]]]))
    print("\n=== %s : %d features, %d rows ===" % (name, X.shape[1], len(X)))
    for t in (0.95, 0.98, 0.999):
        print("  pairs with |r| > %.3f : %4d" % (t, int((pairs > t).sum())))
    print("  most collinear pairs:")
    for (a, b), r in pairs.sort_values(ascending=False).head(8).items():
        print("    %.4f  %s  ~  %s" % (r, a, b))
    v = vif_table(X)
    print("  VIF > 10 : %d of %d   |  VIF > 100 : %d   |  median VIF %.1f"
          % (int((v > 10).sum()), len(v), int((v > 100).sum()), v.median()))
    print("  highest VIF:")
    for k, val in v.head(6).items():
        print("    %10.0f  %s" % (val, k))
    return v


def main():
    panel, tX, ty, sX = get_panel()
    panel["midnight"] = (panel.hour == 0).astype(float)
    folds = geometry_folds()
    lab = panel[~panel.is_test].reset_index(drop=True)

    v_temp = F2.view(panel, "sub_temp")
    v_ec = [c for c in F2.view(panel, "sub_ec") if not any(m in c for m in DOMAIN_MARKS)]
    v5 = USABLE + ["day", "hr_sin", "hr_cos", "midnight"]

    audit("sub_temp view", lab[v_temp])
    audit("sub_ec view (current)", lab[v_ec])
    audit("sub_ec v5 recipe", lab[v5])

    print("\n===== does pruning collinear features change the test-geometry CV? =====")
    lab_t = panel[(~panel.is_test) & panel.sub_temp.notna()].reset_index(drop=True)
    lab_e = panel[(~panel.is_test) & panel.sub_ec.notna()].reset_index(drop=True)
    print("  sub_temp  full %3d feat : %.4f" % (len(v_temp), score(lab_t, folds, "sub_temp", lgb_fp(v_temp, T_HUB))[0]))
    for thr in (0.98, 0.95, 0.90):
        keep = greedy_prune(lab_t[v_temp], thr)
        r = score(lab_t, folds, "sub_temp", lgb_fp(keep, T_HUB))[0]
        print("  sub_temp  |r|<=%.2f -> %3d feat : %.4f" % (thr, len(keep), r))
    cur = lambda cols: blend(et_fp(cols, ET), lgb_fp(cols, E_HUB))
    print("  sub_ec(cur) full %3d feat : %.4f" % (len(v_ec), score(lab_e, folds, "sub_ec", cur(v_ec))[0]))
    for thr in (0.98, 0.95, 0.90):
        keep = greedy_prune(lab_e[v_ec], thr)
        r = score(lab_e, folds, "sub_ec", cur(keep))[0]
        print("  sub_ec(cur) |r|<=%.2f -> %3d feat : %.4f" % (thr, len(keep), r))
    print("  sub_ec(v5)  full %3d feat : %.4f" % (len(v5), score(lab_e, folds, "sub_ec", et_fp(v5, V5_ET))[0]))


if __name__ == "__main__":
    main()
