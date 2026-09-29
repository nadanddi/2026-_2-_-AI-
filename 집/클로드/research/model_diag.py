# -*- coding: utf-8 -*-
"""Target distribution facts that decide which model families are admissible.

Checks: strict positivity (required by Gamma / log-link), skew, the size of
the high-EC segment, and how much of the feature matrix is missing (decides
whether an imputer is mandatory for the non-tree families).
"""
import env  # noqa: F401  MUST be first project import
import numpy as np

from harness import load, views


def main():
    panel, lab_t, lab_e = load()
    v = views(panel)
    for target, lab in [("sub_temp", lab_t), ("sub_ec", lab_e)]:
        y = lab[target].values.astype(float)
        print("\n== %s ==" % target)
        print("  n=%d  min=%.4f  max=%.4f  mean=%.4f  sd=%.4f  skew=%.3f"
              % (len(y), y.min(), y.max(), y.mean(), y.std(),
                 float(((y - y.mean()) ** 3).mean() / y.std() ** 3)))
        print("  quantiles " + " ".join(
            "%s=%.3f" % (q, np.quantile(y, q)) for q in
            (0.01, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0)))
        print("  <=0 rows: %d" % int((y <= 0).sum()))
        for cut in (0.8, 1.0, 1.2, 1.5):
            m = y > cut
            sse = ((y - y.mean()) ** 2)
            print("    y>%.1f : n=%4d (%.1f%%)  share of var %.3f"
                  % (cut, m.sum(), 100.0 * m.mean(), sse[m].sum() / sse.sum()))

    for k in ("ec", "temp", "v5", "raw"):
        cols = v[k]
        X = lab_e[cols]
        nan_frac = float(X.isna().to_numpy().mean())
        col_any = int(X.isna().any().sum())
        print("\nview %-5s : %3d cols, NaN cells %.4f, cols with any NaN %d"
              % (k, len(cols), nan_frac, col_any))
        # how many columns are near-constant (kills scaled linear models)
        sd = X.std(numeric_only=True)
        print("            near-constant cols (sd<1e-9): %d" % int((sd < 1e-9).sum()))

    # day-level variance share for EC (sample-efficiency argument)
    g = lab_e.groupby(["farm", "day"])["sub_ec"]
    dm = g.transform("mean")
    y = lab_e["sub_ec"].values
    print("\nEC: R^2 of the daily mean alone = %.4f  (n greenhouse-days = %d)"
          % (1 - np.sum((y - dm) ** 2) / np.sum((y - y.mean()) ** 2),
             lab_e.groupby(["farm", "day"]).ngroups))


if __name__ == "__main__":
    main()
