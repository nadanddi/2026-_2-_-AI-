# -*- coding: utf-8 -*-
"""AM1 (2026-10-11 집 클로드, user: "진폭 조절 실험도 해봐"): level-dependent within-day amplitude.
Catalog 6.544: high-EC days have ~4x the within-day swing of normal days (temperature-adjusted log-EC range .40 vs .09) and the
swing tracks the level; the model's within-day shape is already shrunk (shrink = .5 p + .5 running mean).  AM1 widens the
deviation from the causal running mean in proportion to the predicted level.  Fixed before running.
  cm_h = running mean of the prediction over hours 0..h (causal);  a = clip((cm_h - 0.8) / 0.4, 0, 1)
  AM1a: p' = cm_h + (1 + 0.5 a) (p - cm_h)          AM1b: p' = cm_h + (1 + 1.0 a) (p - cm_h)
Base: stored RV1 OOF, 9th-round-like CUR + SG2 (column sg), seeds 47/1414/6464.  All pass-2 rows (amplitude is a shape fix;
normal days with cm < .8 are unchanged by construction).  k = 2 -> alpha .0125.
PASS iff all 3 seeds better on DIAG10, P2LOO and EL1 pass-2 rows AND seed-mean farm x 5-day block bootstrap P(worse) on P2LOO
< .0125.  Reported: all-pass DIAG10 rows, high / normal days.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_am1_level_amplitude_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
ALPHA = .0125
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


def amp(G, beta):
    G = G.sort_values(["validator", "fold", "seed", "farm", "day", "hour"])
    cm = G.groupby(["validator", "fold", "seed", "farm", "day"]).sg.transform(lambda s: s.expanding().mean())
    a = np.clip((cm - .8) / .4, 0, 1)
    return (cm + (1 + beta * a) * (G.sg - cm)).reindex(G.index), G


def main():
    rng = np.random.default_rng(20261119)
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
    O = O[O.tag == "CUR"].copy()
    O["dm"] = O.groupby(["validator", "fold", "seed", "farm", "day"]).sub_ec.transform("mean")
    for name, beta in (("AM1a", .5), ("AM1b", 1.0)):
        p, _ = amp(O, beta); O[name] = p.reindex(O.index)
    for name in ("AM1a", "AM1b"):
        ok = True; print("\n==== %s" % name)
        for v in ("DIAG10", "P2LOO", "EL1"):
            X = O[(O.validator == v) & (O.day >= 179)]
            cells = [rm(Z[name] - Z.sub_ec) / rm(Z.sg - Z.sub_ec) - 1 for _, Z in X.groupby("seed")]
            ok &= all(x < 0 for x in cells)
            hi, nm = X.dm >= 1, X.dm < 1
            line = "  %-6s pass-2 cells %s | high %.4f -> %.4f | normal %.4f -> %.4f" % (v, " ".join("%+.2f%%" % (100 * x) for x in cells),
                   rm((X.sg - X.sub_ec)[hi]), rm((X[name] - X.sub_ec)[hi]), rm((X.sg - X.sub_ec)[nm]), rm((X[name] - X.sub_ec)[nm]))
            if v == "P2LOO":
                M = X.groupby("row_id").agg(y=("sub_ec", "first"), b=("sg", "mean"), c=(name, "mean"), farm=("farm", "first"), day=("day", "first"))
                cl = (M.farm + "_" + (M.day // 5).astype(str)).values
                dd = pd.Series((M.c - M.y) ** 2 - (M.b - M.y) ** 2).groupby(cl).agg(["sum", "count"])
                idx = rng.integers(0, len(dd), (20000, len(dd))); pw = float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())
                ok &= pw < ALPHA; line += " | P(worse) %.4f" % pw
            if v == "DIAG10":
                A = O[O.validator == v]; line += " | all-pass %+.2f%%" % (100 * (rm(A[name] - A.sub_ec) / rm(A.sg - A.sub_ec) - 1))
            print(line)
        print("VERDICT %s: %s" % (name, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
