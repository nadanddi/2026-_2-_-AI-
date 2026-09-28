# -*- coding: utf-8 -*-
"""How independent are fold sets A and B?  (answer: not very)

Quoted in the report so that agreement between A and B is not oversold.
Usage:  PYTHONPATH="" python feat_folds.py
"""
import numpy as np

import env  # noqa: F401  MUST be first
from harness import load, folds, gap_stats
from common import split_mask


def main():
    panel, lab_t, lab_e = load()
    U = {}
    for k in ("A", "B"):
        fs = folds(k)
        U[k] = set()
        for fd in fs:
            for f, ds in fd.items():
                U[k] |= {(f, d) for d in ds}
        print("%s: %d greenhouse-days held out over %d folds, gap=%s"
              % (k, len(U[k]), len(fs), gap_stats(lab_e, fs)))
    inter, union = len(U["A"] & U["B"]), len(U["A"] | U["B"])
    print("union overlap: %d shared / %d union -> Jaccard %.3f"
          % (inter, union, inter / union))
    fa, fb = folds("A"), folds("B")
    per = []
    for i, a in enumerate(fa):
        sa = {(f, d) for f, ds in a.items() for d in ds}
        sb = {(f, d) for f, ds in fb[i].items() for d in ds}
        per.append(len(sa & sb) / len(sa | sb))
    print("fold-for-fold Jaccard: %s  mean %.3f"
          % (np.round(per, 3).tolist(), float(np.mean(per))))


if __name__ == "__main__":
    main()
