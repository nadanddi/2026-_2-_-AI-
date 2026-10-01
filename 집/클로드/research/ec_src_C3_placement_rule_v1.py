# -*- coding: utf-8 -*-
"""EC source search C3: does a record placement rule identify the source (동)?
Rules fixed in EC_정보원_입력쪽_사전고정_2026-10-02.md (commit 81ce501).
2026-10-02 집 클로드.

Reference for "same source": deep_cal_11 input-continuity chains (inputs only;
uses future inputs, so it is a yardstick here, never a feature).
Test pairs: every calendar step c -> c+1 within a greenhouse where both dates
have exactly two record days and the two days at c belong to two different
chains, each continuing at c+1.  For each such step the rule must say which
c+1 day continues each c day (2 decisions per step).
Rules:
  R1 occurrence order: the day that is the 1st (2nd) record occurrence of its
     calendar date continues the 1st (2nd) one at c+1
  R2 parity of the record day index is kept
  R3 the c+1 day nearer in record index continues (smallest |day gap|)
Accuracy per segment (1st: day < 179, 2nd: day >= 179; a step belongs to the
2nd segment if both its days are >= 179).  GO if one rule has accuracy >= 0.80
in BOTH segments and binomial p < 0.01/3 against 0.5 in each.

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_src_C3_placement_rule_v1.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def main():
    C = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_11_days.csv"))[["farm", "day", "cal", "chain", "is_test"]]
    C = C.sort_values(["farm", "day"]).reset_index(drop=True)
    C["occ"] = C.groupby(["farm", "cal"]).cumcount()
    rec = []
    for f, g in C.groupby("farm"):
        byc = {c: h for c, h in g.groupby("cal")}
        for c in sorted(byc):
            if c + 1 not in byc:
                continue
            A, B = byc[c], byc[c + 1]
            if len(A) != 2 or len(B) != 2 or A.chain.nunique() != 2:
                continue
            if set(A.chain) != set(B.chain):
                continue   # chains do not both continue -> no yardstick
            for a in A.itertuples():
                truth = B[B.chain == a.chain].iloc[0]
                other = B[B.chain != a.chain].iloc[0]
                seg = "2nd" if (a.day >= 179 and truth.day >= 179 and other.day >= 179) else (
                    "1st" if (a.day < 179 and truth.day < 179 and other.day < 179) else "mixed")
                r1 = (truth.occ == a.occ) and (other.occ != a.occ)
                r2 = (truth.day % 2 == a.day % 2) and (other.day % 2 != a.day % 2)
                r3 = abs(truth.day - a.day) < abs(other.day - a.day)
                rec.append(dict(farm=f, cal=c, seg=seg, R1=r1, R2=r2, R3=r3, test=bool(a.is_test or truth.is_test)))
    R = pd.DataFrame(rec)
    print("decisions:", R.groupby("seg").size().to_dict(), "(involving test days: %d)" % R.test.sum())
    go = False
    for rule in ("R1", "R2", "R3"):
        ok = True
        line = []
        for seg in ("1st", "2nd"):
            g = R[R.seg == seg]
            k, n = int(g[rule].sum()), len(g)
            p = binomtest(k, n, 0.5, alternative="greater").pvalue if n else np.nan
            acc = k / n if n else np.nan
            ok &= (n > 0) and acc >= 0.80 and p < 0.01 / 3
            line.append("%s %.2f (%d/%d, p %.2g)" % (seg, acc, k, n, p))
        mixed = R[R.seg == "mixed"][rule].mean() if (R.seg == "mixed").any() else np.nan
        print("  %s: %s | mixed-segment %.2f -> %s" % (rule, "  ".join(line), mixed, "GO" if ok else "no"))
        go |= ok
    print("\nC3 decision:", "GO" if go else "STOP")


if __name__ == "__main__":
    main()
