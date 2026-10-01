# -*- coding: utf-8 -*-
"""LB2 (analysis only, no submission file): how far could an affine combination
of OUR OWN already-scored submissions go, using only the score identity?
2026-10-02 집 클로드.  Context: user asked for any idea that breaks the EC
deadlock.  NOTE: choosing blend weights from leaderboard scores is
"리더보드 역탐색(점수 기반 혼합)", which the user's 09-29 standard excludes
from candidates; this script only measures the size of that lever so the
user can decide (and ask the organisers).  No individual label is recovered.

Identity: for submissions p_i with RMSE s_i on the same n = 1440 rows,
  (p_i - p_k)·y = (||p_i||^2 - ||p_k||^2 - n s_i^2 + n s_k^2) / 2,
so for q = p_k + sum_i w_i (p_i - p_k)  (affine, weights sum to 1),
  n RMSE(q)^2 = n s_k^2 + ||D w||^2 + 2 w·(D^T p_k - D^T y)
is known exactly.  Scores are rounded to 4 decimals -> Monte-Carlo the
rounding (uniform +-0.00005) to give a range.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u lb2_score_span_bound_v1.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd

S = env.SUBMIT
F = {1: "01회차_2026-09-22/submission_01_scored.csv", 2: "02회차_2026-09-25/submission_03.csv",
     3: "03회차_2026-09-26/submission_04.csv", 4: "04회차_2026-09-26/submission_05.csv",
     5: "05회차_2026-09-26/submission_06.csv", 6: "06회차_2026-09-30(팀원)/submission_08.csv",
     7: "07회차_2026-10-01(팀)/submission_09.csv"}
SC = {"sub_ec": {1: .2442, 2: .2287, 3: .2055, 4: .2055, 5: .2134, 6: .2055, 7: .2774},
      "sub_temp": {1: .7450, 2: .6666, 3: .5624, 4: .5697, 5: .5456, 6: .5251, 7: .5193}}
RNG = np.random.default_rng(1)


def load():
    P = {k: pd.read_csv(os.path.join(S, v)).set_index("row_id") for k, v in F.items()}
    ids = P[3].index
    return {k: v.reindex(ids) for k, v in P.items()}, len(ids)


def analyse(P, n, col, base, bound):
    keys = [k for k in P if k != base]
    # drop exact duplicates of another column
    uniq, seen = [], [P[base][col].values]
    for k in keys:
        v = P[k][col].values
        if all(np.max(np.abs(v - s)) > 1e-9 for s in seen):
            uniq.append(k); seen.append(v)
    pk = P[base][col].values
    D = np.c_[[P[k][col].values - pk for k in uniq]].T
    G = D.T @ D
    rank = np.linalg.matrix_rank(D)
    res = []
    for it in range(2001):
        sc = {k: SC[col][k] + (0 if it == 0 else RNG.uniform(-5e-5, 5e-5)) for k in [base] + uniq}
        Dy = np.array([((P[k][col].values ** 2).sum() - (pk ** 2).sum() - n * sc[k] ** 2 + n * sc[base] ** 2) / 2
                       for k in uniq])
        b = D.T @ pk - Dy                              # D^T (p_k - y)
        w = np.linalg.solve(G + 1e-9 * np.eye(len(uniq)), -b)
        w = np.clip(w, -bound, bound)
        mse = (n * sc[base] ** 2 + w @ G @ w + 2 * w @ b) / n
        res.append((np.sqrt(max(mse, 0)), w))
    best0, w0 = res[0]
    vals = np.array([r[0] for r in res[1:]])
    print("%s: base round %d (%.4f), distinct other submissions %s, rank %d" % (col, base, SC[col][base], uniq, rank))
    print("   weights (|w| <= %.1f): %s" % (bound, dict(zip(uniq, np.round(w0, 3)))))
    print("   implied RMSE of best affine combination %.4f  (rounding range %.4f .. %.4f)" % (best0, vals.min(), vals.max()))
    return best0


def main():
    P, n = load()
    for col, base in (("sub_ec", 3), ("sub_temp", 7)):
        for bound in (1.0, 3.0):
            analyse(P, n, col, base, bound)
    # single direction: round 7 vs round 3 EC
    d = P[7].sub_ec.values - P[3].sub_ec.values
    dd = (d ** 2).mean(); de = ((.2774 ** 2 - .2055 ** 2) - dd) / 2
    lam = -de / dd
    print("\nEC one-direction: q = r3 + lam*(r7 - r3); best lam %.2f -> RMSE %.4f" % (lam, np.sqrt(.2055 ** 2 + lam ** 2 * dd + 2 * lam * de)))


if __name__ == "__main__":
    main()
