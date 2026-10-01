# -*- coding: utf-8 -*-
"""RD3: copy-fill check for F13/F47.  Were stretches of the indoor record
filled by pasting another stretch (another day of the same or the other
farm, any hour shift)?  2026-10-02 집 클로드 (user: F13/F47 may also carry
corrections of a kind the linear detector cannot see).

Match = in_temp, in_hum, in_co2 ALL exactly equal for >= K consecutive hours
(K = 4, 6, 8), between two different positions (farm, t) in the pooled
F13/F47 train+test indoor record (also F32 as a source).  Stretches that are
constant in all three channels are excluded (trivial matches).
Reported by pair type: train-train, test-train, test-test.
Also the 48 integer farms' non-linear fractional runs (rd1 'other runs'):
exact in_hum+in_co2 copy of another stretch of the same farm (in_temp cannot
match exactly there because of the rounding).

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u rd3_segment_copy_v1.py
"""
import env  # noqa: F401
import os
from collections import defaultdict

import numpy as np
import pandas as pd

D = env.DATA


def load():
    tX = pd.read_csv(os.path.join(D, "train_X.csv"))
    sX = pd.read_csv(os.path.join(D, "test_X.csv"))
    tX["set"], sX["set"] = "train", "test"
    X = pd.concat([tX, sX], ignore_index=True)
    p = X.row_id.str.split("_", expand=True)
    X["farm"], X["day"], X["hour"] = p[0], p[1].astype(int), p[2].astype(int)
    X["t"] = X.day * 24 + X.hour
    return X.sort_values(["farm", "t"]).reset_index(drop=True)


def windows(g, K, cols):
    """hash of K-hour windows -> list of (farm, t0, set)"""
    v = g[cols].values
    t = g.t.values
    out = []
    for i in range(len(g) - K + 1):
        w = v[i:i + K]
        if np.isnan(w).any() or t[i + K - 1] - t[i] != K - 1:
            continue
        if all(np.all(w[:, c] == w[0, c]) for c in range(len(cols))):
            continue
        out.append((w.tobytes(), i))
    return out


def main():
    X = load()
    cols = ["in_temp", "in_hum", "in_co2"]
    S = X[X.farm.isin(["F13", "F47", "F32"])]
    for K in (4, 6, 8):
        idx = defaultdict(list)
        for f, g in S.groupby("farm"):
            g = g.reset_index(drop=True)
            for h, i in windows(g, K, cols):
                idx[h].append((f, int(g.t[i]), g.set[i]))
        pairs = defaultdict(set)
        for h, L in idx.items():
            if len(L) < 2:
                continue
            for a in range(len(L)):
                for b in range(a + 1, len(L)):
                    A, B = L[a], L[b]
                    if A[0] == B[0] and abs(A[1] - B[1]) < K:
                        continue  # overlapping self window
                    typ = "-".join(sorted([A[2], B[2]]))
                    pairs[typ].add((A[:2], B[:2]))
        print("K=%d: exact 3-channel copies (window pairs):" % K,
              {k: len(v) for k, v in pairs.items()} or "none")
        for typ, ps in pairs.items():
            ex = sorted(ps)[:6]
            print("   %s e.g. %s" % (typ, [("%s d%d h%d" % (a[0], a[1] // 24, a[1] % 24),
                                            "%s d%d h%d" % (b[0], b[1] // 24, b[1] % 24)) for a, b in ex]))

    # integer farms: non-linear fractional runs, exact hum+co2 copies within the same farm
    I = X[~X.farm.isin(["F13", "F47", "F32"]) & (X.set == "train")].copy()
    I["fr"] = I.in_temp.notna() & (np.abs(I.in_temp - np.round(I.in_temp)) > 1e-9)
    n_runs = n_hit = 0
    n_ctrl = n_ctrl_hit = 0
    rng = np.random.default_rng(3)
    for f, g in I.groupby("farm"):
        g = g.reset_index(drop=True)
        hv = g[["in_hum", "in_co2"]].values
        t, fr = g.t.values, g.fr.values
        W = {}
        for i in range(len(g) - 6 + 1):
            w = hv[i:i + 6]
            if np.isnan(w).any() or t[i + 5] - t[i] != 5:
                continue
            W.setdefault(w.tobytes(), []).append(i)
        i = 0
        starts = []
        while i < len(g):
            if not fr[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(g) and fr[j + 1] and t[j + 1] == t[j] + 1:
                j += 1
            if j - i + 1 >= 6:
                starts.append(i)
            i = j + 1
        for i in starts:
            n_runs += 1
            hits = [k for k in W.get(hv[i:i + 6].tobytes(), []) if abs(k - i) >= 6]
            n_hit += bool(hits)
        # control: random integer-row windows of the same farm
        cand = [i for i in range(len(g) - 6) if not fr[i:i + 6].any()]
        for i in rng.choice(cand, size=min(len(starts) * 5 + 5, len(cand)), replace=False):
            n_ctrl += 1
            n_ctrl_hit += bool([k for k in W.get(hv[i:i + 6].tobytes(), []) if abs(k - i) >= 6])
    print("\ninteger farms: fractional runs >=6h whose first 6 h of (in_hum, in_co2) exactly repeat elsewhere"
          " in the same farm: %d / %d (%.1f%%); control integer windows %d / %d (%.1f%%)"
          % (n_hit, n_runs, 100 * n_hit / max(n_runs, 1), n_ctrl_hit, n_ctrl, 100 * n_ctrl_hit / max(n_ctrl, 1)))


if __name__ == "__main__":
    main()
