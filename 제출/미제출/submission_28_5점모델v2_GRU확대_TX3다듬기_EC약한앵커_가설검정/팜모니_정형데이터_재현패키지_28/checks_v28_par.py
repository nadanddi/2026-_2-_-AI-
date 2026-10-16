# -*- coding: utf-8 -*-
"""submission_28 rule checks — PARALLEL driver for checks_v28.py (same runs, same comparisons, same output lines).
checks_v28.py runs the full pipeline six times one after another (base, 3 future-input cuts, 2 other-farm isolations), ~12 min each.
This driver imports checks_v28.run / perturbed and runs the six pipelines concurrently in a thread pool (each run already uses its own temp
directory and starts every step as its own Python process; GRU training is CPU with 4 fixed threads -> max 4 runs at once on 16 cores).
Nothing in the pipelines or the comparisons changes.
Run from the package root:  python checks_v28_par.py <folder with the ORIGINAL public train_X.csv, train_y.csv, test_X.csv>
"""
import os, sys, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import checks_v28 as C

CUTS = (("F13", 206, 4), ("F13", 237, 3), ("F47", 202, 5))
ISO = (("F13", "F47"), ("F47", "F13"))


def cut_mask(farm, d, h):
    def m(X):
        f, dy, hr = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
        return (f == farm) & ((dy > d) | ((dy == d) & (hr > h)))
    return m


def job(kind, arg, pub):
    if kind == "base":
        return C.run(pub)
    w = C.perturbed(pub, cut_mask(*arg) if kind == "cut" else (lambda X, o=arg[1]: X.row_id.str[:3] == o))
    try:
        return C.run(w)
    finally:
        shutil.rmtree(w, ignore_errors=True)


def main():
    pub = sys.argv[1]
    jobs = [("base", None)] + [("cut", c) for c in CUTS] + [("iso", i) for i in ISO]
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {j: ex.submit(job, j[0], j[1], pub) for j in jobs}
        res = {j: f.result() for j, f in futs.items()}
    T0, E0, _ = res[("base", None)]; ok = True
    out_csv = os.path.join(C.ROOT, "submission_28_regen.csv")
    if not os.path.exists(out_csv):
        pd.DataFrame({"row_id": T0.index, "sub_temp": T0.values, "sub_ec": E0.values}).to_csv(out_csv, index=False, float_format="%.6f")
    r = pd.read_csv(os.path.join(C.ROOT, "submission_27.csv")).set_index("row_id")
    print("[0] regenerated: temp changed vs round-17 rows %d (RMS %.4f), EC changed vs round-17 rows %d (RMS %.4f)"
          % (int((np.abs(T0 - r.sub_temp.reindex(T0.index)) > 5e-7).sum()), float(np.sqrt(((T0 - r.sub_temp.reindex(T0.index)) ** 2).mean())),
             int((np.abs(E0 - r.sub_ec.reindex(T0.index)) > 5e-7).sum()), float(np.sqrt(((E0 - r.sub_ec.reindex(T0.index)) ** 2).mean()))), flush=True)
    ix = T0.index; ff, fd, fh = ix.str[:3], ix.str[4:7].astype(int), ix.str[8:10].astype(int)
    for farm, d, h in CUTS:
        T1, E1, _ = res[("cut", (farm, d, h))]
        past = (ff == farm) & ((fd < d) | ((fd == d) & (fh <= h))); fut = (ff == farm) & ~past
        a = bool((T0[past] == T1[past]).all()); b = bool((E0[past] == E1[past]).all()); ok &= a and b
        print("[F] %s cut %d h%02d: temp past unchanged %s, EC past unchanged %s (past rows %d; future moved temp %.2f, EC %.3f)"
              % (farm, d, h, a, b, int(past.sum()), float(np.abs(T0[fut] - T1[fut]).sum()), float(np.abs(E0[fut] - E1[fut]).sum())), flush=True)
    for farm, other in ISO:
        T1, E1, _ = res[("iso", (farm, other))]; own = ff == farm
        a = bool((T0[own] == T1[own]).all()); b = bool((E0[own] == E1[own]).all()); ok &= a and b
        print("[X] %s isolation: temp unchanged %s, EC unchanged %s" % (farm, a, b), flush=True)
    sub = os.path.join(C.ROOT, "submission_28.csv")
    if os.path.exists(sub):
        S = pd.read_csv(sub).set_index("row_id")
        same = bool((np.abs(S.sub_temp.reindex(T0.index) - T0) < 5e-7).all() and (np.abs(S.sub_ec.reindex(T0.index) - E0) < 5e-7).all())
        print("[R] full pipeline reproduces submission_28.csv (6 decimals): %s" % same); ok &= same
    print("ALL PASS" if ok else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
