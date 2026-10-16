# -*- coding: utf-8 -*-
"""submission_18 rule checks: future-input and other-farm isolation for both new steps (temperature chain correction, EC C18).
For each cut (farm, record d, hour h): every test_X input of that farm strictly after (d, h) is multiplied by 11 and +777;
predictions of that farm's rows at or before (d, h) must be bit-identical.  Isolation: perturb all test_X inputs of the other farm.
Run from the package root:  python checks_v18.py"""
import os, sys, shutil, runpy, tempfile, contextlib, io
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
COLS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
        "act_heating", "act_circfan", "act_co2", "act_fog"]


def script(path, args):
    """run a package script in-process (same interpreter and environment), silencing its prints"""
    old = sys.argv; sys.argv = [path] + args
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(path, run_name="__main__")
    finally:
        sys.argv = old


def run(data_dir, tag):
    w = tempfile.mkdtemp(prefix="c18chk_")
    t_out = os.path.join(w, "t.csv")
    script(os.path.join(ROOT, "온도", "code", "make_temp_v18_chain.py"), [data_dir, os.path.join(ROOT, "온도", "data", "oof_diag10_tt1"),
           os.path.join(ROOT, "온도", "temp_candidate_v13.csv"), t_out])
    ec_dir = os.path.join(w, "EC"); shutil.copytree(os.path.join(ROOT, "EC"), ec_dir, ignore=shutil.ignore_patterns("output_18", "data", "dlls", "*.ckpt"))
    shutil.copytree(data_dir, os.path.join(ec_dir, "data"))
    script(os.path.join(ec_dir, "c18post.py"), [])
    t = pd.read_csv(t_out).set_index("row_id").sub_temp; e = pd.read_csv(os.path.join(ec_dir, "output_18", "ec_c18.csv")).set_index("row_id").sub_ec
    e = e.reindex(t.index)        # same row order for both series (masks below are built on this index)
    shutil.rmtree(w, ignore_errors=True)
    return t, e


def main():
    base_dir = os.path.join(ROOT, "EC", "data")
    T0, E0 = run(base_dir, "base")
    ok = True
    for farm, d, h in (("F13", 222, 5), ("F13", 235, 3), ("F47", 220, 6)):
        w = tempfile.mkdtemp(prefix="c18dat_"); shutil.copytree(base_dir, os.path.join(w, "d")); dd = os.path.join(w, "d")
        X = pd.read_csv(os.path.join(dd, "test_X.csv")); f, dy, hr = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
        fut = (f == farm) & ((dy > d) | ((dy == d) & (hr > h)))
        X.loc[fut, COLS] = X.loc[fut, COLS] * 11 + 777; X.to_csv(os.path.join(dd, "test_X.csv"), index=False)
        T1, E1 = run(dd, "cut"); shutil.rmtree(w, ignore_errors=True)
        idx = T0.index; ff, fd, fh = idx.str[:3], idx.str[4:7].astype(int), idx.str[8:10].astype(int)
        past = (ff == farm) & ((fd < d) | ((fd == d) & (fh <= h)))
        a = bool((T0[past] == T1[past]).all()); b = bool((E0[past] == E1[past]).all()); ok &= a and b
        print("[F] %s cut record %d h%02d: temp past rows unchanged %s, EC past rows unchanged %s (past rows %d; future temp moved %.2f, EC moved %.3f)"
              % (farm, d, h, a, b, int(past.sum()), float(np.abs(T0[~past & (ff == farm)] - T1[~past & (ff == farm)]).sum()), float(np.abs(E0[~past & (ff == farm)] - E1[~past & (ff == farm)]).sum())))
    for farm, other in (("F13", "F47"), ("F47", "F13")):
        w = tempfile.mkdtemp(prefix="c18dat_"); shutil.copytree(base_dir, os.path.join(w, "d")); dd = os.path.join(w, "d")
        X = pd.read_csv(os.path.join(dd, "test_X.csv")); m = X.row_id.str[:3] == other
        X.loc[m, COLS] = X.loc[m, COLS] * 11 + 777; X.to_csv(os.path.join(dd, "test_X.csv"), index=False)
        T1, E1 = run(dd, "iso"); shutil.rmtree(w, ignore_errors=True)
        own = T0.index.str[:3] == farm
        a = bool((T0[own] == T1[own]).all()); b = bool((E0[own] == E1[own]).all()); ok &= a and b
        print("[X] %s isolation (other farm perturbed): temp unchanged %s, EC unchanged %s" % (farm, a, b))
    print("ALL PASS" if ok else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
