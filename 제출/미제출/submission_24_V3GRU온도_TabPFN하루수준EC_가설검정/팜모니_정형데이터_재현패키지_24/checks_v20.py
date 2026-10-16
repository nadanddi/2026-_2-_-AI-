# -*- coding: utf-8 -*-
"""submission_20 rule checks (future-input cuts and other-farm isolation) for the full new pipelines:
temperature  = make_temp_v18_chain.py -> linker (base, t1, t3, ap_v2) -> linker_v5 ap_v5 -> make_temp_v19_anchor.py
EC           = ea1_post.py (anchors from linker_v5)
For each cut (farm, record d, hour h) every test_X input of that farm strictly after (d, h) is multiplied by 11 and +777; that
farm's predictions at or before (d, h) must be bit-identical.  Isolation: perturb all test_X inputs of the other farm.
Run from the package root:  python checks_v19.py <folder with the ORIGINAL public train_X.csv, train_y.csv, test_X.csv>"""
import os, sys, shutil, runpy, tempfile, contextlib, io
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
COLS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
        "act_heating", "act_circfan", "act_co2", "act_fog"]


def script(path, args, env=None):
    old = sys.argv; sys.argv = [path] + args; oldenv = dict(os.environ); os.environ.update(env or {})
    for m in ("base", "feat2", "common2", "c18post", "feat4"):
        sys.modules.pop(m, None)
    sys.path.insert(0, os.path.dirname(path))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(path, run_name="__main__")
    finally:
        sys.argv = old; os.environ.clear(); os.environ.update(oldenv); sys.path.pop(0)


def run(pub):
    w = tempfile.mkdtemp(prefix="c20chk_")
    t18 = os.path.join(w, "t18.csv"); t19 = os.path.join(w, "t19.csv"); lw = os.path.join(w, "lw")
    script(os.path.join(ROOT, "온도", "code", "make_temp_v18_chain.py"), [pub, os.path.join(ROOT, "온도", "data", "oof_diag10_tt1"),
           os.path.join(ROOT, "온도", "temp_candidate_v13.csv"), t18])
    for s in ("base.py", "t1.py", "t3.py", "ap_v2.py"):
        script(os.path.join(ROOT, "온도", "linker", s), [], {"LINKER_DATA": pub, "LINKER_WORK": lw})
    l5 = os.path.join(w, "l5")
    script(os.path.join(ROOT, "온도", "linker_v5", "ap_v5.py"), [], {"LINKER_WORK": lw, "LINKER5_OUT": l5})
    anc = os.path.join(l5, "eval60_confident_1step_anchor_v5.csv")
    script(os.path.join(ROOT, "온도", "code", "make_temp_v19_anchor.py"), [t18, anc, t19])
    ec_dir = os.path.join(w, "EC"); shutil.copytree(os.path.join(ROOT, "EC"), ec_dir, ignore=shutil.ignore_patterns("output_18", "output_19", "output_20", "data", "dlls", "*.ckpt"))
    os.makedirs(os.path.join(ec_dir, "data"))
    for f in ("train_X.csv", "test_X.csv", "sample_submission.csv"):
        src = os.path.join(pub, f) if os.path.exists(os.path.join(pub, f)) else os.path.join(ROOT, "EC", "data", f)
        shutil.copy(src, os.path.join(ec_dir, "data", f))
    pd.read_csv(os.path.join(pub, "train_y.csv"))[["row_id", "sub_ec"]].to_csv(os.path.join(ec_dir, "data", "train_y.csv"), index=False)
    script(os.path.join(ec_dir, "ea1_post.py"), [anc])
    t = pd.read_csv(t19).set_index("row_id").sub_temp
    e = pd.read_csv(os.path.join(ec_dir, "output_20", "ec_ea1.csv")).set_index("row_id").sub_ec.reindex(t.index)
    shutil.rmtree(w, ignore_errors=True)
    return t, e


def perturbed(pub, mask_fn):
    w = tempfile.mkdtemp(prefix="c20dat_")
    for f in ("train_X.csv", "train_y.csv"):
        shutil.copy(os.path.join(pub, f), os.path.join(w, f))
    X = pd.read_csv(os.path.join(pub, "test_X.csv")); m = mask_fn(X)
    X.loc[m, COLS] = X.loc[m, COLS] * 11 + 777; X.to_csv(os.path.join(w, "test_X.csv"), index=False)
    return w


def main():
    pub = sys.argv[1]
    T0, E0 = run(pub); ok = True
    for farm, d, h in (("F13", 206, 4), ("F13", 235, 3), ("F47", 202, 5)):
        def m(X):
            f, dy, hr = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
            return (f == farm) & ((dy > d) | ((dy == d) & (hr > h)))
        w = perturbed(pub, m); T1, E1 = run(w); shutil.rmtree(w, ignore_errors=True)
        ix = T0.index; ff, fd, fh = ix.str[:3], ix.str[4:7].astype(int), ix.str[8:10].astype(int)
        past = (ff == farm) & ((fd < d) | ((fd == d) & (fh <= h))); fut = (ff == farm) & ~past
        a = bool((T0[past] == T1[past]).all()); b = bool((E0[past] == E1[past]).all()); ok &= a and b
        print("[F] %s cut %d h%02d: temp past unchanged %s, EC past unchanged %s (past rows %d; future moved temp %.2f, EC %.3f)"
              % (farm, d, h, a, b, int(past.sum()), float(np.abs(T0[fut] - T1[fut]).sum()), float(np.abs(E0[fut] - E1[fut]).sum())), flush=True)
    for farm, other in (("F13", "F47"), ("F47", "F13")):
        w = perturbed(pub, lambda X: X.row_id.str[:3] == other); T1, E1 = run(w); shutil.rmtree(w, ignore_errors=True)
        own = T0.index.str[:3] == farm
        a = bool((T0[own] == T1[own]).all()); b = bool((E0[own] == E1[own]).all()); ok &= a and b
        print("[X] %s isolation: temp unchanged %s, EC unchanged %s" % (farm, a, b), flush=True)
    sub = pd.read_csv(os.path.join(ROOT, "submission_20.csv")).set_index("row_id") if os.path.exists(os.path.join(ROOT, "submission_20.csv")) else None
    if sub is not None:
        same = bool((np.abs(sub.sub_temp.reindex(T0.index) - T0) < 5e-7).all() and (np.abs(sub.sub_ec.reindex(T0.index) - E0) < 5e-7).all())
        print("[R] full pipeline reproduces submission_20.csv (6 decimals): %s" % same); ok &= same
    print("ALL PASS" if ok else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
