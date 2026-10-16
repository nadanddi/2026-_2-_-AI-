# -*- coding: utf-8 -*-
"""submission_25 full-pipeline reproduction + rule checks (future-input cuts and other-farm isolation).
links       = linker (base, t1, t3, ap_v2) on the real data -> linker_v15r/v15r_eval_pkg.py (evaluation picks);
              the same linker chain on a copy whose test_X inputs are ALL NaN (MASK world) -> linker_v15r/v15r_mirror_pkg.py (training links)
temperature = code/make_temp_v25_v3lbase.py (P9 + V3L GRU member, same-house previous day window) -> make_temp_v18_chain.py -> linker_v7 ap_v7
              -> make_temp_v22_anchor.py -> code/make_temp_v23_tna8.py
EC          = EC/f5e_post.py (pre-SG2 stage + HR5 + EV3L; SG2/W2/B5 removed)
Every step runs in its own Python process (no module cache between runs).  For each cut (farm, record d, hour h) every test_X input of that
farm strictly after (d, h) is multiplied by 11 and +777; that farm's predictions at or before (d, h) must be bit-identical.  Isolation: perturb
all test_X inputs of the other farm; own predictions unchanged.  (temp_candidate_v13.csv = round-9 base, fixed input as in checks_v23.)
Run from the package root:  python checks_v24.py <folder with the ORIGINAL public train_X.csv, train_y.csv, test_X.csv>
"""
import os, sys, shutil, subprocess, tempfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
COLS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
        "act_heating", "act_circfan", "act_co2", "act_fog"]


def script(path, args, env=None):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run([sys.executable, "-B", path] + args, cwd=os.path.dirname(path), env=e, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("%s failed:\n%s" % (path, r.stderr[-3000:]))


def masked_copy(pub, w):
    os.makedirs(w)
    for f in ("train_X.csv", "train_y.csv", "sample_submission.csv"):
        if os.path.exists(os.path.join(pub, f)):
            shutil.copy(os.path.join(pub, f), os.path.join(w, f))
    X = pd.read_csv(os.path.join(pub, "test_X.csv")); cols = [c for c in X.columns if c != "row_id"]; X[cols] = np.nan
    X.to_csv(os.path.join(w, "test_X.csv"), index=False)
    return w


def run(pub):
    w = tempfile.mkdtemp(prefix="c25chk_")
    tb = os.path.join(w, "tb.csv"); t18 = os.path.join(w, "t18.csv"); t22 = os.path.join(w, "t22.csv"); t25 = os.path.join(w, "t25.csv")
    lw = os.path.join(w, "lw"); lwm = os.path.join(w, "lwm"); l5 = os.path.join(w, "l5"); T = os.path.join(ROOT, "온도")
    mpub = masked_copy(pub, os.path.join(w, "maskpub"))
    for s in ("base.py", "t1.py", "t3.py", "ap_v2.py"):
        script(os.path.join(T, "linker", s), [], {"LINKER_DATA": pub, "LINKER_WORK": lw})
        script(os.path.join(T, "linker", s), [], {"LINKER_DATA": mpub, "LINKER_WORK": lwm})
    pk = os.path.join(w, "v15r_eval.csv"); pm = os.path.join(w, "v15r_mirror_masked.csv")
    script(os.path.join(T, "linker_v15r", "v15r_eval_pkg.py"), [pk], {"LINKER_WORK": lw, "LINKER_DATA": pub})
    script(os.path.join(T, "linker_v15r", "v15r_mirror_pkg.py"), [pm], {"LINKER_WORK": lwm, "LINKER_DATA": mpub})
    script(os.path.join(T, "code", "make_temp_v25_v3lbase.py"), [os.path.join(T, "temp_candidate_v13.csv"), pm, pk, tb], {"AGRI_DATA": pub})
    script(os.path.join(T, "code", "make_temp_v18_chain.py"), [pub, os.path.join(T, "data", "oof_diag10_tt1"), tb, t18])
    script(os.path.join(T, "linker_v7", "ap_v7.py"), [], {"LINKER_WORK": lw, "LINKER5_OUT": l5})
    anc = os.path.join(l5, "eval60_confident_1step_anchor_v7.csv")
    script(os.path.join(T, "code", "make_temp_v22_anchor.py"), [t18, anc, t22])
    script(os.path.join(T, "code", "make_temp_v23_tna8.py"), [t22, pk, anc, t25], {"LINKER_WORK": lw, "LINKER_DATA": pub})
    ec_dir = os.path.join(w, "EC"); shutil.copytree(os.path.join(ROOT, "EC"), ec_dir, ignore=shutil.ignore_patterns(
        "output_18", "output_19", "output_20", "output_21", "output_23", "output_24", "output_25", "data", "dlls", "__pycache__"))
    os.makedirs(os.path.join(ec_dir, "data"))
    for f in ("train_X.csv", "test_X.csv", "sample_submission.csv"):
        src = os.path.join(pub, f) if os.path.exists(os.path.join(pub, f)) else os.path.join(ROOT, "EC", "data", f)
        shutil.copy(src, os.path.join(ec_dir, "data", f))
    pd.read_csv(os.path.join(pub, "train_y.csv"))[["row_id", "sub_ec"]].to_csv(os.path.join(ec_dir, "data", "train_y.csv"), index=False)
    script(os.path.join(ec_dir, "f5e_post.py"), [pm, pk])
    t = pd.read_csv(t25).set_index("row_id").sub_temp
    e = pd.read_csv(os.path.join(ec_dir, "output_25", "ec_f5e.csv")).set_index("row_id").sub_ec.reindex(t.index)
    shutil.rmtree(w, ignore_errors=True)
    return t, e, {}


def perturbed(pub, mask_fn):
    w = tempfile.mkdtemp(prefix="c25dat_")
    for f in ("train_X.csv", "train_y.csv", "sample_submission.csv"):
        if os.path.exists(os.path.join(pub, f)):
            shutil.copy(os.path.join(pub, f), os.path.join(w, f))
    X = pd.read_csv(os.path.join(pub, "test_X.csv")); m = mask_fn(X)
    X.loc[m, COLS] = X.loc[m, COLS] * 11 + 777; X.to_csv(os.path.join(w, "test_X.csv"), index=False)
    return w


def main():
    pub = sys.argv[1]
    T0, E0, K0 = run(pub); ok = True
    out_csv = os.path.join(ROOT, "submission_25_regen.csv")
    if not os.path.exists(out_csv):
        pd.DataFrame({"row_id": T0.index, "sub_temp": T0.values, "sub_ec": E0.values}).to_csv(out_csv, index=False, float_format="%.6f")
    r14 = pd.read_csv(os.path.join(ROOT, "submission_24.csv")).set_index("row_id")
    print("[0] regenerated: temp changed vs round-15 rows %d (RMS %.4f), EC changed vs round-15 rows %d (RMS %.4f)"
          % (int((np.abs(T0 - r14.sub_temp.reindex(T0.index)) > 5e-7).sum()), float(np.sqrt(((T0 - r14.sub_temp.reindex(T0.index)) ** 2).mean())),
             int((np.abs(E0 - r14.sub_ec.reindex(T0.index)) > 5e-7).sum()), float(np.sqrt(((E0 - r14.sub_ec.reindex(T0.index)) ** 2).mean()))), flush=True)
    for farm, d, h in (("F13", 206, 4), ("F13", 237, 3), ("F47", 202, 5)):
        def m(X):
            f, dy, hr = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
            return (f == farm) & ((dy > d) | ((dy == d) & (hr > h)))
        w = perturbed(pub, m); T1, E1, _ = run(w); shutil.rmtree(w, ignore_errors=True)
        ix = T0.index; ff, fd, fh = ix.str[:3], ix.str[4:7].astype(int), ix.str[8:10].astype(int)
        past = (ff == farm) & ((fd < d) | ((fd == d) & (fh <= h))); fut = (ff == farm) & ~past
        a = bool((T0[past] == T1[past]).all()); b = bool((E0[past] == E1[past]).all()); ok &= a and b
        print("[F] %s cut %d h%02d: temp past unchanged %s, EC past unchanged %s (past rows %d; future moved temp %.2f, EC %.3f)"
              % (farm, d, h, a, b, int(past.sum()), float(np.abs(T0[fut] - T1[fut]).sum()), float(np.abs(E0[fut] - E1[fut]).sum())), flush=True)
    for farm, other in (("F13", "F47"), ("F47", "F13")):
        w = perturbed(pub, lambda X: X.row_id.str[:3] == other); T1, E1, _ = run(w); shutil.rmtree(w, ignore_errors=True)
        own = T0.index.str[:3] == farm
        a = bool((T0[own] == T1[own]).all()); b = bool((E0[own] == E1[own]).all()); ok &= a and b
        print("[X] %s isolation: temp unchanged %s, EC unchanged %s" % (farm, a, b), flush=True)
    sub = os.path.join(ROOT, "submission_25.csv")
    if os.path.exists(sub):
        S = pd.read_csv(sub).set_index("row_id")
        same = bool((np.abs(S.sub_temp.reindex(T0.index) - T0) < 5e-7).all() and (np.abs(S.sub_ec.reindex(T0.index) - E0) < 5e-7).all())
        print("[R] full pipeline reproduces submission_25.csv (6 decimals): %s" % same); ok &= same
    print("ALL PASS" if ok else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
