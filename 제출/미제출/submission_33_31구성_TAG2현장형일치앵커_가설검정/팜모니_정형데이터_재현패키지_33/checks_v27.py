# -*- coding: utf-8 -*-
"""submission_27 full-pipeline reproduction + rule checks (future-input cuts and other-farm isolation).
links       = linker base/t1/t3 on the real data (LINKER_WORK lw) and on a copy whose test_X inputs are ALL NaN (MASK world, lwm);
              온도/linker_v7/ap_fl.py train (MASK world: FL weights + training links) and ap_fl.py eval (real data: per-record hour-0 picks,
              fl_eval_anchor.csv).  No ap_v2 / claimed assignment, no v15R mirror geometry, no record-number or date-order features.
temperature = 온도/temp_candidate_nx27.csv (NX base = W40G-S without day/second; FIXED stored input — regenerate with
              온도/code/make_submission_v26_temp.py, ~35 min, not repeated per check) -> code/make_temp_v22_anchor.py (FL anchors)
              -> code/make_temp_v27_tx3.py (high-EC cold-row correction, gate from EC/output_27/ec_nh.csv + current in_temp)
EC          = EC/ec27_post.py (pre-SG2 stage + HR5 = NH, then FL yesterday-EC anchor)
Every step runs in its own Python process.  For each cut (farm, record d, hour h) every test_X input of that farm strictly after (d, h)
is multiplied by 11 and +777; that farm's predictions at or before (d, h) must be bit-identical.  Isolation: perturb all test_X inputs of
the other farm; own predictions unchanged.
Run from the package root:  python checks_v27.py <folder with the ORIGINAL public train_X.csv, train_y.csv, test_X.csv>
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
    w = tempfile.mkdtemp(prefix="c27chk_")
    lw = os.path.join(w, "lw"); lwm = os.path.join(w, "lwm"); fl = os.path.join(w, "fl"); T = os.path.join(ROOT, "온도")
    tfl = os.path.join(w, "t_fl.csv"); t27 = os.path.join(w, "t27.csv")
    mpub = masked_copy(pub, os.path.join(w, "maskpub"))
    for s in ("base.py", "t1.py", "t3.py"):
        script(os.path.join(T, "linker", s), [], {"LINKER_DATA": pub, "LINKER_WORK": lw})
        script(os.path.join(T, "linker", s), [], {"LINKER_DATA": mpub, "LINKER_WORK": lwm})
    script(os.path.join(T, "linker_v7", "ap_fl.py"), ["train", fl], {"LINKER_WORK": lwm})
    script(os.path.join(T, "linker_v7", "ap_fl.py"), ["eval", fl, os.path.join(fl, "w_fl.npy")], {"LINKER_WORK": lw})
    anc = os.path.join(fl, "fl_eval_anchor.csv")
    script(os.path.join(T, "code", "make_temp_v22_anchor.py"), [os.path.join(T, "temp_candidate_nx27.csv"), anc, tfl])
    ec_dir = os.path.join(w, "EC"); shutil.copytree(os.path.join(ROOT, "EC"), ec_dir, ignore=shutil.ignore_patterns(
        "output_18", "output_19", "output_20", "output_21", "output_23", "output_24", "output_25", "output_26", "output_27", "data", "dlls", "__pycache__"))
    os.makedirs(os.path.join(ec_dir, "data"))
    for f in ("train_X.csv", "test_X.csv", "sample_submission.csv"):
        src = os.path.join(pub, f) if os.path.exists(os.path.join(pub, f)) else os.path.join(ROOT, "EC", "data", f)
        shutil.copy(src, os.path.join(ec_dir, "data", f))
    pd.read_csv(os.path.join(pub, "train_y.csv"))[["row_id", "sub_ec"]].to_csv(os.path.join(ec_dir, "data", "train_y.csv"), index=False)
    script(os.path.join(ec_dir, "ec27_post.py"), [anc])
    script(os.path.join(T, "code", "make_temp_v27_tx3.py"), [tfl, os.path.join(ec_dir, "output_27", "ec_nh.csv"), os.path.join(pub, "test_X.csv"), t27])
    t = pd.read_csv(t27).set_index("row_id").sub_temp
    e = pd.read_csv(os.path.join(ec_dir, "output_27", "ec_nh_fl.csv")).set_index("row_id").sub_ec.reindex(t.index)
    shutil.rmtree(w, ignore_errors=True)
    return t, e, {}


def perturbed(pub, mask_fn):
    w = tempfile.mkdtemp(prefix="c27dat_")
    for f in ("train_X.csv", "train_y.csv", "sample_submission.csv"):
        if os.path.exists(os.path.join(pub, f)):
            shutil.copy(os.path.join(pub, f), os.path.join(w, f))
    X = pd.read_csv(os.path.join(pub, "test_X.csv")); m = mask_fn(X)
    X.loc[m, COLS] = X.loc[m, COLS] * 11 + 777; X.to_csv(os.path.join(w, "test_X.csv"), index=False)
    return w


def main():
    pub = sys.argv[1]
    T0, E0, K0 = run(pub); ok = True
    out_csv = os.path.join(ROOT, "submission_27_regen.csv")
    if not os.path.exists(out_csv):
        pd.DataFrame({"row_id": T0.index, "sub_temp": T0.values, "sub_ec": E0.values}).to_csv(out_csv, index=False, float_format="%.6f")
    r14 = pd.read_csv(os.path.join(ROOT, "submission_25.csv")).set_index("row_id")
    print("[0] regenerated: temp changed vs round-16 rows %d (RMS %.4f), EC changed vs round-16 rows %d (RMS %.4f)"
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
    sub = os.path.join(ROOT, "submission_27.csv")
    if os.path.exists(sub):
        S = pd.read_csv(sub).set_index("row_id")
        same = bool((np.abs(S.sub_temp.reindex(T0.index) - T0) < 5e-7).all() and (np.abs(S.sub_ec.reindex(T0.index) - E0) < 5e-7).all())
        print("[R] full pipeline reproduces submission_27.csv (6 decimals): %s" % same); ok &= same
    print("ALL PASS" if ok else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
