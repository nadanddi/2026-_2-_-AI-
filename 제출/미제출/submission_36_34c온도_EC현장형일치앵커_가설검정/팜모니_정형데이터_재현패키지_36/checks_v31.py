# -*- coding: utf-8 -*-
"""submission_31 full-pipeline (= submission_29 with the prediction-chain threshold p0 >= .5, make_temp_v31_chain.py) production / reproduction + rule checks (future-input cuts and other-farm isolation), runs in PARALLEL.
links       = linker base/t1/t3 on the real data (lw) and on a copy whose test_X inputs are ALL NaN (MASK world, lwm);
              온도/linker_v7/ap_fl.py train (MASK world) and ap_fl.py eval (real data: per-record hour-0 picks -> fl_eval_links.csv, fl_eval_anchor.csv).
temperature = 온도/temp_candidate_nx27.csv (NX base, FIXED stored input as in round 27) -> code/make_temp_v22_anchor.py (FL anchors)
              -> code/make_temp_v27_tx3.py (high-EC cold-row correction) -> code/make_temp_v29_chain.py (time-decaying prediction chain M1)
EC          = EC/ec29_post.py (round-27 NH, then FL yesterday-EC anchor x 0.8 x p0)
Every step runs in its own Python process.  Cuts: for (farm, record d, hour h) every test_X input of that farm strictly after (d, h) is
multiplied by 11 and +777; that farm's predictions at or before (d, h) must be bit-identical.  Isolation: perturb all test_X inputs of the
other farm; own predictions unchanged.  Base + 3 cuts + 2 isolation runs are launched together (thread pool).
Run from the package root:
  python checks_v31.py produce <folder with the ORIGINAL public train_X.csv, train_y.csv, test_X.csv>   -> submission_29.csv (+ intermediates)
  python checks_v31.py check   <same folder>                                                              -> checks (ALL PASS) incl. reproduction
"""
import os, sys, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
COLS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
        "act_heating", "act_circfan", "act_co2", "act_fog"]
CUTS = (("F13", 206, 4), ("F13", 237, 3), ("F47", 202, 5)); ISO = (("F13", "F47"), ("F47", "F13"))


def script(path, args, env=None):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run([sys.executable, "-B", path] + args, cwd=os.path.dirname(path), env=e, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("%s failed:\n%s" % (path, r.stderr[-3000:]))
    return r.stdout


def masked_copy(pub, w):
    os.makedirs(w)
    for f in ("train_X.csv", "train_y.csv", "sample_submission.csv"):
        if os.path.exists(os.path.join(pub, f)):
            shutil.copy(os.path.join(pub, f), os.path.join(w, f))
    X = pd.read_csv(os.path.join(pub, "test_X.csv")); cols = [c for c in X.columns if c != "row_id"]; X[cols] = np.nan
    X.to_csv(os.path.join(w, "test_X.csv"), index=False)
    return w


def run(pub, keep=None):
    w = tempfile.mkdtemp(prefix="c31chk_")
    lw = os.path.join(w, "lw"); lwm = os.path.join(w, "lwm"); fl = os.path.join(w, "fl"); T = os.path.join(ROOT, "온도")
    tfl = os.path.join(w, "t_fl.csv"); t27 = os.path.join(w, "t27.csv"); t29 = os.path.join(w, "t29.csv"); log = []
    mpub = masked_copy(pub, os.path.join(w, "maskpub"))
    for s in ("base.py", "t1.py", "t3.py"):
        script(os.path.join(T, "linker", s), [], {"LINKER_DATA": pub, "LINKER_WORK": lw})
        script(os.path.join(T, "linker", s), [], {"LINKER_DATA": mpub, "LINKER_WORK": lwm})
    script(os.path.join(T, "linker_v7", "ap_fl.py"), ["train", fl], {"LINKER_WORK": lwm})
    log.append(script(os.path.join(T, "linker_v7", "ap_fl.py"), ["eval", fl, os.path.join(fl, "w_fl.npy")], {"LINKER_WORK": lw}))
    anc = os.path.join(fl, "fl_eval_anchor.csv"); lnk = os.path.join(fl, "fl_eval_links.csv"); nx = os.path.join(T, "temp_candidate_nx27.csv")
    log.append(script(os.path.join(T, "code", "make_temp_v22_anchor.py"), [nx, anc, tfl]))
    ec_dir = os.path.join(w, "EC"); shutil.copytree(os.path.join(ROOT, "EC"), ec_dir, ignore=shutil.ignore_patterns(
        "output_18", "output_19", "output_20", "output_21", "output_23", "output_24", "output_25", "output_26", "output_27", "output_29", "data", "__pycache__"))
    os.makedirs(os.path.join(ec_dir, "data"))
    for f in ("train_X.csv", "test_X.csv", "sample_submission.csv"):
        src = os.path.join(pub, f) if os.path.exists(os.path.join(pub, f)) else os.path.join(ROOT, "EC", "data", f)
        shutil.copy(src, os.path.join(ec_dir, "data", f))
    pd.read_csv(os.path.join(pub, "train_y.csv"))[["row_id", "sub_ec"]].to_csv(os.path.join(ec_dir, "data", "train_y.csv"), index=False)
    log.append(script(os.path.join(ec_dir, "ec29_post.py"), [anc]))
    log.append(script(os.path.join(T, "code", "make_temp_v27_tx3.py"), [tfl, os.path.join(ec_dir, "output_29", "ec_nh.csv"), os.path.join(pub, "test_X.csv"), t27]))
    log.append(script(os.path.join(T, "code", "make_temp_v31_chain.py"), [t27, nx, lnk, anc, t29]))
    t = pd.read_csv(t29).set_index("row_id").sub_temp
    e = pd.read_csv(os.path.join(ec_dir, "output_29", "ec_nh_fl.csv")).set_index("row_id").sub_ec.reindex(t.index)
    if keep:
        os.makedirs(keep, exist_ok=True)
        for src, dst in ((lnk, "fl_eval_links.csv"), (anc, "fl_eval_anchor.csv"), (tfl, "temp_after_anchor.csv"), (t27, "temp_after_tx3.csv"), (t29, "temp_candidate_v31.csv"),
                         (os.path.join(ec_dir, "output_29", "ec_nh.csv"), "ec_nh.csv"), (os.path.join(ec_dir, "output_29", "ec_nh_fl.csv"), "ec_nh_fl.csv")):
            shutil.copy(src, os.path.join(keep, dst))
        open(os.path.join(keep, "produce_v31.log"), "w", encoding="utf-8").write("\n".join(log))
    shutil.rmtree(w, ignore_errors=True)
    return t, e


def perturbed(pub, mask_fn):
    w = tempfile.mkdtemp(prefix="c31dat_")
    for f in ("train_X.csv", "train_y.csv", "sample_submission.csv"):
        if os.path.exists(os.path.join(pub, f)):
            shutil.copy(os.path.join(pub, f), os.path.join(w, f))
    X = pd.read_csv(os.path.join(pub, "test_X.csv")); m = mask_fn(X)
    X.loc[m, COLS] = X.loc[m, COLS] * 11 + 777; X.to_csv(os.path.join(w, "test_X.csv"), index=False)
    return w


def job(kind, arg, pub):
    if kind == "base": return run(pub)
    if kind == "cut":
        farm, d, h = arg
        def m(X):
            f, dy, hr = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
            return (f == farm) & ((dy > d) | ((dy == d) & (hr > h)))
    else:
        farm, other = arg; m = lambda X: X.row_id.str[:3] == other
    w = perturbed(pub, m)
    try: return run(w)
    finally: shutil.rmtree(w, ignore_errors=True)


def produce(pub):
    out = os.path.join(ROOT, "submission_31.csv"); assert not os.path.exists(out), "refusing to overwrite " + out
    keep = os.path.join(ROOT, "output_31"); t, e = run(pub, keep=keep)
    script(os.path.join(ROOT, "merge_temp_ec_v29.py"), [os.path.join(keep, "temp_candidate_v31.csv"), os.path.join(keep, "ec_nh_fl.csv"), out])
    print(open(os.path.join(keep, "produce_v31.log"), encoding="utf-8").read()); print("wrote", out)


def check(pub):
    jobs = [("base", None)] + [("cut", c) for c in CUTS] + [("iso", i) for i in ISO]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {j: ex.submit(job, j[0], j[1], pub) for j in jobs}
        res = {j: f.result() for j, f in futs.items()}
    T0, E0 = res[("base", None)]; ok = True
    r = pd.read_csv(os.path.join(ROOT, "submission_29.csv")).set_index("row_id")
    print("[0] vs round-19 (submission_29): temp changed rows %d (RMS %.4f), EC changed rows %d (RMS %.4f)"
          % (int((np.abs(T0 - r.sub_temp.reindex(T0.index)) > 5e-7).sum()), float(np.sqrt(((T0 - r.sub_temp.reindex(T0.index)) ** 2).mean())),
             int((np.abs(E0 - r.sub_ec.reindex(T0.index)) > 5e-7).sum()), float(np.sqrt(((E0 - r.sub_ec.reindex(T0.index)) ** 2).mean()))), flush=True)
    ix = T0.index; ff, fd, fh = ix.str[:3], ix.str[4:7].astype(int), ix.str[8:10].astype(int)
    for farm, d, h in CUTS:
        T1, E1 = res[("cut", (farm, d, h))]
        past = (ff == farm) & ((fd < d) | ((fd == d) & (fh <= h))); fut = (ff == farm) & ~past
        a = bool((T0[past] == T1[past]).all()); b = bool((E0[past] == E1[past]).all()); ok &= a and b
        print("[F] %s cut %d h%02d: temp past unchanged %s, EC past unchanged %s (past rows %d; future moved temp %.2f, EC %.3f)"
              % (farm, d, h, a, b, int(past.sum()), float(np.abs(T0[fut] - T1[fut]).sum()), float(np.abs(E0[fut] - E1[fut]).sum())), flush=True)
    for farm, other in ISO:
        T1, E1 = res[("iso", (farm, other))]; own = ff == farm
        a = bool((T0[own] == T1[own]).all()); b = bool((E0[own] == E1[own]).all()); ok &= a and b
        print("[X] %s isolation: temp unchanged %s, EC unchanged %s" % (farm, a, b), flush=True)
    S = pd.read_csv(os.path.join(ROOT, "submission_31.csv")).set_index("row_id")
    same = bool((np.abs(S.sub_temp.reindex(T0.index) - T0) < 5e-7).all() and (np.abs(S.sub_ec.reindex(T0.index) - E0) < 5e-7).all())
    print("[R] full pipeline reproduces submission_31.csv (6 decimals): %s" % same); ok &= same
    print("ALL PASS" if ok else "SOME CHECK FAILED")


if __name__ == "__main__":
    {"produce": produce, "check": check}[sys.argv[1]](sys.argv[2])
