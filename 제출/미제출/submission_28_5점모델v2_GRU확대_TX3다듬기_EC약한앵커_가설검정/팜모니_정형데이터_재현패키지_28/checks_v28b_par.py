# -*- coding: utf-8 -*-
"""submission_28 rule checks — PARALLEL + GRU-CACHE driver (v28b).  Same six runs and comparisons as checks_v28.py, but the temperature GRU stage is
온도/code/make_temp_v28b_v3lf.py (= v28 + gru_cache_v28: trained weights reused ONLY when the SHA-256 of the MASK-world training sequences, targets,
weights and seed is identical; any leak of evaluation inputs into training changes the key and forces retraining).  The base run goes first and
fills the cache; the five perturbed runs then run concurrently.  Output lines are the same as checks_v28.py.
(Original parallel-driver description follows.)
checks_v28.py runs the full pipeline six times one after another (base, 3 future-input cuts, 2 other-farm isolations), ~12 min each.
This driver imports checks_v28.run / perturbed and runs the six pipelines concurrently in a thread pool (each run already uses its own temp
directory and starts every step as its own Python process; GRU training is CPU with 4 fixed threads -> max 4 runs at once on 16 cores).
Nothing in the pipelines or the comparisons changes.
Run from the package root:  python checks_v28b_par.py <folder with the ORIGINAL public train_X.csv, train_y.csv, test_X.csv>
"""
import os, sys, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import tempfile
import checks_v28 as C
CACHE = tempfile.mkdtemp(prefix='c28gru_')

CUTS = (("F13", 206, 4), ("F13", 237, 3), ("F47", 202, 5))
ISO = (("F13", "F47"), ("F47", "F13"))


def cut_mask(farm, d, h):
    def m(X):
        f, dy, hr = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
        return (f == farm) & ((dy > d) | ((dy == d) & (hr > h)))
    return m


def run_b(pub):
    w = tempfile.mkdtemp(prefix="c28bchk_")
    lw = os.path.join(w, "lw"); lwm = os.path.join(w, "lwm"); fl = os.path.join(w, "fl"); fln = os.path.join(w, "fln"); T = os.path.join(C.ROOT, "온도")
    tg = os.path.join(w, "t_g.csv"); uni = os.path.join(w, "uni.csv"); ta = os.path.join(w, "t_a.csv"); t28 = os.path.join(w, "t28.csv")
    mpub = C.masked_copy(pub, os.path.join(w, "maskpub"))
    for s_ in ("base.py", "t1.py", "t3.py"):
        C.script(os.path.join(T, "linker", s_), [], {"LINKER_DATA": pub, "LINKER_WORK": lw})
        C.script(os.path.join(T, "linker", s_), [], {"LINKER_DATA": mpub, "LINKER_WORK": lwm})
    C.script(os.path.join(T, "linker_v7", "ap_fl.py"), ["train", fl], {"LINKER_WORK": lwm})
    C.script(os.path.join(T, "linker_v7", "ap_fl.py"), ["eval", fl, os.path.join(fl, "w_fl.npy")], {"LINKER_WORK": lw})
    C.script(os.path.join(T, "linker_v7", "ap_fln.py"), ["train", fln], {"LINKER_WORK": lwm})
    C.script(os.path.join(T, "linker_v7", "ap_fln.py"), ["eval", fln, os.path.join(fln, "w_fln.npy")], {"LINKER_WORK": lw})
    anc = os.path.join(fl, "fl_eval_anchor.csv"); flnl = os.path.join(fln, "fln_eval_links.csv")
    C.script(os.path.join(T, "code", "make_temp_v28b_v3lf.py"), [os.path.join(T, "temp_candidate_nx27.csv"), os.path.join(fl, "fl_train_links.csv"), os.path.join(fl, "fl_eval_links.csv"), tg], {"AGRI_DATA": pub, "GRU_CACHE": CACHE})
    C.script(os.path.join(T, "code", "make_temp_v28_uni_anchor.py"), [anc, flnl, uni])
    C.script(os.path.join(T, "code", "make_temp_v22_anchor.py"), [tg, uni, ta])
    ec_dir = os.path.join(w, "EC"); shutil.copytree(os.path.join(C.ROOT, "EC"), ec_dir, ignore=shutil.ignore_patterns(
        "output_18", "output_19", "output_20", "output_21", "output_23", "output_24", "output_25", "output_26", "output_27", "output_28", "data", "dlls", "__pycache__"))
    os.makedirs(os.path.join(ec_dir, "data"))
    for f in ("train_X.csv", "test_X.csv", "sample_submission.csv"):
        src = os.path.join(pub, f) if os.path.exists(os.path.join(pub, f)) else os.path.join(C.ROOT, "EC", "data", f)
        shutil.copy(src, os.path.join(ec_dir, "data", f))
    pd.read_csv(os.path.join(pub, "train_y.csv"))[["row_id", "sub_ec"]].to_csv(os.path.join(ec_dir, "data", "train_y.csv"), index=False)
    C.script(os.path.join(ec_dir, "ec27_post.py"), [anc]); C.script(os.path.join(ec_dir, "ec28_post.py"), [anc, flnl])
    C.script(os.path.join(T, "code", "make_temp_v28_tx3.py"), [ta, os.path.join(ec_dir, "output_27", "ec_nh.csv"), os.path.join(pub, "test_X.csv"), t28])
    t = pd.read_csv(t28).set_index("row_id").sub_temp
    e = pd.read_csv(os.path.join(ec_dir, "output_28", "ec_nh_fl_kea.csv")).set_index("row_id").sub_ec.reindex(t.index)
    shutil.rmtree(w, ignore_errors=True)
    return t, e, {}


def job(kind, arg, pub):
    if kind == "base":
        return run_b(pub)
    w = C.perturbed(pub, cut_mask(*arg) if kind == "cut" else (lambda X, o=arg[1]: X.row_id.str[:3] == o))
    try:
        return run_b(w)
    finally:
        shutil.rmtree(w, ignore_errors=True)


def main():
    pub = sys.argv[1]
    jobs = [("base", None)] + [("cut", c) for c in CUTS] + [("iso", i) for i in ISO]
    res = {("base", None): job("base", None, pub)}; print("base run done (GRU cache filled)", flush=True)
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {j: ex.submit(job, j[0], j[1], pub) for j in jobs[1:]}
        res.update({j: f.result() for j, f in futs.items()})
    print("GRU cache files:", sorted(os.listdir(CACHE)), flush=True)
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
