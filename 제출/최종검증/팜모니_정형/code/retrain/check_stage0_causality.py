# -*- coding: utf-8 -*-
"""Rule checks for the two retrained stages that read test_X (final verification, 2026-10-17, team 팜모니):
    temperature NX base (온도/code/make_submission_v26_temp.py)  and  EC round-9 members (EC/model.py via EC/reproduce.py).
checks_v35e.py check covers every step AFTER these stored intermediates; this script covers the intermediates themselves.
(The third intermediate, EC/hr5_train_base_nosg_oof.csv, is built from train_X / train_y only.)

Same perturbations as checks_v35e.py:
  cut  (farm, record d, hour h): every test_X input of that farm strictly after (d, h) is multiplied by 11 and +777
       -> that farm's predictions at or before (d, h) must be bit-identical to the unperturbed run;
  iso  farm: every test_X input of the OTHER farm is perturbed the same way -> this farm's predictions must be bit-identical.
Compared arrays: NX sub_temp; EC raw_r3_by_seed and raw_pfn_by_context (what the final pipeline reads).

Run:  python check_stage0_causality.py <folder with the ORIGINAL CSVs> <new work folder>      (about 5 x 1 hour on an 8-core CPU)
"""
import os, sys, shutil, subprocess, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); PIPE = os.path.join(os.path.dirname(HERE), "pipeline_35e")
MODELS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "models")
PUB, WORK = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
assert not os.path.exists(WORK), "work folder must be new: " + WORK
os.makedirs(WORK); REPORT = os.path.join(WORK, "check_stage0_report.txt"); T0 = time.time()
COLS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
        "act_heating", "act_circfan", "act_co2", "act_fog"]
CUTS = (("F13", 206, 4), ("F13", 237, 3), ("F47", 202, 5)); ISO = ("F13", "F47")


def say(s):
    line = "[%5.0fs] %s" % (time.time() - T0, s); print(line, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(line + "\n")


def data_copy(name, mask_fn):
    d = os.path.join(WORK, "data_" + name); os.makedirs(d)
    for f in ("train_X.csv", "train_y.csv", "sample_submission.csv"): shutil.copy(os.path.join(PUB, f), os.path.join(d, f))
    X = pd.read_csv(os.path.join(PUB, "test_X.csv"))
    if mask_fn is not None:
        m = mask_fn(X); X.loc[m, COLS] = X.loc[m, COLS] * 11 + 777
    X.to_csv(os.path.join(d, "test_X.csv"), index=False)
    return d


def run(args, cwd, env_extra, log):
    env = dict(os.environ); env.update(env_extra or {})
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "-B", "-u"] + args, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0: raise RuntimeError("failed: %s (see %s)" % (" ".join(args), log))


def stage0(name, data):
    """NX + EC members on one data folder; returns (nx Series by row_id, dict of EC arrays, EC row_id)."""
    W = os.path.join(WORK, "pipe_" + name); shutil.copytree(PIPE, W, ignore=shutil.ignore_patterns("__pycache__", "submission_35e.csv", "output_35e"))
    ecd = os.path.join(W, "EC", "data")
    for f in ("train_X.csv", "train_y.csv"):          # EC/model.py reads F13/F47 rows only (same extract as full_retrain.py)
        with open(os.path.join(data, f), encoding="utf-8-sig") as src, open(os.path.join(ecd, f), "w", encoding="utf-8", newline="") as dst:
            dst.write(src.readline())
            for line in src:
                if line[:3] in ("F13", "F47"): dst.write(line)
    shutil.copy(os.path.join(data, "test_X.csv"), os.path.join(ecd, "test_X.csv"))
    o14 = os.path.join(WORK, "ec14_" + name)
    run(["reproduce.py", "--output", o14], os.path.join(W, "EC"), None, os.path.join(WORK, "ec14_%s.log" % name))
    nx = os.path.join(WORK, "nx_%s.csv" % name)
    run(["make_submission_v26_temp.py", nx], os.path.join(W, "온도", "code"), {"AGRI_DATA": data, "TABPFN_MODEL_CACHE_DIR": MODELS}, os.path.join(WORK, "nx_%s.log" % name))
    z = np.load(os.path.join(o14, "prediction_details.npz"))
    shutil.rmtree(W, ignore_errors=True)
    return pd.read_csv(nx).set_index("row_id").sub_temp, {k: z[k] for k in ("raw_r3_by_seed", "raw_pfn_by_context")}, z["row_id"]


def main():
    farm = lambda X: X.row_id.str[:3]; day = lambda X: X.row_id.str[4:7].astype(int); hr = lambda X: X.row_id.str[8:10].astype(int)
    T0n, E0, rid = stage0("base", data_copy("base", None)); say("base run done")
    ix = pd.Index(rid); ff, fd, fh = ix.str[:3], ix.str[4:7].astype(int), ix.str[8:10].astype(int); ok = True
    jobs = [("cut_%s_%d_%02d" % c, c, lambda X, c=c: (farm(X) == c[0]) & ((day(X) > c[1]) | ((day(X) == c[1]) & (hr(X) > c[2])))) for c in CUTS]
    jobs += [("iso_%s" % f, f, lambda X, f=f: farm(X) != f) for f in ISO]
    for name, arg, mfn in jobs:
        T1, E1, rid1 = stage0(name, data_copy(name, mfn)); assert (rid1 == rid).all()
        if name.startswith("cut"):
            f, d, h = arg; keep = np.asarray((ff == f) & ((fd < d) | ((fd == d) & (fh <= h)))); moved = ~keep & np.asarray(ff == f)
        else:
            keep = np.asarray(ff == arg); moved = None
        a = bool((T0n.reindex(ix)[keep].values == T1.reindex(ix)[keep].values).all())
        b = all(bool((E0[k][..., keep] == E1[k][..., keep]).all()) for k in E0)
        ok &= a and b
        extra = "" if moved is None else " (perturbed-side rows %d moved: temp %.2f, EC r3 %.3f)" % (
            int(moved.sum()), float(np.abs(T0n.reindex(ix)[moved].values - T1.reindex(ix)[moved].values).sum()),
            float(np.abs(E0["raw_r3_by_seed"][..., moved] - E1["raw_r3_by_seed"][..., moved]).sum()))
        say("%s: NX kept rows unchanged %s, EC members kept rows unchanged %s, kept rows %d%s" % (name, a, b, int(keep.sum()), extra))
    say("ALL PASS" if ok else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
