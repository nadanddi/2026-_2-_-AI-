# -*- coding: utf-8 -*-
"""Full retraining of the final answer submission_35e from the original competition CSVs (final verification, 2026-10-17, team 팜모니).

code/pipeline_35e/checks_v35e.py produce rebuilds submission_35e from the original CSVs, but it reads four stored intermediate files.
This script rebuilds those files from the original CSVs first and then runs the same produce step on them:
  1. EC round-9 members   EC/output_14/prediction_details.npz           <- EC/model.py (via EC/reproduce.py)
     season tables        온도/code/season_ec/{training,evaluation}_day_to_season.csv  <- same EC/model.py run (training-only transform)
  2. temperature NX base  온도/temp_candidate_nx27.csv                   <- 온도/code/make_submission_v26_temp.py
  3. HR5 training base    EC/hr5_train_base_nosg_oof.csv                 <- retrain/hr5_base_oof/집/클로드/research/make_hr5_base_oof_v1.py
  4. checks_v35e.py produce on the rebuilt copy -> submission_35e.csv, compared with submission/submission_35e.csv.
Each rebuilt file is compared with the stored one before it replaces it (report: <work>/full_retrain_report.txt).
Nothing in code/ is modified; everything is written under <work>.

Run:  python full_retrain.py <folder with the ORIGINAL train_X.csv, train_y.csv, test_X.csv, sample_submission.csv> <new work folder>
Time: about 1.5-2 hours on an 8-core CPU (TabPFN on CPU dominates).
"""
import os, sys, shutil, subprocess, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE); PKG = os.path.dirname(CODE)
PIPE = os.path.join(CODE, "pipeline_35e"); MODELS = os.path.join(PKG, "models")
CKPT = os.path.join(MODELS, "tabpfn-v2-regressor.ckpt")
FINAL = os.path.join(PKG, "submission", "submission_35e.csv")
PUB, WORK = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
assert not os.path.exists(WORK), "work folder must be new: " + WORK
os.makedirs(WORK); REPORT = os.path.join(WORK, "full_retrain_report.txt"); T0 = time.time()


def say(s):
    line = "[%5.0fs] %s" % (time.time() - T0, s); print(line, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f: f.write(line + "\n")


def run(args, cwd, env_extra=None, log=None):
    env = dict(os.environ); env.update(env_extra or {})
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "-B", "-u"] + args, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError("failed: %s (see %s)" % (" ".join(args), log))


def cmp_csv(new, old, key, col):
    a = pd.read_csv(new).set_index(key)[col]; b = pd.read_csv(old).set_index(key)[col].reindex(a.index); d = (a - b).to_numpy(float)
    return "rows %d, max |diff| %.3e, RMS diff %.3e, rows differing at 6 decimals %d" % (len(d), np.nanmax(np.abs(d)), np.sqrt(np.nanmean(d ** 2)), int((np.abs(d) > 5e-7).sum()))


def main():
    for f in ("train_X.csv", "train_y.csv", "test_X.csv", "sample_submission.csv"):
        assert os.path.exists(os.path.join(PUB, f)), "missing " + f
    W = os.path.join(WORK, "pipeline_35e"); shutil.copytree(PIPE, W, ignore=shutil.ignore_patterns("__pycache__", "submission_35e.csv", "output_35e"))
    say("copied pipeline_35e -> %s" % W)
    # EC/model.py expects F13/F47 rows only in train_X / train_y (the round-9 EC package shipped that extract): copy those lines verbatim.
    ecd = os.path.join(W, "EC", "data")
    for f in ("train_X.csv", "train_y.csv"):
        with open(os.path.join(PUB, f), encoding="utf-8-sig") as src, open(os.path.join(ecd, f), "w", encoding="utf-8", newline="") as dst:
            head = src.readline(); dst.write(head); n = 0
            for line in src:
                if line[:3] in ("F13", "F47"): dst.write(line); n += 1
        say("EC/data/%s: F13/F47 lines %d" % (f, n))
    shutil.copy(os.path.join(PUB, "test_X.csv"), os.path.join(ecd, "test_X.csv"))
    shutil.copy(os.path.join(PUB, "sample_submission.csv"), os.path.join(ecd, "sample_submission.csv"))

    # 1. EC members + season tables
    o14 = os.path.join(WORK, "retrain_ec14")
    run(["reproduce.py", "--output", o14], os.path.join(W, "EC"), log=os.path.join(WORK, "retrain_ec14.log"))
    zn, zo = np.load(os.path.join(o14, "prediction_details.npz")), np.load(os.path.join(W, "EC", "output_14", "prediction_details.npz"))
    for k in ("raw_r3_by_seed", "raw_pfn_by_context"):
        d = zn[k] - zo[k]; say("1. EC %s vs stored: max |diff| %.3e, RMS %.3e" % (k, np.abs(d).max(), np.sqrt((d ** 2).mean())))
    assert (zn["row_id"] == zo["row_id"]).all()
    shutil.copy(os.path.join(o14, "prediction_details.npz"), os.path.join(W, "EC", "output_14", "prediction_details.npz"))
    for n in ("training_day_to_season.csv", "evaluation_day_to_season.csv"):
        new, old = os.path.join(o14, n), os.path.join(W, "온도", "code", "season_ec", n)
        a, b = pd.read_csv(new).set_index(["farm", "day"]).season, pd.read_csv(old).set_index(["farm", "day"]).season
        say("1. season table %s vs stored: rows %d/%d, max |diff| %.3e" % (n, len(a), len(b), float(np.abs(a - b.reindex(a.index)).max())))
        shutil.copy(new, old)

    # 2. temperature NX base (TabPFN weights = models/tabpfn-v2-regressor.ckpt through the tabpfn cache-dir setting)
    nx = os.path.join(WORK, "retrain_nx.csv")
    run(["make_submission_v26_temp.py", nx], os.path.join(W, "온도", "code"), {"AGRI_DATA": PUB, "TABPFN_MODEL_CACHE_DIR": MODELS}, log=os.path.join(WORK, "retrain_nx.log"))
    say("2. temperature NX vs stored temp_candidate_nx27.csv: " + cmp_csv(nx, os.path.join(W, "온도", "temp_candidate_nx27.csv"), "row_id", "sub_temp"))
    shutil.copy(nx, os.path.join(W, "온도", "temp_candidate_nx27.csv"))

    # 3. HR5 training base (DIAG10 out-of-fold, training data only)
    hb = os.path.join(WORK, "retrain_hr5_base.csv")
    shutil.copytree(os.path.join(HERE, "hr5_base_oof"), os.path.join(WORK, "hr5_base_oof"), ignore=shutil.ignore_patterns("__pycache__"))  # its modules create folders next to themselves
    tree = os.path.join(WORK, "hr5_base_oof", "집", "클로드", "research")
    run(["make_hr5_base_oof_v1.py", CKPT, hb, os.path.join(W, "EC", "hr5_train_base_nosg_oof.csv")], tree,
        {"AGRI_DATA": PUB, "HR5_LOCAL": os.path.join(WORK, "hr5_local")}, log=os.path.join(WORK, "retrain_hr5_base.log"))
    say("3. HR5 base vs stored hr5_train_base_nosg_oof.csv: " + cmp_csv(hb, os.path.join(W, "EC", "hr5_train_base_nosg_oof.csv"), "row_id", "base"))
    shutil.copy(hb, os.path.join(W, "EC", "hr5_train_base_nosg_oof.csv"))

    # 4. produce on the rebuilt files
    run(["checks_v35e.py", "produce", PUB], W, log=os.path.join(WORK, "produce_v35e.log"))
    out = os.path.join(W, "submission_35e.csv")
    for col in ("sub_temp", "sub_ec"):
        say("4. retrained submission_35e %s vs submitted: %s" % (col, cmp_csv(out, FINAL, "row_id", col)))
    say("done")


if __name__ == "__main__":
    main()
