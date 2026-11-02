# -*- coding: utf-8 -*-
"""FL linker counterfactual check (final verification, 2026-10-20, team 팜모니).  Descriptive only: nothing here changes the final answer.

Question: how much do the evaluation 'yesterday' decisions depend on (a) the `excl` feature (candidate already has a confirmed next day
in the TRAINING labels) and (b) the presence of earlier evaluation records among the candidates?
Variants, all with the same FL weights (fitted in the MASK world exactly as checks_v35e.py does):
  base      : ap_fl.py eval as submitted (training records + same-farm earlier records, training or evaluation)
  excl0     : same, but the `excl` weight set to 0
  no_eval   : candidates restricted to TRAINING records (every evaluation record removed from the candidate set)
For each variant: the chosen previous record and p0 per evaluation record, which records would get a temperature/EC anchor
(previous = labelled training record, p0 >= .8, |dT| <= 1.5; TAG2 not re-evaluated) and which would enter the prediction chain
(previous = unlabelled same-farm earlier record, p0 >= .5, |dT| <= 1.5; CHT not re-evaluated).
Run:  python check_linker_counterfactual.py <folder with the ORIGINAL CSVs> <new work folder>
"""
import os, sys, shutil, subprocess
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PIPE = os.path.join(os.path.dirname(HERE), "pipeline_35e")
PUB, WORK = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
assert not os.path.exists(WORK); os.makedirs(WORK)
T = os.path.join(WORK, "온도"); shutil.copytree(os.path.join(PIPE, "온도", "linker"), os.path.join(T, "linker")); shutil.copytree(os.path.join(PIPE, "온도", "linker_v7"), os.path.join(T, "linker_v7"))
lw, lwm, fl, mpub = (os.path.join(WORK, n) for n in ("lw", "lwm", "fl", "maskpub"))
os.makedirs(mpub)
for f in ("train_X.csv", "train_y.csv", "sample_submission.csv"): shutil.copy(os.path.join(PUB, f), os.path.join(mpub, f))
X = pd.read_csv(os.path.join(PUB, "test_X.csv")); c = [k for k in X.columns if k != "row_id"]; X[c] = np.nan; X.to_csv(os.path.join(mpub, "test_X.csv"), index=False)
def run(script, args, env):
    e = dict(os.environ); e.update(env); e["LINKER5_OUT"] = os.path.join(WORK, "l5out")
    r = subprocess.run([sys.executable, "-B", script] + args, cwd=os.path.dirname(script), env=e, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode: raise RuntimeError(r.stderr[-2000:])
    return r.stdout
for s in ("base.py", "t1.py", "t3.py"):
    run(os.path.join(T, "linker", s), [], {"LINKER_DATA": PUB, "LINKER_WORK": lw}); run(os.path.join(T, "linker", s), [], {"LINKER_DATA": mpub, "LINKER_WORK": lwm})
run(os.path.join(T, "linker_v7", "ap_fl.py"), ["train", fl], {"LINKER_WORK": lwm})
print(run(os.path.join(T, "linker_v7", "ap_fl.py"), ["eval", fl, os.path.join(fl, "w_fl.npy")], {"LINKER_WORK": lw}))
# variants in-process on the real-data linker work folder
os.environ["LINKER_WORK"] = lw; os.environ["LINKER5_OUT"] = os.path.join(WORK, "l5out"); sys.path.insert(0, os.path.join(T, "linker_v7"))
from feat7 import *  # noqa
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]
w = np.load(os.path.join(fl, "w_fl.npy")); w0 = w.copy(); w0[FL.index("excl")] = 0.0
EV = set(np.where(~TRAIN)[0].tolist())
def pick(b, wv, removed):
    df, _ = features7(b, 0, TRAIN.copy(), removed=removed)
    if len(df) == 0: return None, 0.0
    s = df[FL].values.astype(float) @ wv; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p)); return int(df.a.values[k]), float(p[k])
rows = []
for f in ["F13", "F47"]:
    for r_ in sorted(np.where((FARM == f) & ~TRAIN)[0], key=lambda i: NUM[i]):
        for name, wv, rem in (("base", w, set()), ("excl0", w0, set()), ("no_eval", w, EV - {r_})):
            a0, p0 = pick(r_, wv, rem)
            if a0 is None: rows.append(dict(rec=rid(r_), variant=name)); continue
            dT = IN_[r_, 0, 0] - IN_[a0, 23, 0]; lab = bool(TRAIN[a0])
            rows.append(dict(rec=rid(r_), variant=name, prev=rid(a0), p0=round(p0, 4), prev_labelled=lab, xf=FARM[a0] != f, dT=round(float(dT), 2),
                             anchor=bool(lab and p0 >= .8 and np.isfinite(dT) and abs(dT) <= 1.5),
                             chain=bool((not lab) and FARM[a0] == f and NUM[a0] < NUM[r_] and p0 >= .5 and np.isfinite(dT) and abs(dT) <= 1.5)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(WORK, "linker_counterfactual_picks.csv"), index=False, encoding="utf-8-sig")
B = R[R.variant == "base"].set_index("rec")
sub = pd.read_csv(os.path.join(fl, "fl_eval_links.csv")).set_index("rec")
same = (B.prev.reindex(sub.index) == sub.prev0).all() and np.allclose(B.p0.reindex(sub.index), sub.p0, atol=1e-4)
lines = ["base == submitted fl_eval_links.csv (prev, p0): %s" % same]
for v in ("excl0", "no_eval"):
    V = R[R.variant == v].set_index("rec").reindex(B.index)
    ch = (V.prev != B.prev); lines.append("%s: previous record changed %d / %d | anchors base %d -> %d (changed %d) | chain base %d -> %d (changed %d)" % (
        v, int(ch.sum()), len(B), int(B.anchor.sum()), int(V.anchor.sum()), int((V.anchor != B.anchor).sum()), int(B.chain.sum()), int(V.chain.sum()), int((V.chain != B.chain).sum())))
    if v == "no_eval":
        aa = B[B.anchor].index; lines.append("  no_eval: anchor records whose previous record stays the same: %d / %d" % (int((V.prev.reindex(aa) == B.prev.reindex(aa)).sum()), len(aa)))
open(os.path.join(WORK, "linker_counterfactual_summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
