# -*- coding: utf-8 -*-
"""submission_24 EC (2026-10-14 집 클로드, LEADERBOARD HYPOTHESIS TEST; catalog 6.887-6.892): round-14 EC (output_23/ec_b5.csv = round-12 EC
+ B5) + HR5 'day-level residual corrector' (TabPFN v2):
  For anchor hours h' in {0,3,6,9,12,15,18,21} one TabPFN v2 regressor (CPU, n_estimators 4, random_state 0) is fitted on the TRAINING records
  (hr5_train_base_oof.csv: 360 labelled F13/F47 records, lock days excluded; base = stored DIAG10 out-of-fold EC prediction of the round-9/12
  pipeline) with
     features = for each of [14 inputs, base prediction]: mean over hours 0..h', mean over hours 0..min(h',7) (only when h' > 7), mean over
                     hours 9..h' (only when h' >= 9) -- blocks identical by definition are dropped (bit-reproducible TabPFN; research HR5 kept them)
     target        = day-level residual d = mean(actual EC) - mean(base) of the record.
  Evaluation row (record r, hour h): h' = largest anchor hour <= h; features from r's own inputs at hours 0..h' and r's own round-14 EC
  prediction at hours 0..h' (missing feature -> training column mean); pred = round14_h + 0.5 * d_hat.   (0.5 chosen on DIAG10 before judging.)
Uses only r's own inputs/predictions up to hour h, training inputs and labels -> causal, no other evaluation record, nothing fitted on
evaluation data.  Local evidence (P2LOO pass-2 -9.1%, EL1 -4.1%, DIAG10 all rows -1.0%; critic: gain concentrated in high-EC-adjacent
records, base for validation was pre-W2/B5 - HYPOTHESIS TEST).
Run:  python hr5_post.py      (reads output_23/ec_b5.csv, data/, hr5_train_base_oof.csv; writes output_24/ec_hr5.csv)
"""
import os, sys, subprocess
if os.environ.get("PYTHONHASHSEED") != "0":   # fixed hash order -> bit-reproducible TabPFN preprocessing; re-launch once
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=dict(os.environ, PYTHONHASHSEED="0")))
import random
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[_k] = "1"   # bit-reproducible CPU inference
for _k in ("HF_HUB_OFFLINE", "TABPFN_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY"): os.environ[_k] = "1"
import env  # noqa: F401
import env_extra  # noqa: F401
import numpy as np, pandas as pd
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(0)
random.seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
USE = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
       "act_heating", "act_circfan", "act_co2", "act_fog"]
HS = (0, 3, 6, 9, 12, 15, 18, 21); LAM = 0.5


def feat(inp, base, h):
    Z = np.concatenate([inp, base[:, None]], 1)
    with np.errstate(all="ignore"):
        cm = np.nanmean(Z[:h + 1], 0); nt = np.nanmean(Z[:min(h, 7) + 1], 0); dy = np.nanmean(Z[9:h + 1], 0) if h >= 9 else cm
    return np.concatenate([cm, nt, dy])


def keep(h):
    """drop columns that are identical BY DEFINITION (night mean = cumulative mean for h <= 7, day mean = cumulative mean for h < 9);
    TabPFN preprocessing is not bit-reproducible with exactly duplicated columns."""
    return np.r_[np.arange(0, 15), np.arange(15, 30) if h > 7 else np.arange(0), np.arange(30, 45) if h >= 9 else np.arange(0)].astype(int)


def cube(X):
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].assign(rec=lambda d: d.row_id.str[:7], h=lambda d: d.row_id.str[8:10].astype(int))
    return {r: g.sort_values("h")[USE].values.astype(float) for r, g in X.groupby("rec")}


def main():
    out_dir = os.path.join(HERE, "output_24"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ec_hr5.csv"); assert not os.path.exists(out), "refusing to overwrite " + out
    P = pd.read_csv(os.path.join(HERE, "output_23", "ec_b5.csv"))
    TI = cube(pd.read_csv(os.path.join(DATA, "train_X.csv"))); EI = cube(pd.read_csv(os.path.join(DATA, "test_X.csv")))
    ty = pd.read_csv(os.path.join(DATA, "train_y.csv")).set_index("row_id").sub_ec
    B = pd.read_csv(os.path.join(HERE, "hr5_train_base_oof.csv")); B["rec"] = B.row_id.str[:7]; B["y"] = ty.reindex(B.row_id).values
    recs = sorted(B.rec.unique()); BB = {r: g.sort_values("row_id") for r, g in B.groupby("rec")}
    TD = np.array([float(BB[r].y.mean() - BB[r].base.mean()) for r in recs])
    TF = {}
    for h in HS:
        F = np.stack([feat(TI[r], BB[r].base.values, h) for r in recs])[:, keep(h)]; TF[h] = np.where(np.isfinite(F), F, np.nanmean(F, 0))
    ck = os.path.join(HERE, "tabpfn-v2-regressor.ckpt"); models = {}
    for h in HS:
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", model_path=ck, n_estimators=4, random_state=0,
                                                       ignore_pretraining_limits=True, inference_precision=torch.float32, n_preprocessing_jobs=1)
        m.fit(TF[h], TD); models[h] = m
    P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:10].astype(int); new = P.sub_ec.values.copy()
    G = {r: g.sort_values("h") for r, g in P[P.rec.str[:3].isin(["F13", "F47"])].groupby("rec")}; er = sorted(G)
    DH = {}
    for h in HS:   # one batched call per anchor hour (TabPFN test rows do not attend to each other)
        Q = np.stack([feat(EI[r], G[r].sub_ec.values, h) for r in er])[:, keep(h)]; Q = np.where(np.isfinite(Q), Q, TF[h].mean(0)[None, :])
        DH[h] = dict(zip(er, models[h].predict(Q)))
    for r in er:
        g = G[r]; new[g.index.values] = g.sub_ec.values + LAM * np.array([DH[max(x for x in HS if x <= hh)][r] for hh in g.h.values])
    P["sub_ec"] = new; P[["row_id", "sub_ec"]].to_csv(out, index=False, float_format="%.6f")
    d = new - pd.read_csv(os.path.join(HERE, "output_23", "ec_b5.csv")).sub_ec.values
    print("wrote %s | change vs round 14: RMS %.4f, mean %+.4f, min %+.3f, max %+.3f" % (out, np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max()))


if __name__ == "__main__":
    main()
