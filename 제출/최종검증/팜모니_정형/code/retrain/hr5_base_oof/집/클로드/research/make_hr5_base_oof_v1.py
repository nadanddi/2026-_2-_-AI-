# -*- coding: utf-8 -*-
"""Rebuild EC/hr5_train_base_nosg_oof.csv from the original competition data (final-verification package, 2026-10-17 집 클로드).

What the stored file is: for the 360 labelled F13/F47 training records (lock days excluded), the DIAG10 out-of-fold EC prediction of
the round-9 EC member blend BEFORE the SG2 step, averaged over three seeds:
    nosg_s = clip(0.8 * (0.6 ET_s + 0.3 LGB_s + 0.1 MLP_s) + 0.2 * PFN, lo, hi),   base = mean_s nosg_s,   s in (47, 1414, 6464)
It is the training-side input of the HR5 day-level corrector (EC/f5e_post.py stage_hr5).  It uses train_X / train_y only.

Original chain (research code, unchanged here):
  ec3_WT0_r3_member_weights_v1.py  ET / LGB-tweedie / MLP per DIAG10 fold and seed (season DC4 + DP1 features), each shrunk (core.shrink)
  ec3_WT1_dp1_removal_pfn_share_v1.py  PFN = shrink(mean of TabPFN v2 over contexts 5..8, 2000 random training rows each)
  연구실/클로드/code/ec_rv1_wt1_on_sg2_v1.py  nosg = clip(0.8 r3 + 0.2 pfn, lo, hi), tag CUR, validator DIAG10
This script repeats exactly those steps for the 10 DIAG10 folds only.  One difference: WT1 ran TabPFN on a GPU (device="cuda",
speed only); by default this script runs it on the CPU with the shipped checkpoint (same weights, SHA256 2ab5a07d...0736).
2026-10-17 CPU run: ET / LGB / MLP members identical to the stored research run (max diff 4.4e-16); the TabPFN part differs
(max 0.048, weight 0.2 -> stored base max diff 0.0096, RMS 0.0007).  HR5_PFN_DEVICE=cuda uses the GPU as in the original run.
The comparison against the stored file is printed at the end.

Run:  set AGRI_DATA=<folder with the ORIGINAL train_X.csv, train_y.csv, test_X.csv>
      python make_hr5_base_oof_v1.py <tabpfn-v2-regressor.ckpt> <out.csv> [<stored hr5_train_base_nosg_oof.csv to compare>]
"""
import env  # noqa: F401  (must be first)
import importlib.util, os, sys, time
import numpy as np, pandas as pd
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT, OUT = sys.argv[1], sys.argv[2]
REF = sys.argv[3] if len(sys.argv) > 3 else None
assert not os.path.exists(OUT), "refusing to overwrite " + OUT
_argv = sys.argv; sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
sys.argv = _argv
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
SEEDS = wt0.SEEDS                                    # (47, 1414, 6464)
CK = os.path.join(env.LOCAL, "hr5_base_ckpt")


DEVICE = os.environ.get("HR5_PFN_DEVICE", "cpu")    # original WT1 run: "cuda" (set HR5_PFN_DEVICE=cuda on a CUDA machine to match it)


def pfn(Xc, yc, Xq, seed):                           # = WT1 pfn(), explicit checkpoint
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEVICE, model_path=CKPT, n_estimators=4, random_state=seed,
                                                   ignore_pretraining_limits=True, inference_precision=torch.float32)
    m.fit(Xc, yc)
    return np.asarray(m.predict(Xq), float)


def main():
    os.makedirs(CK, exist_ok=True); t0 = time.time()
    raw, full, lab0, lock, signatures, fds = p3.prepare()
    lab = lab0.join(pd.concat([dp1.day_feats(g) for _, g in lab0.groupby(["farm", "day"])]))    # WT0: DP1 features joined
    wv = dc4.weather_vectors(full)
    FSm = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW                           # WT0 member features
    BSm = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    FSp = [c for c in core.FULL if c != "day"] + ["season"]                                     # WT1 PFN features
    folds = [x for x in fds if x[0] == "DIAG10"]
    for name, i, vd in folds:
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["lo"], frame["hi"] = tr.sub_ec.min(), tr.sub_ec.max()
        for s in SEEDS:
            e, l, m = wt0.members(tr, va, s, FSm, BSm)
            for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                frame["%s_%d" % (nm, s)] = core.shrink(v, va)
        Xtr, ytr, Xva = tr[FSp].to_numpy(np.float32), tr.sub_ec.to_numpy(float), va[FSp].to_numpy(np.float32)
        ps = []
        for c in (5, 6, 7, 8):
            ix = np.random.default_rng(c).choice(len(tr), size=min(2000, len(tr)), replace=False)
            ps.append(pfn(Xtr[ix], ytr[ix], Xva, c))
        frame["pfn"] = core.shrink(np.mean(ps, axis=0), va)
        frame.to_csv(path, index=False)
        print("%s/%d done (%.0fs)" % (name, i, time.time() - t0), flush=True)
    G = pd.concat([pd.read_csv(os.path.join(CK, "DIAG10_%d.csv" % i)) for i in range(10)], ignore_index=True)
    nos = []
    for s in SEEDS:                                                                              # RV1 build(), w = .2 (CUR)
        r3 = .6 * G["et_%d" % s] + .3 * G["lgb_%d" % s] + .1 * G["mlp_%d" % s]
        nos.append(np.clip(.8 * r3.values + .2 * G.pfn.values, G.lo, G.hi))
    B = pd.DataFrame({"row_id": G.row_id, "base": np.mean(nos, axis=0)}).sort_values("row_id").reset_index(drop=True)
    assert len(B) == 8640 and B.row_id.is_unique
    B.to_csv(OUT, index=False, float_format="%.10f")
    print("wrote %s | rows %d, records %d | %.0fs" % (OUT, len(B), B.row_id.str[:7].nunique(), time.time() - t0))
    if REF:
        R = pd.read_csv(REF).set_index("row_id").base.reindex(B.row_id).values; d = B.base.values - R
        print("vs stored %s: max |diff| %.3e, RMS diff %.3e, mean diff %+.3e" % (os.path.basename(REF), np.abs(d).max(), np.sqrt((d ** 2).mean()), d.mean()))


if __name__ == "__main__":
    main()
