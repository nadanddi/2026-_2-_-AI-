# -*- coding: utf-8 -*-
"""EC stage-3 TF1: predicted substrate temperature as an EC feature (multi-task
transfer of the sub_temp labels; fixed before running; 2026-10-04 집 클로드).
Basis: NX0 - on sealed days the W40G-predicted substrate temperature correlates with
the R3S EC day residual (+.30); the EC model never sees sub_temp labels.
Per outer fold (DIAG10/A/B/EXT10/EXT12 + EL1):
  temperature model = ExtraTrees(300 trees, min_samples_leaf 5, seed 7) on
  FS + 4 outdoor channels, target sub_temp (train_y), fitted on the fold's TRAINING
  rows only.  Training rows get 5-fold cross-fitted predictions (5-record chunks
  round robin, +-1 purge), validation rows the full-train prediction.  No sub_temp
  label of a validation row is used anywhere.
  New EC features: tpred, tgap = tpred - in_temp; added to FS and BS of the DC5 R3
  recipe (seeds 7/101/2024).
Judge (EC rule 2026-10-04, 6.247): every seed x DIAG10/A/B better, DIAG10 P(worse)
< .025; guard: EL1 or DIAG10 pass-2 rows worse by >= 2 % -> HOLD; EXT reported.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_TF1_pred_substrate_temp_feature_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "tf1_ckpt")
OUTW = ["out_temp", "out_hum", "out_rad", "out_wspd"]


def tmodel():
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesRegressor(300, min_samples_leaf=5, max_features=1.0, n_jobs=4, random_state=7))


def temp_preds(tr, va, cols):
    days = tr[["farm", "day"]].drop_duplicates().sort_values(["farm", "day"]).reset_index(drop=True)
    days["inner"] = (days.groupby("farm").cumcount() // 5) % 5
    key = dict(zip(zip(days.farm, days.day), days.inner))
    tr_pred = np.full(len(tr), np.nan)
    trk = np.array([key[(f, d)] for f, d in zip(tr.farm, tr.day)])
    for k in range(5):
        vd = {fd for fd, i in key.items() if i == k}
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)}
        fit = np.array([(f, d) not in forb for f, d in zip(tr.farm, tr.day)])
        m = tmodel().fit(tr.loc[fit, cols], tr.loc[fit, "sub_temp"])
        tr_pred[trk == k] = m.predict(tr.loc[trk == k, cols])
    m = tmodel().fit(tr[cols], tr.sub_temp)
    return tr_pred, m.predict(va[cols])


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
    lab = lab.merge(Y, on="row_id", how="left")
    assert lab.sub_temp.notna().mean() > .99
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    BS = [c for c in core.BASE if c != "day"] + ["season"]
    TC = FS + [c for c in OUTW if c not in FS]
    el = []
    for f in ("F13", "F47"):
        days = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(days), 5):
            el.append(("EL1", len(el), {(f, int(d)) for d in days[k:k + 5]}))
    for name, i, vd in list(fds) + el:
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates()
        vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
        vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        trt = tr.dropna(subset=["sub_temp"])
        tp_tr, tp_va = temp_preds(trt, va.assign(sub_temp=np.nan), TC)
        tr["tpred"] = pd.Series(tp_tr, index=trt.index).reindex(tr.index)
        va["tpred"] = tp_va
        tr["tgap"], va["tgap"] = tr.tpred - tr.in_temp, va.tpred - va.in_temp
        frame = va[["row_id", "farm", "day", "hour", "sub_ec", "tpred"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["tf_%d" % s] = dc5.r3(tr, va, s, FS + ["tpred", "tgap"], BS + ["tpred", "tgap"])
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    O.to_csv(os.path.join(env.LOCAL, "ec3_TF1_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261004)
    judge = True
    print("\npooled RMSE R3S -> TF1 (judge: DIAG10/A/B; others reported)")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["tf_%d" % s] - G.sub_ec)
            if v in ("DIAG10", "A", "B"):
                judge &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["tf_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    L = D.day >= 179
    E = O[O.validator == "EL1"]
    guard = []
    for s in SEEDS:
        guard.append(100 * (r(E["tf_%d" % s] - E.sub_ec) / r(E["r3s_%d" % s] - E.sub_ec) - 1))
        guard.append(100 * (r((D["tf_%d" % s] - D.sub_ec)[L]) / r((D["r3s_%d" % s] - D.sub_ec)[L]) - 1))
    hi = D.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1
    print("  DIAG10 late R3S %.4f -> TF1 %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["tf_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 high-EC days (rows %d): label %.3f  R3S %.3f  TF1 %.3f" % (hi.sum(), D.sub_ec[hi].mean(), D.r3s_7[hi].mean(), D.tf_7[hi].mean()))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("  guard changes (EL1, DIAG10-late per seed, %%): %s" % [round(x, 2) for x in guard])
    ok = judge and all(p < 0.025 for p in ps)
    hold = any(x >= 2.0 for x in guard)
    print("\nTF1 decision:", ("HOLD (guard)" if hold else "PASS") if ok else "FAIL", "(judge %s, guard trip %s)" % (ok, hold))


if __name__ == "__main__":
    main()
