# -*- coding: utf-8 -*-
"""Inspector 2 (audit 4): END-TO-END causality checks on PREDICTIONS, not only
features.  Read-only with respect to existing files; writes only
research/local/audit4_2_e2e.json.

Parts
  [C] Codex resid_reset model (Ridge on PHYSICS_COLUMNS + LGBM residual,
      round-5 training weights), trained on all F13/F47 labels, predicting
      the 1,440 test rows.  For cut points inside the test blocks:
        C1  perturb ONLY later TEST inputs of that greenhouse (x10 + noise +
            33% blanking), rebuild features + weights, retrain -> earlier test
            predictions must be bit-identical (no transductive path).
        C2  perturb ALL later inputs (train + test) -> earlier test FEATURES
            bit-identical (predictions may change: the model is retrained on
            changed later training rows, which is legitimate).
        C3  other greenhouse's test inputs perturbed -> this greenhouse's
            predictions identical.
        C4  labels of the other 49 greenhouses shuffled -> predictions
            identical; F13/F47 labels shuffled -> predictions change (sanity).
        C5  every test input perturbed -> training weights identical.
  [R] Round-5 temperature features (make_submission_v6 columns): perturb ONLY
      later TEST inputs -> count TRAINING rows whose model features change
      (transductive dependence through cross-day ewm / rolling features).
  [S] Strain EWM (ec_denoise_v1: halflife 1.5 h over train+test per
      greenhouse): perturb all test inputs -> how many rough TRAINING rows
      change and by how much.
  [K] causal_shrink: perturb predictions of later hours -> earlier outputs
      identical; row-order independent.

Run:  cd research && PYTHONPATH="" <python> -u audit4_2_e2e.py
"""
import json
import os
import sys

import env  # noqa: F401
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from common import USABLE, TARGET_FARMS
import train_flags_v6 as TF
from make_submission_v3 import causal_shrink

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis", "codex_independent", "2차"))
sys.dont_write_bytecode = True
from resid_reset_features import build_features, FEATURE_COLUMNS, PHYSICS_COLUMNS  # noqa: E402

ORIG = common.load_raw
RAW = ORIG()
ID = ["row_id", "farm", "day", "hour", "t"]


def corrupt(df, mask, seed):
    q = df.copy(deep=True)
    rng = np.random.default_rng(seed)
    cols = [c for c in q.columns if c not in ID and pd.api.types.is_numeric_dtype(q[c])]
    n = int(mask.sum())
    if n:
        q.loc[mask, cols] = q.loc[mask, cols] * 10 + rng.normal(0, 5, (n, len(cols)))
        pos = np.flatnonzero(mask)
        blank = pos[rng.random(n) < .33]
        q.loc[q.index[blank], cols] = np.nan
    return q


def bits(a):
    return np.ascontiguousarray(np.asarray(a, np.float64)).view(np.uint64)


def weights_for(lab, raw):
    common.load_raw = lambda: tuple(x.copy() for x in raw)
    try:
        return TF.row_weights(lab, 0.2, w_noisy=0.2)
    finally:
        common.load_raw = ORIG


def codex_run(tX, ty, sX):
    z = build_features(tX, sX)
    test_ids = set(sX.row_id)
    lab = z[~z.row_id.isin(test_ids)].merge(ty[["row_id", "sub_temp"]], on="row_id")
    lab = lab[lab.sub_temp.notna()].reset_index(drop=True)
    te = z.set_index("row_id").loc[sX.row_id].reset_index()
    w = weights_for(lab, (tX, ty, sX))
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.))
    lin.fit(lab[PHYSICS_COLUMNS], lab.sub_temp.values, ridge__sample_weight=w)
    m = LGBMRegressor(n_estimators=220, learning_rate=.035, num_leaves=12, max_depth=-1, min_child_samples=100,
                      reg_lambda=15, verbosity=-1, n_jobs=4, random_state=726, deterministic=True, force_row_wise=True)
    m.fit(lab[FEATURE_COLUMNS], lab.sub_temp.values - lin.predict(lab[PHYSICS_COLUMNS]), sample_weight=w)
    p = lin.predict(te[PHYSICS_COLUMNS]) + m.predict(te[FEATURE_COLUMNS])
    return p, te, w, lab


def cut_points(sX):
    cuts = []
    for f in TARGET_FARMS:
        ts = np.sort(sX.loc[sX.farm == f, "t"].values)
        days = np.sort(sX.loc[sX.farm == f, "day"].unique())
        cuts += [(f, int(ts[0])), (f, int(ts[5])), (f, int(days[4] * 24 + 23)),   # end of block 1
                 (f, int(days[12] * 24 + 13)), (f, int(ts[-13]))]
    return cuts


def main():
    tX, ty, sX = RAW
    res = {"C": [], "R": [], "S": {}, "K": {}}
    p0, te0, w0, lab0 = codex_run(tX, ty, sX)
    print("codex base: %d train rows, %d test preds, weights<1: %d" % (len(lab0), len(p0), int((w0 < 1).sum())))
    ok = True

    for f, cut in cut_points(sX):
        # C1: later TEST inputs only
        m = ((sX.farm == f) & (sX.t > cut)).to_numpy()
        p1, te1, w1, _ = codex_run(tX, ty, corrupt(sX, m, cut))
        keep = ((te0.farm == f) & (te0.t <= cut)).to_numpy() | (te0.farm != f).to_numpy()
        same = bool((bits(p0[keep]) == bits(p1[keep])).all())
        wsame = bool((bits(w0) == bits(w1)).all())
        # C2: ALL later inputs (train + test) -> earlier test features identical
        mt = ((tX.farm == f) & (tX.t > cut)).to_numpy()
        z2 = build_features(corrupt(tX, mt, cut + 7), corrupt(sX, m, cut + 8)).set_index("row_id")
        z0 = build_features(tX, sX).set_index("row_id")
        ids = sX.row_id[((sX.farm == f) & (sX.t <= cut)).to_numpy()]
        fsame = bool((bits(z0.loc[ids, FEATURE_COLUMNS]) == bits(z2.loc[ids, FEATURE_COLUMNS])).all()) if len(ids) else True
        r = dict(farm=f, cut=cut, earlier_test_rows=int(len(ids)), C1_pred_identical=same, C1_weights_identical=wsame,
                 C2_features_identical=fsame, C1_changed_later_rows=int((bits(p0[~keep]) != bits(p1[~keep])).sum()))
        print(r, flush=True)
        res["C"].append(r)
        ok &= same and wsame and fsame

    # C3 other greenhouse test inputs
    for f in TARGET_FARMS:
        m = (sX.farm != f).to_numpy()
        p3, _, _, _ = codex_run(tX, ty, corrupt(sX, m, 31))
        k = (te0.farm == f).to_numpy()
        s3 = bool((bits(p0[k]) == bits(p3[k])).all())
        res["C"].append(dict(test="C3_other_farm_test", farm=f, identical=s3))
        print("C3", f, s3)
        ok &= s3
    # C4 labels
    rng = np.random.default_rng(5)
    ty2 = ty.copy()
    other = (~ty2.farm.isin(TARGET_FARMS)).to_numpy()
    for c in ("sub_temp", "sub_ec"):
        ty2.loc[other, c] = rng.permutation(ty2.loc[other, c].values)
    p4, _, _, _ = codex_run(tX, ty2, sX)
    s4 = bool((bits(p0) == bits(p4)).all())
    ty3 = ty.copy()
    tg = ~other
    ty3.loc[tg, "sub_temp"] = rng.permutation(ty3.loc[tg, "sub_temp"].values)
    p5, _, _, _ = codex_run(tX, ty3, sX)
    res["C"].append(dict(test="C4_other_labels_shuffled", identical=s4,
                         target_labels_shuffled_rms_change=float(np.sqrt(np.mean((p5 - p0) ** 2)))))
    print("C4", s4, float(np.sqrt(np.mean((p5 - p0) ** 2))))
    ok &= s4
    # C5 weights vs all test inputs
    w5 = weights_for(lab0, (tX, ty2, corrupt(sX, np.ones(len(sX), bool), 99)))
    s5 = bool((bits(w0) == bits(w5)).all())
    res["C"].append(dict(test="C5_weights_vs_all_test_inputs_and_other_labels", identical=s5))
    print("C5", s5)
    ok &= s5

    # [R] round-5 temperature features: training rows depending on test inputs
    import features_v2 as F2
    import feat_new
    import features_v4 as F4
    man = json.load(open(os.path.join(HERE, "submissions", "manifest_06.json"), encoding="utf-8"))
    cols = sorted(set(man["sub_temp"]["features"]) | set(man["sub_temp"]["phys_features"]))

    def build_all():
        a, b, c = common.load_raw()
        p = F2.build(a, c)
        p["midnight"] = (p.hour == 0).astype(float)
        p = p.set_index("row_id")
        ex, _ = feat_new.build_extra()
        parts = [p, ex.set_index("row_id"), F4.seg_features().set_index("row_id"),
                 F4.phys_features().set_index("row_id"), F4.fp_features().set_index("row_id")]
        out = pd.concat([x.loc[:, ~x.columns.isin(["farm", "t"])] for x in parts], axis=1)
        return out.loc[:, ~out.columns.duplicated()]

    def with_raw(raw):
        common.load_raw = lambda: tuple(x.copy() for x in raw)
        try:
            return build_all()
        finally:
            common.load_raw = ORIG

    B0 = with_raw((tX, ty, sX))
    trn = ty[ty.farm.isin(TARGET_FARMS) & ty.sub_temp.notna()].row_id
    trn = trn[trn.isin(B0.index)]
    for f in TARGET_FARMS:
        days = np.sort(sX.loc[sX.farm == f, "day"].unique())
        cut = int(days[4] * 24 + 23)          # after test block 1 -> later test blocks perturbed
        m = ((sX.farm == f) & (sX.t > cut)).to_numpy()
        B1 = with_raw((tX, ty, corrupt(sX, m, 3)))
        a = B0.loc[trn, [c for c in cols if c in B0.columns]].astype(float)
        b = B1.loc[trn, a.columns].astype(float)
        diff = ~((a.values == b.values) | (np.isnan(a.values) & np.isnan(b.values)))
        rows = diff.any(1)
        ci = [c for j, c in enumerate(a.columns) if diff[:, j].any()]
        rtest = sX.row_id[((sX.farm == f) & (sX.t <= cut)).to_numpy()]
        dt = ~((B0.loc[rtest, a.columns].values == B1.loc[rtest, a.columns].values)
               | (B0.loc[rtest, a.columns].isna().values & B1.loc[rtest, a.columns].isna().values))
        r = dict(farm=f, cut=cut, train_rows_changed=int(rows.sum()), train_rows=int(len(trn)),
                 cols_changed=len(ci), example_cols=ci[:12], earlier_test_feature_cells_changed=int(dt.sum()))
        print("R", r, flush=True)
        res["R"].append(r)

    # [S] Strain EWM
    NOISY = ["in_temp", "in_hum", "in_co2"]

    def ewm_frame(a, c):
        p = pd.concat([a, c], ignore_index=True)
        p = p[p.farm.isin(TARGET_FARMS)].sort_values(["farm", "t"])
        out = {x: p.groupby("farm")[x].transform(lambda s: s.ewm(halflife=1.5, ignore_na=True).mean()).values
               for x in NOISY}
        return pd.DataFrame(out, index=p.row_id.values)
    E0 = ewm_frame(tX, sX)
    E1 = ewm_frame(tX, corrupt(sX, np.ones(len(sX), bool), 4))
    nd = TF.noisy_days()
    rough = set(map(tuple, nd[nd.noise_score >= nd.noise_score.quantile(.75)][["farm", "day"]].values))
    tr = ty[ty.farm.isin(TARGET_FARMS) & ty.sub_ec.notna()]
    tr = tr[[(f, d) in rough for f, d in zip(tr.farm, tr.day)]].row_id
    d = (E0.loc[tr] - E1.loc[tr]).abs()
    ch = (d > 0).any(axis=1)
    test_days = set(zip(sX.farm, sX.day))
    rough_after_test = sorted({(f, dd) for f, dd in rough if (f, dd - 1) in test_days})
    res["S"] = {"rough_train_rows": int(len(tr)), "rows_changed_any": int(ch.sum()),
                "rows_changed_gt_1e-6": int((d > 1e-6).any(axis=1).sum()),
                "rows_changed_gt_0p01": int((d > 0.01).any(axis=1).sum()),
                "max_abs_change": d.max().to_dict(),
                "rough_days_right_after_test": [[x[0], int(x[1])] for x in rough_after_test]}
    print("S", res["S"], flush=True)

    # [K] causal_shrink
    te = sX[["farm", "day", "hour"]].reset_index(drop=True)
    rng = np.random.default_rng(11)
    p = rng.normal(1, .3, len(te))
    base = causal_shrink(p, te, 0.5)
    kok = True
    for f in TARGET_FARMS:
        for cut in cut_points(sX):
            if cut[0] != f:
                continue
            q = p.copy()
            later = ((sX.farm == f) & (sX.t > cut[1])).to_numpy()
            q[later] = q[later] * 10 + 3
            o = causal_shrink(q, te, 0.5)
            kok &= bool((bits(o[~later]) == bits(base[~later])).all())
    perm = rng.permutation(len(te))
    o2 = causal_shrink(p[perm], te.iloc[perm].reset_index(drop=True), 0.5)
    back = np.empty(len(te)); back[perm] = o2
    korder = bool(np.allclose(back, base, atol=1e-12, rtol=0))
    res["K"] = dict(future_invariant=kok, order_independent=korder)
    print("K", res["K"])
    ok &= kok and korder

    res["codex_all_pass"] = bool(ok)
    os.makedirs(env.LOCAL, exist_ok=True)
    with open(os.path.join(env.LOCAL, "audit4_2_e2e.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
