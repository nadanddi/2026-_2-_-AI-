# -*- coding: utf-8 -*-
"""EC stage-2 DC5: the full R3 member set with the DC4 season index instead of
`day`, no TabPFN (fixed before running; 2026-10-02 집 클로드).  Claude's own
check while Codex does the official integration (request 2026-10-02).

Why without TabPFN: on DIAG10 late days (test period) TabPFN alone .449 vs R3
.375 and v2 .383 > R3 (C6.156).
Candidate R3S_s = final(0.6 ET(FS) + 0.3 LGB-tweedie(BS) + 0.1 MLP(BS)), where
FS = core.FULL - day + season, BS = core.BASE - day + season; season = DC4
(exact-twin anchors, train-only, validation days interpolated by record day).
Recipe otherwise identical to Codex phase-3 R3 (seeds 7/101/2024, same folds,
lock purge, shrink + clip).  Reproduction check: plain R3 on DIAG10 fold 0 vs
stored r3_s.
Decision baseline = v2_s (current EC candidate).  Rule: all 3 seeds x 5
validators better than v2_s and DIAG10 P(worse) < .025 per seed.  Also
reported vs stored r3_s, and late / early segments.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_DC5_r3_season_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc4", os.path.join(HERE, "ec2_DC4_exact_twin_anchor_v1.py"))
dc4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc4)
p3, core = dc4.p3, dc4.core
SEEDS = (7, 101, 2024)


def r3(tr, va, s, full_cols, base_cols):
    e = core.predict_model(core.et(s), tr, va, full_cols)
    l = core.predict_model(core.lg(s, "tweedie"), tr, va, base_cols)
    mlp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        core.MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3, max_iter=800,
                                          early_stopping=True, n_iter_no_change=25, validation_fraction=.12, random_state=s))
    ml = core.predict_model(mlp, tr, va, base_cols)
    return p3.final(.6 * e + .3 * l + .1 * ml, tr, va)


def main():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    BS = [c for c in core.BASE if c != "day"] + ["season"]
    oof = pd.read_csv(p3.OUT / "oof_predictions.csv", encoding="utf-8-sig")
    rec = []
    for name, i, vd in fds:
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
        frame = va[["row_id"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["r3s_%d" % s] = r3(tr, va, s, FS, BS)
            if name == "DIAG10" and i == 0:
                b = r3(tr, va, s, list(core.FULL), list(core.BASE))
                st = oof[(oof.validator == name) & (oof.validation_fold == i)].set_index("row_id")["r3_%d" % s]
                print("  reproduction seed %d: max |R3 - stored r3| %.2e" % (s, np.max(np.abs(b - st.reindex(va.row_id).values))), flush=True)
        rec.append(frame)
        print("%s/%d done" % (name, i), flush=True)
    N = pd.concat(rec, ignore_index=True)
    N.to_csv(os.path.join(env.LOCAL, "ec2_DC5_r3s_only.csv"), index=False)
    o = oof.merge(N, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(o) == len(oof), (len(o), len(oof))
    tx = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "act_circfan", "act_vent"])
    o = o.merge(tx, on="row_id", how="left")
    d = o.groupby(["farm", "day"])
    o["sealed"] = (d.act_circfan.transform("mean") < 10) & (d.act_vent.transform(lambda x: (x == 0).mean()) > 0.85)
    o.to_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"), index=False)
    from ec2_common import judge, rmse
    ok = judge(o, lambda s: "r3s_%d" % s, 1, "DC5 R3 + season (no TabPFN) vs v2")
    print("\nalso vs stored R3 (r3_s), DIAG10 / late:")
    D = o[o.validator == "DIAG10"]
    for s in SEEDS:
        print("  seed %d: DIAG10 r3 %.4f -> r3s %.4f | late r3 %.4f -> r3s %.4f" % (
            s, rmse(D["r3_%d" % s] - D.sub_ec), rmse(D["r3s_%d" % s] - D.sub_ec),
            rmse((D["r3_%d" % s] - D.sub_ec)[D.day >= 179]), rmse((D["r3s_%d" % s] - D.sub_ec)[D.day >= 179])))
    print("\nDC5 decision:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
