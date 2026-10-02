# -*- coding: utf-8 -*-
"""EC stage-2 CW1: re-test catalog 6.85 under the USER's rule (fixed before
running; 2026-10-02 집 클로드, user: "6.85 단독 시험 진행해").
6.85 (Codex, 09-29): ExtraTrees sample weight 2 on training days whose hour-0
in_temp <= 10 C (same-day hour-0 input, causal), everything else unchanged.
It was rejected on Codex's stricter screen (12/12 fold cells; one cell +0.56%)
and never judged on the full validator set.  Test-input share of such days:
27/60.
Implementation on Codex phase-3 (same folds, lock purge, core.et seeds 7/101/
2024, core.FULL with `day` as in v2): retrain only the ET member with weights;
post-processing is linear, so  v2' = v2_s + 0.48 * (ET'_s - full_et_s)
(ET share in v2 = 0.8 * 0.6).  A reproduction check compares an unweighted ET
with the stored full_et_s on DIAG10 fold 0.
Rule: all 3 seeds x 5 validators better than v2_s and DIAG10 P(worse) < .025
per seed (farm x 5-day blocks, 20,000).  Late / early / sealed reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_CW1_cold_day_weight_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

P3 = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "ec_restart_phase3_20261001_v1")
sys.path.insert(0, P3)
spec = importlib.util.spec_from_file_location("p3_readonly", os.path.join(P3, "run_benchmark.py"))
p3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p3)
core = p3.core
SEEDS = (7, 101, 2024)


def main():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    h0 = lab[lab.hour == 0].set_index(["farm", "day"]).in_temp
    lab["cold"] = [bool(h0.get((f, d), np.nan) <= 10) for f, d in zip(lab.farm, lab.day)]
    print("cold training-candidate days: %d of %d" % (lab[lab.cold][["farm", "day"]].drop_duplicates().shape[0],
                                                     lab[["farm", "day"]].drop_duplicates().shape[0]))
    oof = pd.read_csv(p3.OUT / "oof_predictions.csv", encoding="utf-8-sig")
    rec = []
    for name, i, vd in fds:
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m], lab[va_m]
        w = np.where(tr.cold, 2.0, 1.0)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["etw_%d" % s] = p3.final(core.predict_model(core.et(s), tr, va, core.FULL, weight=w), tr, va)
            if name == "DIAG10" and i == 0:
                base = p3.final(core.predict_model(core.et(s), tr, va, core.FULL), tr, va)
                st = oof[(oof.validator == name) & (oof.validation_fold == i)].set_index("row_id")["full_et_%d" % s]
                print("  reproduction seed %d: max |ET - stored full_et| %.2e" % (s, np.max(np.abs(base - st.reindex(va.row_id).values))))
        rec.append(frame)
        print("%s/%d done" % (name, i), flush=True)
    N = pd.concat(rec, ignore_index=True)
    N.to_csv(os.path.join(env.LOCAL, "ec2_CW1_et_only.csv"), index=False)
    # execution fix (no metric seen): merge on ids only; float labels differ after the CSV round trip
    o = oof.merge(N[["row_id", "validator", "validation_fold"] + ["etw_%d" % s for s in SEEDS]],
                  on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(o) == len(oof), (len(o), len(oof))
    for s in SEEDS:
        o["cw_%d" % s] = o["v2_%d" % s] + 0.48 * (o["etw_%d" % s] - o["full_et_%d" % s])
    o.to_csv(os.path.join(env.LOCAL, "ec2_CW1_oof.csv"), index=False)
    from ec2_common import judge
    tx = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "act_circfan", "act_vent"])
    o = o.merge(tx, on="row_id", how="left")
    d = o.groupby(["farm", "day"])
    o["sealed"] = (d.act_circfan.transform("mean") < 10) & (d.act_vent.transform(lambda x: (x == 0).mean()) > 0.85)
    ok = judge(o, lambda s: "cw_%d" % s, 1, "CW1 cold-day ET weight x2 (6.85)")
    print("\nCW1 decision:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
