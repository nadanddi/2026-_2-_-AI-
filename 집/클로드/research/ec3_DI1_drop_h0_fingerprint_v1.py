# -*- coding: utf-8 -*-
"""EC stage-3 DI1: drop the day-constant hour-0 fingerprint features from the ET
member of R3S (fixed before running; 2026-10-03 집 클로드).
Basis: statistics / competition surveys (lit_survey_2026-10-03 C, D) both rank a
'day-identifier feature audit' first: with ~400 effective days, day-constant
features let trees memorise day levels.  DI0: the 10 `*_h0` columns of the ET
input (core.FULL) are constant within every day (round-3 'source fingerprint');
LGB/MLP (core.BASE) do not use them.
Candidate per seed: R3S' = R3S + 0.6 * (ET_noh0 - ET_S), where ET_S uses FS =
FULL - day + season (as DC5) and ET_noh0 uses FS minus the 10 *_h0 columns; both
ETs refitted here with the same folds / seeds / post-processing (shrink is linear).
Baseline R3S = stored DC5 / EL1 OOF.  Rule (user's): all 3 seeds x 5 validators
better than R3S and DIAG10 P(worse) < .025; also EL1 better for all seeds.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_DI1_drop_h0_fingerprint_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc5)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "di1_ckpt")


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    FN = [c for c in FS if not c.endswith("_h0")]
    print("dropped:", [c for c in FS if c.endswith("_h0")], flush=True)
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
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["etS_%d" % s] = p3.final(core.predict_model(core.et(s), tr, va, FS), tr, va)
            frame["etN_%d" % s] = p3.final(core.predict_model(core.et(s), tr, va, FN), tr, va)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    for s in SEEDS:
        O["di_%d" % s] = O["r3s_%d" % s] + 0.6 * (O["etN_%d" % s] - O["etS_%d" % s])
    O.to_csv(os.path.join(env.LOCAL, "ec3_DI1_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261003)
    allb = True
    print("\npooled RMSE R3S -> R3S without *_h0 in ET")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        g = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(g["r3s_%d" % s] - g.sub_ec), r(g["di_%d" % s] - g.sub_ec)
            if v != "EL1":
                allb &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["di_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    E = O[O.validator == "EL1"]
    elok = all(r(E["di_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    L = D.day >= 179
    print("  DIAG10 late R3S %.4f -> DI %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["di_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nDI1 decision:", "PASS" if allb and all(p < 0.025 for p in ps) and elok else "FAIL",
          "(rule %s, EL1 %s)" % (allb and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
