# -*- coding: utf-8 -*-
"""EC stage-3 LR1: log-ratio target for the ET member of R3S (fixed before running;
2026-10-03 집 클로드).
Basis: surveys C/D (multiplicative targets) + 6.177: the high-EC state is ranked
almost perfectly (AUC .987) but its size is under-predicted (1.65 vs 1.26).  If
the state acts as a multiplier on a seasonal level, a target z = log(EC / b(season))
lets the trees share one 'x k' across seasons instead of averaging raw levels.
b(season): per farm, from the fold's TRAINING days only: day-mean EC sorted by
season index, centred rolling median over 21 days (min 5), np.interp at any season.
ET_LR: core.et(s) on FS (FULL - day + season) fitted to z; prediction
b(season_va) * exp(z_hat) (no smearing), then p3.final as every member.
Candidate per seed: R3S' = R3S + 0.6 * (ET_LR - ET_S), ET_S = DI1 checkpoint
(identical recipe/seed/folds).  Baseline R3S = stored DC5 / EL1 OOF.
Rule (user's): all 3 seeds x 5 validators better and DIAG10 P(worse) < .025;
EL1 better for all seeds also required.  High-EC days (label day mean >= 1)
DIAG10 bias reported.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_LR1_log_ratio_target_v1.py
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
CK = os.path.join(env.LOCAL, "lr1_ckpt")
DI1 = os.path.join(env.LOCAL, "di1_ckpt")


def base_fn(tr):
    out = {}
    for f, G in tr.groupby("farm"):
        d = G.groupby("day").agg(s=("season", "first"), y=("sub_ec", "mean")).sort_values("s")
        b = d.y.rolling(21, center=True, min_periods=5).median().bfill().ffill()
        out[f] = (d.s.values, b.values)
    return lambda farm, s: np.array([np.interp(x, *out[f]) for f, x in zip(farm, s)])


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"]
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
        b = base_fn(tr)
        btr, bva = b(tr.farm, tr.season), b(va.farm, va.season)
        trz = tr.copy(); trz["sub_ec"] = np.log(tr.sub_ec.values / btr)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        frame["b"] = bva
        for s in SEEDS:
            z = core.predict_model(core.et(s), trz, va, FS)
            frame["etL_%d" % s] = p3.final(bva * np.exp(z), tr, va)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    S = pd.concat([pd.read_csv(os.path.join(DI1, f)) for f in sorted(os.listdir(DI1))], ignore_index=True)
    P = P.merge(S[["row_id", "validator", "validation_fold"] + ["etS_%d" % s for s in SEEDS]],
                on=["row_id", "validator", "validation_fold"], how="inner")
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P) == len(S), (len(O), len(P), len(S))
    for s in SEEDS:
        O["lr_%d" % s] = O["r3s_%d" % s] + 0.6 * (O["etL_%d" % s] - O["etS_%d" % s])
    O.to_csv(os.path.join(env.LOCAL, "ec3_LR1_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261003)
    allb = True
    print("\npooled RMSE R3S -> R3S with log-ratio ET")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, bb = r(G["r3s_%d" % s] - G.sub_ec), r(G["lr_%d" % s] - G.sub_ec)
            if v != "EL1":
                allb &= bb < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, bb, 100 * (bb / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["lr_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    E = O[O.validator == "EL1"]
    elok = all(r(E["lr_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    L = D.day >= 179
    hi = D.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1
    print("  DIAG10 late R3S %.4f -> LR %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["lr_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 high-EC days (n rows %d): true %.3f  R3S %.3f  LR %.3f  ET_S %.3f  ET_LR %.3f" % (
        hi.sum(), D.sub_ec[hi].mean(), D.r3s_7[hi].mean(), D.lr_7[hi].mean(), D.etS_7[hi].mean(), D.etL_7[hi].mean()))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nLR1 decision:", "PASS" if allb and all(p < 0.025 for p in ps) and elok else "FAIL",
          "(rule %s, EL1 %s)" % (allb and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
