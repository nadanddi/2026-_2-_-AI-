# -*- coding: utf-8 -*-
"""EC stage-3 HM1: high-EC magnitude specialist blended only on rows the model flags
as high (fixed before running; 2026-10-04 집 클로드).
Basis: HC2 (6.241) - on high-EC days half of the day-level error is a common
under-shoot (1.55 true vs 1.22 predicted) and half is day-to-day size; the general
model is trained on 91 % normal days.  Not tried before: a model trained ONLY on
high / near-high days, used only where R3S already flags high (HG1 used labels of
earlier days and is not applicable in pass 2; SB1 keyed on pair role).
Specialist: ExtraTrees (core.et(seed), FS = FULL - day + season, DC4 season of the
fold) fitted on the fold's training rows whose day label mean >= 0.8; p3.final with
those rows as reference.  Inputs only (no label history) -> evaluation-legal.
Flag (causal): cm_h = expanding mean of R3S (seed s) over hours 0..h >= 0.9.
Candidate: flagged rows 0.5 * R3S + 0.5 * specialist; other rows R3S.  Fixed now.
Judge (EC rule decided 2026-10-04, 6.247): every seed x DIAG10/A/B better and DIAG10
P(worse) < .025.  Guard: if EL1 or DIAG10 pass-2 rows (day >= 179) worsen by >= 2 %
for any seed -> HOLD.  EXT10/EXT12 reported only.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_HM1_high_magnitude_specialist_v1.py
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
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "hm1_ckpt")
TRAIN_THR, FLAG_THR, W = 0.8, 0.9, 0.5


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
        hs = tr[tr.groupby(["farm", "day"]).sub_ec.transform("mean") >= TRAIN_THR]
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        frame["n_spec_days"] = hs[["farm", "day"]].drop_duplicates().shape[0]
        for s in SEEDS:
            frame["sp_%d" % s] = p3.final(core.predict_model(core.et(s), hs, va, FS), hs, va)
        frame.to_csv(path, index=False)
        print("%s/%d done (specialist days %d)" % (name, i, frame.n_spec_days.iloc[0]), flush=True)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    O = O.sort_values(["validator", "validation_fold", "farm", "day", "hour"]).reset_index(drop=True)
    g = O.groupby(["validator", "validation_fold", "farm", "day"])
    for s in SEEDS:
        cm = g["r3s_%d" % s].transform(lambda x: x.expanding().mean())
        O["flag_%d" % s] = cm >= FLAG_THR
        O["hm_%d" % s] = np.where(O["flag_%d" % s], (1 - W) * O["r3s_%d" % s] + W * O["sp_%d" % s], O["r3s_%d" % s])
    O.to_csv(os.path.join(env.LOCAL, "ec3_HM1_all.csv"), index=False)
    print("flagged share (seed 7):", O.groupby("validator").flag_7.mean().round(3).to_dict())
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261004)
    judge = True
    print("\npooled RMSE R3S -> HM1 (judge: DIAG10/A/B; others reported)")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["hm_%d" % s] - G.sub_ec)
            if v in ("DIAG10", "A", "B"):
                judge &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["hm_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    L = D.day >= 179
    E = O[O.validator == "EL1"]
    guard = []
    for s in SEEDS:
        guard.append(100 * (r(E["hm_%d" % s] - E.sub_ec) / r(E["r3s_%d" % s] - E.sub_ec) - 1))
        guard.append(100 * (r((D["hm_%d" % s] - D.sub_ec)[L]) / r((D["r3s_%d" % s] - D.sub_ec)[L]) - 1))
    hi = D.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1
    print("  DIAG10 late R3S %.4f -> HM1 %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["hm_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 high-EC days (rows %d): label %.3f  R3S %.3f  HM1 %.3f  specialist %.3f" % (
        hi.sum(), D.sub_ec[hi].mean(), D.r3s_7[hi].mean(), D.hm_7[hi].mean(), D.sp_7[hi].mean()))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("  guard changes (EL1, DIAG10-late per seed, %%): %s" % [round(x, 2) for x in guard])
    ok = judge and all(p < 0.025 for p in ps)
    hold = any(x >= 2.0 for x in guard)
    print("\nHM1 decision:", ("HOLD (guard)" if hold else "PASS") if ok else "FAIL", "(judge %s, guard trip %s)" % (ok, hold))


if __name__ == "__main__":
    main()
