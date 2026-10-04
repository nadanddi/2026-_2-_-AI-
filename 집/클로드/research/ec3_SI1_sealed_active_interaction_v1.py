# -*- coding: utf-8 -*-
"""EC stage-3 SI1: sealed x active-operation interaction features (from SD0, catalog
6.293: under-predicted sealed days had daytime heating / circulation fan / less shade,
over-predicted ones were quietly sealed; family-wise p .12, not a clue -> one test,
user request 2026-10-04 집 클로드).  Fixed before running.
Candidate: R3 with FS/BS + 3 causal to-date interactions (hours 0..h of the record):
  si_heat  = act_vent_tdz * act_heating_tdm
  si_fan   = act_vent_tdz * act_circfan_tdm
  si_open  = act_vent_tdz * (100 - act_shade_tdm)
Baseline: R3S (FS/BS, DC4 season) refitted with the same new seeds 13 / 505 / 6060.
Judged by the pass-2-only protocol (catalog 6.279), applied to pass-2 rows only.
Sets (pass-2 rows only): DIAG10 original layout, DIAG10w fresh layout (chunk =
(day+3)//5, fold = chunk % 10, +-1 purge, lock excluded; never used before), EL1.
PASS iff every seed x all three sets improve AND seed-mean block bootstrap (farm x
5-day, 20000) P(worse) on DIAG10w pass-2 rows < .025.
Descriptive extra: sealed pass-2 days and high days RMSE.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_SI1_sealed_active_interaction_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp1)
dc5, dc4, p3, core = dp1.dc5, dp1.dc4, dp1.p3, dp1.core
SEEDS = (13, 505, 6060)
NEW = ["si_heat", "si_fan", "si_open"]
CK = os.path.join(env.LOCAL, "si1_ckpt")


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab["si_heat"] = lab.act_vent_tdz * lab.act_heating_tdm
    lab["si_fan"] = lab.act_vent_tdz * lab.act_circfan_tdm
    lab["si_open"] = lab.act_vent_tdz * (100 - lab.act_shade_tdm)
    wv = dc4.weather_vectors(full)
    FS0 = [c for c in core.FULL if c != "day"] + ["season"]; BS0 = [c for c in core.BASE if c != "day"] + ["season"]
    FS1, BS1 = FS0 + NEW, BS0 + NEW
    days = lab[["farm", "day"]].drop_duplicates()
    folds = [x for x in fds if x[0] == "DIAG10"]
    for k in range(10):
        folds.append(("DIAG10w", k, {(f, int(d)) for f, d in zip(days.farm, days.day) if ((d + 3) // 5) % 10 == k and (f, int(d)) not in lock}))
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    for name, i, vd in folds:
        if not any(d >= 179 for f, d in vd):
            continue
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
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["base_%d" % s] = dc5.r3(tr, va, s, FS0, BS0)
            frame["dp_%d" % s] = dc5.r3(tr, va, s, FS1, BS1)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O = O[O.day >= 179].copy()                     # protocol: judge pass-2 rows only
    O.to_csv(os.path.join(env.LOCAL, "ec3_SI1_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    ok = True
    print("\nPASS-2 rows: refitted R3S -> SI1 (new seeds)")
    for v in ("DIAG10", "DIAG10w", "EL1"):
        G = O[O.validator == v]; cells = []
        for s in SEEDS:
            a, b = r(G["base_%d" % s] - G.sub_ec), r(G["dp_%d" % s] - G.sub_ec)
            ok &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-8s days %2d  %s" % (v, G.groupby(["farm", "day"]).ngroups, "  ".join(cells)))
    T = O[O.validator == "DIAG10w"].copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
    bm, dm = T[["base_%d" % s for s in SEEDS]].mean(axis=1), T[["dp_%d" % s for s in SEEDS]].mean(axis=1)
    dd = (dm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2
    cl = dd.groupby(T.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
    idx = np.random.default_rng(20261004).integers(0, len(sm), (20000, len(sm)))
    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    print("  DIAG10w pass-2 seed-mean RMSE %.4f -> %.4f  P(worse) %.4f" % (r(bm - T.sub_ec), r(dm - T.sub_ec), p))
    print("\nDPC decision:", "PASS" if ok and p < 0.025 else "FAIL", "(all seeds x sets better %s, P %.4f)" % (ok, p))


if __name__ == "__main__":
    main()
