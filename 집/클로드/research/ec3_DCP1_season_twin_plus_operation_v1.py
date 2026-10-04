# -*- coding: utf-8 -*-
"""EC stage-3 DCP1: DP1 operation-sequence features + DC7 own-weather twin season for
query rows, applied to pass-2 rows only; judged by the pass-2-only protocol (catalog
6.279), fixed before running; 2026-10-04 집 클로드.
Components were seen before (DC7 6.253: pass-2 gains; DP1 C6.256/6.278/6.280: pass-2
gains 9/9 cells but P .144), so this is ONE run with new seeds + a fresh layout.
Candidate: R3 with FS/BS + DP1's 9 features; validation-row season = DC7 rule (hours
0..h exact twin with >= 6 matched hours, else DC4 interpolation).  Baseline: R3S
(FS/BS, DC4 season) refitted with the same new seeds 61 / 707 / 8080.
Sets (pass-2 rows only): DIAG10 original layout, DIAG10u fresh layout (chunk =
(day+1)//5, fold = chunk % 10, +-1 purge, lock excluded; never used before), EL1.
PASS iff every seed x all three sets improve AND seed-mean block bootstrap P(worse)
on DIAG10u pass-2 rows < .025.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_DCP1_season_twin_plus_operation_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
def load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
dp1 = load("dp1", "ec3_DP1_daily_operation_pattern_v1.py")
dc7 = load("dc7", "ec3_DC7_query_twin_season_v1.py")
dc5, dc4, p3, core = dp1.dc5, dp1.dc4, dp1.p3, dp1.core
SEEDS = (61, 707, 8080)
CK = os.path.join(env.LOCAL, "dcp1_ckpt")


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    FS0 = [c for c in core.FULL if c != "day"] + ["season"]; BS0 = [c for c in core.BASE if c != "day"] + ["season"]
    FS1, BS1 = FS0 + dp1.NEW, BS0 + dp1.NEW
    days = lab[["farm", "day"]].drop_duplicates()
    folds = [x for x in fds if x[0] == "DIAG10"]
    for k in range(10):
        folds.append(("DIAG10u", k, {(f, int(d)) for f, d in zip(days.farm, days.day) if ((d + 1) // 5) % 10 == k and (f, int(d)) not in lock}))
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
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
        vq_day = dict(zip(zip(vdays.farm, vdays.day), vq))
        va_base = va.copy(); va_base["season"] = [vq_day[(f, d)] for f, d in zip(va.farm, va.day)]
        va_cand = va.copy(); va_cand["season"] = dc7.row_season(va, vq_day, tdays, wv)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["base_%d" % s] = dc5.r3(tr, va_base, s, FS0, BS0)
            frame["cand_%d" % s] = dc5.r3(tr, va_cand, s, FS1, BS1)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O = O[O.day >= 179].copy()
    O.to_csv(os.path.join(env.LOCAL, "ec3_DCP1_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    ok = True
    print("\nPASS-2 rows: refitted R3S -> DCP1 (new seeds)")
    for v in ("DIAG10", "DIAG10u", "EL1"):
        G = O[O.validator == v]; cells = []
        for s in SEEDS:
            a, b = r(G["base_%d" % s] - G.sub_ec), r(G["cand_%d" % s] - G.sub_ec)
            ok &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-8s days %2d  %s" % (v, G.groupby(["farm", "day"]).ngroups, "  ".join(cells)))
    T = O[O.validator == "DIAG10u"].copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
    bm, cm = T[["base_%d" % s for s in SEEDS]].mean(axis=1), T[["cand_%d" % s for s in SEEDS]].mean(axis=1)
    dd = (cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2
    cl = dd.groupby(T.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
    idx = np.random.default_rng(20261004).integers(0, len(sm), (20000, len(sm)))
    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    print("  DIAG10u pass-2 seed-mean RMSE %.4f -> %.4f  P(worse) %.4f" % (r(bm - T.sub_ec), r(cm - T.sub_ec), p))
    print("\nDCP1 decision:", "PASS" if ok and p < 0.025 else "FAIL", "(all seeds x sets better %s, P %.4f)" % (ok, p))


if __name__ == "__main__":
    main()
