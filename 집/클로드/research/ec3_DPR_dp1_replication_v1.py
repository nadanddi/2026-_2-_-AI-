# -*- coding: utf-8 -*-
"""EC stage-3 DPR: replication of DP1 with NEW seeds and a NEW fold layout (fixed
before running; 2026-10-04 집 클로드).
DP1 (C6.256) improved DIAG10 -1.3~-1.7 % (P .05-.11), mostly in pass 2 (-4.3~-5.3 %,
P .04-.07), not in pass 1 / A / B.  Same seeds and folds would just repeat it, so:
  seeds 11 / 202 / 3030 (new) for BOTH the baseline R3S and DP1 (both refitted here);
  DIAG10s = DIAG10 construction with 5-record chunks shifted by 2 (chunk = (day+2)//5,
  fold = chunk % 10 per farm), +-1 purge, lock excluded; plus A, B and EL1 as before.
Features: DP1's 9 operation-sequence features (same code).
Judge: every new seed x DIAG10s / A / B: DP1 better than refitted R3S, and DIAG10s
P(worse) < .025 for every seed.  Guard: EL1 or DIAG10s pass-2 rows worse by >= 2 %.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_DPR_dp1_replication_v1.py
"""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp1)
dc5, dc4, p3, core = dp1.dc5, dp1.dc4, dp1.p3, dp1.core
SEEDS = (11, 202, 3030)
CK = os.path.join(env.LOCAL, "dpr_ckpt")


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    FS0 = [c for c in core.FULL if c != "day"] + ["season"]; BS0 = [c for c in core.BASE if c != "day"] + ["season"]
    FS1, BS1 = FS0 + dp1.NEW, BS0 + dp1.NEW
    days = lab[["farm", "day"]].drop_duplicates()
    folds = []
    for k in range(10):
        folds.append(("DIAG10s", k, {(f, int(d)) for f, d in zip(days.farm, days.day) if ((d + 2) // 5) % 10 == k and (f, int(d)) not in lock}))
    folds += [x for x in fds if x[0] in ("A", "B")]
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    for name, i, vd in folds:
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        if not va_m.any():
            continue
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
    O.to_csv(os.path.join(env.LOCAL, "ec3_DPR_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261004)
    judge = True
    print("\npooled RMSE refitted R3S -> DP1 (new seeds; judge DIAG10s/A/B)")
    for v in ("DIAG10s", "A", "B", "EL1"):
        G = O[O.validator == v]; cells = []
        for s in SEEDS:
            a, b = r(G["base_%d" % s] - G.sub_ec), r(G["dp_%d" % s] - G.sub_ec)
            if v != "EL1":
                judge &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-7s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10s"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps, ps2 = [], []
    for s in SEEDS:
        for S, store in ((D, ps), (D[D.day >= 179], ps2)):
            dd = (S["dp_%d" % s] - S.sub_ec) ** 2 - (S["base_%d" % s] - S.sub_ec) ** 2
            cl = dd.groupby(S.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
            idx = rng.integers(0, len(sm), (20000, len(sm)))
            store.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    L = D.day >= 179; P1 = D.day < 179; E = O[O.validator == "EL1"]; guard = []
    for s in SEEDS:
        guard.append(100 * (r(E["dp_%d" % s] - E.sub_ec) / r(E["base_%d" % s] - E.sub_ec) - 1))
        guard.append(100 * (r((D["dp_%d" % s] - D.sub_ec)[L]) / r((D["base_%d" % s] - D.sub_ec)[L]) - 1))
    for nm, m in (("pass-1", P1), ("pass-2", L)):
        print("  DIAG10s %s: %s" % (nm, " ".join("s%d %+.1f%%" % (s, 100 * (r((D["dp_%d" % s] - D.sub_ec)[m]) / r((D["base_%d" % s] - D.sub_ec)[m]) - 1)) for s in SEEDS)))
    print("  DIAG10s P(worse) all:", [round(p, 4) for p in ps], " pass-2 only (descriptive):", [round(p, 4) for p in ps2])
    print("  guard changes (EL1, DIAG10s pass-2 per seed, %%): %s" % [round(x, 2) for x in guard])
    ok = judge and all(p < 0.025 for p in ps); hold = any(x >= 2.0 for x in guard)
    print("\nDPR decision:", ("HOLD (guard)" if hold else "PASS") if ok else "FAIL", "(judge %s, guard trip %s)" % (ok, hold))


if __name__ == "__main__":
    main()
