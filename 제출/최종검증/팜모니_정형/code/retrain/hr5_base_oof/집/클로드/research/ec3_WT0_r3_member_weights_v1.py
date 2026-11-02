# -*- coding: utf-8 -*-
"""EC stage-3 WT0: re-check the R3 member weights (0.6 ET / 0.3 LGB / 0.1 MLP, set long ago,
before the season index and DP1 features) on the CURRENT configuration (season DC4 + DP1
features in all three members, as submission_13/14).  Fixed before running; 2026-10-05 집 클로드.
Members stored per fold and seed (new seeds 47 / 1414 / 6464): ET (FULL+season+DP1), LGB-tweedie
(BASE+season+DP1), MLP (BASE+season+DP1).  final(w) = clip(shrink(w . members)) as p3.final.
SELECTION (one rule, fixed): the weight triple on the simplex grid (step .1, 66 triples) with the
lowest seed-mean RMSE on DIAG10 PASS-1 rows (day < 179; pass-1 rows are never judged below).
JUDGEMENT (pass-2-only protocol 6.279, P on the structure-matched validator P2LOO 6.328): chosen
w* vs current (.6, .3, .1) on pass-2 rows of DIAG10, P2LOO and EL1; PASS iff every seed x all three
sets improve AND seed-mean farm x 5-day block bootstrap P(worse) on P2LOO pass-2 rows < .025.
If w* == current, the result is 'no change'.  Also reported: the full grid on pass-2 (descriptive).
Run (checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_WT0_r3_member_weights_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys, itertools
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp1)
dc5, dc4, p3, core = dp1.dc5, dp1.dc4, dp1.p3, dp1.core
SEEDS = (47, 1414, 6464)
CK = os.path.join(env.LOCAL, "wt0_ckpt")
CUR = (.6, .3, .1)


def members(tr, va, s, FS, BS):
    e = core.predict_model(core.et(s), tr, va, FS)
    l = core.predict_model(core.lg(s, "tweedie"), tr, va, BS)
    mlp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        core.MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3, max_iter=800,
                                          early_stopping=True, n_iter_no_change=25, validation_fraction=.12, random_state=s))
    m = core.predict_model(mlp, tr, va, BS)
    return e, l, m


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    folds = [x for x in fds if x[0] == "DIAG10"]
    for f in ("F13", "F47"):
        for d in sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique()):
            folds.append(("P2LOO", len(folds), {(f, int(d))}))
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    for name, i, vd in folds:
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
        frame["lo"], frame["hi"] = tr.sub_ec.min(), tr.sub_ec.max()
        for s in SEEDS:
            e, l, m = members(tr, va, s, FS, BS)
            for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                frame["%s_%d" % (nm, s)] = core.shrink(v, va)   # shrink is linear: shrink(sum w x) = sum w shrink(x)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O.to_csv(os.path.join(env.LOCAL, "ec3_WT0_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))

    def pred(G, w, s):
        return np.clip(w[0] * G["et_%d" % s] + w[1] * G["lgb_%d" % s] + w[2] * G["mlp_%d" % s], G.lo, G.hi)
    grid = [(a / 10, b / 10, round(1 - a / 10 - b / 10, 1)) for a in range(11) for b in range(11 - a)]
    S1 = O[(O.validator == "DIAG10") & (O.day < 179)]
    sel = sorted(grid, key=lambda w: r(np.mean([pred(S1, w, s) for s in SEEDS], axis=0) - S1.sub_ec))
    w = sel[0]
    print("\nSELECTION on DIAG10 pass-1 rows: best %s (RMSE %.4f) | current %s (RMSE %.4f)" % (
        w, r(np.mean([pred(S1, w, s) for s in SEEDS], axis=0) - S1.sub_ec), CUR,
        r(np.mean([pred(S1, CUR, s) for s in SEEDS], axis=0) - S1.sub_ec)))
    print("top 5:", sel[:5])
    P2 = O[O.day >= 179]
    print("\nDESCRIPTIVE pass-2 seed-mean RMSE by weights (DIAG10 | P2LOO | EL1), best 8 by mean:")
    desc = []
    for g in grid:
        vals = [r(np.mean([pred(P2[P2.validator == v], g, s) for s in SEEDS], axis=0) - P2[P2.validator == v].sub_ec) for v in ("DIAG10", "P2LOO", "EL1")]
        desc.append((np.mean(vals), g, vals))
    for m_, g, vals in sorted(desc)[:8]:
        print("  %s  %.4f | %.4f | %.4f" % (g, *vals))
    cur_vals = [x for x in desc if x[1] == CUR][0]; print("  current %s  %.4f | %.4f | %.4f" % (CUR, *cur_vals[2]))
    if w == CUR:
        print("\nWT0 decision: NO CHANGE (selection keeps the current weights)"); return
    ok = True
    print("\nJUDGEMENT pass-2 rows: current %s -> %s" % (CUR, w))
    for v in ("DIAG10", "P2LOO", "EL1"):
        G = P2[P2.validator == v]; cells = []
        for s in SEEDS:
            a, b = r(pred(G, CUR, s) - G.sub_ec), r(pred(G, w, s) - G.sub_ec); ok &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-7s %s" % (v, "  ".join(cells)))
    T = P2[P2.validator == "P2LOO"].copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
    bm = np.mean([pred(T, CUR, s) for s in SEEDS], axis=0); cm = np.mean([pred(T, w, s) for s in SEEDS], axis=0)
    dd = pd.Series((cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2, index=T.index).groupby(T.cl).agg(["sum", "count"])
    sm, n = dd["sum"].values, dd["count"].values
    idx = np.random.default_rng(20261005).integers(0, len(sm), (20000, len(sm)))
    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    print("  P2LOO seed-mean %.4f -> %.4f  P(worse) %.4f" % (r(bm - T.sub_ec), r(cm - T.sub_ec), p))
    print("\nWT0 decision:", "PASS" if ok and p < .025 else "FAIL", "(all better %s, P %.4f)" % (ok, p))


if __name__ == "__main__":
    main()
