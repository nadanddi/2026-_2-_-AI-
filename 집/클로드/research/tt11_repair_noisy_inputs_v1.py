# -*- coding: utf-8 -*-
"""TT11: repair the injected-noise TRAINING inputs instead of distrusting their labels (2026-10-10 집 클로드, user: "운영진이
더미 데이터를 잘 분간하는 걸 목표로 했다면?" -> TD17: on the 30 strict CO2-rough training days (6.437, rw1_rough_days_v1)
in_co2 / in_hum / in_temp are corrupted (diff autocorr .52/.49/.59 -> -.23/.15/.29) but sub_temp is as smooth as on normal
days (.89 vs .90) -> labels look genuine, noise is input-only.  User ruling 10-10 (memory training-cleaning-noncausal-ok):
cleaning TRAINING rows may use past+future values; evaluation-row features stay causal; held-out rows keep original
inputs.  6.21 (causal 1 h EWM repair) hurt - likely lag; this uses CENTRED smoothing (no lag).  Fixed before running.
Repair: per farm, t-ordered train_X; centred rolling mean (window w, min_periods 1) of in_co2, in_hum, in_temp; values
  replaced ONLY on rows of the 30 strict rough days.  All features rebuilt in the MASK world from the repaired train_X.
  Validation rows always use features from the ORIGINAL inputs.  Fold sets, season mapping, TabPFN member (stored,
  joint, unrepaired - partial) unchanged.
Weights: W02 = existing (restored rows 0.2, quartile-noisy days 0.2);  W1 = same but the 30 strict days back to 1
  (labels trusted once inputs are repaired); restored-row flags keep 0.2.
Candidates (k = 3 -> alpha .025/3 = .00833): S3W1 (w=3), S5W1 (w=5), S5W02 (w=5, weights unchanged = repair only).
RULE (user temperature rule, scope ALL): every BASE seed {7,101} x PFN family {A,B} better on DIAG10 AND EXT10, farm x
  5-day block bootstrap P(worse) < .00833 on both; EXT12/EL1 fail iff seed-mean worse and share(better) < .00833.
  CODEX deterministic (seed 726 only).  Reported: pass-2 rows, rows of the 30 rough days, in_temp < 6 rows.
Sanity asserts: repaired world differs from original only on rough-day rows of in_co2/in_hum/in_temp inputs; row order
  identical; test inputs untouched.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt11_repair_noisy_inputs_v1.py
"""
import os, sys, pathlib
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt11_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt11_season"
ROUGH = os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv")
COLS = ["in_co2", "in_hum", "in_temp"]
VALS = ("DIAG10", "EXT10", "EXT12", "EL1"); ALPHA = .025 / 3
CANDS = {"S3W1": (3, "W1"), "S5W1": (5, "W1"), "S5W02": (5, "W02")}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
RD = set(map(tuple, pd.read_csv(ROUGH)[["farm", "day"]].values))


def repaired_loader(orig, w):
    def ld():
        tX, ty, sX = orig()
        tX = tX.copy()
        is_rd = np.array([(f, d) in RD for f, d in zip(tX.farm, tX.day)])
        for f in ("F13", "F47"):
            m = (tX.farm == f).values
            idx = tX.index[m][np.argsort(tX.t.values[m])]
            for c in COLS:
                sm = tX.loc[idx, c].rolling(w, center=True, min_periods=1).mean()
                rep = idx[is_rd[tX.index.get_indexer(idx)]]
                tX.loc[rep, c] = sm.loc[rep]
        return tX, ty, sX
    return ld


def build_world_with(loader):
    TM = W.TM; save_o, save_c = TM.ORIG, common.load_raw
    TM.ORIG = loader; W.harness._CACHE.clear()
    try:
        out = W.worlds()
    finally:
        TM.ORIG = save_o; common.load_raw = save_c; W.harness._CACHE.clear()
    return out


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab0, pfn0, ct, phc, wb0, wp0, wv0, sets, z = W.worlds()
    wb0, wp0 = np.asarray(wb0, float), np.asarray(wp0, float)
    TF = W.TF
    flag_only = np.asarray(TF.row_weights(lab0, .2, w_noisy=None), float)
    nd = TF.noisy_days(); qn = set(map(tuple, nd[nd.noisy][["farm", "day"]].values))
    key = list(zip(lab0.farm, lab0.day))
    w1 = flag_only * np.array([.2 if (k in qn and k not in RD) else 1.0 for k in key])
    print("rough strict days %d, rows %d; W02 down-weighted %d rows, W1 %d rows" % (len(RD), sum(k in RD for k in key), int((wb0 < 1).sum()), int((w1 < 1).sum())), flush=True)
    worlds = {}
    for w in (3, 5):
        labS, pfnS, ctS, phcS, *_ = build_world_with(repaired_loader(W.TM.ORIG, w))
        assert (labS.row_id.values == lab0.row_id.values).all() and ctS == ct and phcS == phc
        rd = np.array([k in RD for k in key])
        d = np.abs(labS.in_temp.values - lab0.in_temp.values); d = np.where(np.isnan(d), 0, d)
        assert d[~rd].max() == 0 and d[rd].max() > 0, "repair leaked outside rough days"
        print("w=%d: rough-day rows in_temp mean |change| %.3f, CODEX feature max diff on non-rough rows %.3g" % (
            w, d[rd].mean(), np.nanmax(np.abs(pfnS.loc[~rd, list(W.FEATURE_COLUMNS)].to_numpy(float) - pfn0.loc[~rd, list(W.FEATURE_COLUMNS)].to_numpy(float)))), flush=True)
        worlds[w] = (labS, pfnS)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab0, fd)
            if not vm.sum():
                continue
            out = pd.DataFrame({"validator": name, "row_id": lab0.row_id[vm].values})
            for cn, (w, wt) in CANDS.items():
                labS, pfnS = worlds[w]
                wbx = w1 if wt == "W1" else wb0
                wpx = w1 if wt == "W1" else wp0
                tr, va = W.season_fold(pfnS[tm], pfn0[vm], wv0, "tt11_" + name, k)
                for s in (7, 101):
                    out["base_%s_%d" % (cn, s)] = W.base_predict(labS[tm], lab0[vm], wbx[tm], ct, phc, s)
                save = W.TM.FEATURE_COLUMNS
                W.TM.FEATURE_COLUMNS = FC
                try:
                    out["codex_%s" % cn] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wpx[tm], 726)
                finally:
                    W.TM.FEATURE_COLUMNS = save
            out.to_csv(path, index=False); print("%s/%d done" % (name, k), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261013)
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))])
    G = G.merge(C, on=["validator", "row_id"], how="left"); assert G.codex_S5W1.notna().all()
    G["g"] = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    G["rd"] = [(f, d) in RD for f, d in zip(G.farm, G.day)]
    bl = lambda X, b, c, f: .4 * X[b] + (.2 + .4 * (1 - X.g)) * X[c] + .4 * X.g * X["pfn_%s" % f]
    for cn in CANDS:
        print("\n==== %s" % cn); ok = True; rel = {v: [] for v in VALS}; pw = {v: [] for v in VALS}
        for s in (7, 101):
            for f in "AB":
                line = "seed %3d PFN %s |" % (s, f)
                for v in VALS:
                    X = G[G.validator == v]; y = X.sub_temp.values
                    a = bl(X, "base_REF_%d" % s, "codex_REF", f).values; b = bl(X, "base_%s_%d" % (cn, s), "codex_%s" % cn, f).values
                    p = boot(X, a, b, rng); rel[v].append(r(b - y) / r(a - y) - 1); pw[v].append(p)
                    L = X.day.values >= 179; R = X.rd.values
                    line += " %s %+.2f%% P %.3f (2차 %+.2f%%, 거친날 %s) |" % (v, 100 * rel[v][-1], p, 100 * (r((b - y)[L]) / r((a - y)[L]) - 1),
                                                                         "%.3f->%.3f" % (r((a - y)[R]), r((b - y)[R])) if R.any() else "-")
                print(line)
        for v in VALS:
            m = np.mean(rel[v])
            if v in ("DIAG10", "EXT10"):
                ok &= all(x < 0 for x in rel[v]) and max(pw[v]) < ALPHA
            else:
                ok &= not (m > 0 and (1 - min(pw[v])) < ALPHA)
            print("  %-6s seed-mean %+.2f%%  P(worse) max %.4f" % (v, 100 * m, max(pw[v])))
        print("VERDICT %s: %s" % (cn, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT11_SUM") != "1":
        run()
    summarize()
