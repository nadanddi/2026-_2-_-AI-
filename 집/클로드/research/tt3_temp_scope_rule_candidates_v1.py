# -*- coding: utf-8 -*-
"""TT3/TT4/TT5: temperature candidates judged with the NEW scope-based temperature rule (user decision 2026-10-10;
memory skill-not-luck.md).  (집 클로드, user: "1,2 가자").  Fixed before running.  k = 4, alpha = .025 / 4 = .00625.
Reference W40G-S:  .4 BASE + (.6 - .4 g) CODEX + .4 g PFN,  g = clip((in_temp - 8)/2, 0, 1).
Candidates:
  TT3  [scope WARM]  .4 BASE + (.6 - w g) CODEX + w g PFN with w chosen ONCE from {.4, .5, .6} by the lowest mean
       RMSE on DIAG10 PASS-1 rows of the STORED TT1 members (seeds 7/101, PFN families A/B); cold rows (g = 0)
       are identical to W40G-S by construction.  If w = .4 is chosen, TT3 = reference (reported as 'no change').
  TT4  [scope COLD]  support gate: g' = g * s, s = clip((q99 - d) / (q99 - q90), 0, 1), d = mean Euclidean distance
       to the 10 nearest TRAINING rows of the fold (other days) in z-scored (in_temp, out_temp, in_hum, out_rad,
       hr_sin, hr_cos); q90/q99 = quantiles of d over training rows (their 10-NN among training rows of OTHER days).
       Blend .4 BASE + (.6 - .4 g') CODEX + .4 g' PFN.  (TabPFN share shrinks where the input was not seen.)
  TT5a / TT5b [scope ALL]  CO2-rough training days (6.437 definition, 연구실/클로드/results/rw1_rough_days_v1.csv,
       30 days) get training weight min(w, .2) / 0 for BASE and CODEX (PFN unchanged); blend as reference.
Confirmation members (never used): BASE seeds 4141 / 5151; PFN TabPFN v2 season contexts 25-28 (family C) and
29-32 (family D), 2000 weighted rows, n_estimators 4, float32, GPU; CODEX deterministic (seed 726).
World/folds/weights/season = Codex TK2 world.py.  Validators DIAG10, EXT10, EXT12, EL1.
RULE (per candidate):  primary = DIAG10 (WARM) / EXT10 (COLD) / DIAG10 and EXT10 (ALL);
  PASS iff every BASE seed x PFN family better on each primary validator AND primary farm x 5-day block bootstrap
  (20,000) P(worse) < .00625 for every seed x family;  and no other validator significantly worse (seed-x-family-mean
  worse AND bootstrap share(candidate better) < .00625).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt3_temp_scope_rule_candidates_v1.py [sum]
"""
import os, sys, importlib.util, gc
HERE = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
TKDIR = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1")
sys.path.insert(0, TKDIR)
import world as W  # noqa: E402
import common  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
CK = os.path.join(env.LOCAL, "tt3_ckpt"); TT1 = os.path.join(env.LOCAL, "tt1_ckpt")
SEEDS = (4141, 5151); FAMS = {"C": range(25, 29), "D": range(29, 33)}
ALPHA = .025 / 4
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
SUPV = ["in_temp", "out_temp", "in_hum", "out_rad"]


def choose_w():
    G = pd.concat([pd.read_csv(os.path.join(TT1, f)) for f in sorted(os.listdir(TT1))], ignore_index=True)
    X = G[(G.validator == "DIAG10") & (G.day < 179)]
    g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1))
    sc = {w: np.mean([r(.4 * X["base_REF_%d" % s] + (.6 - w * g) * X.codex_REF + w * g * X["pfn_%s" % f] - X.sub_temp)
                      for s in (7, 101) for f in ("A", "B")]) for w in (.4, .5, .6)}
    w = min(sc, key=sc.get)
    print("TT3 selection on stored DIAG10 pass-1:", {k: round(v, 5) for k, v in sc.items()}, "-> w =", w, flush=True)
    return w


def support(trX, vaX, trday, vaday):
    mu, sd = np.nanmean(trX, 0), np.nanstd(trX, 0); sd[sd == 0] = 1
    A = np.nan_to_num((trX - mu) / sd); B = np.nan_to_num((vaX - mu) / sd)
    nn = NearestNeighbors(n_neighbors=40).fit(A)
    dv, _ = nn.kneighbors(B, n_neighbors=10)
    dt, it = nn.kneighbors(A, n_neighbors=40)
    other = trday[it] != trday[:, None]
    dtr = np.array([row[m][:10].mean() if m.sum() >= 10 else row[m].mean() for row, m in zip(dt, other)])
    q90, q99 = np.quantile(dtr, .90), np.quantile(dtr, .99)
    return np.clip((q99 - dv.mean(1)) / (q99 - q90), 0, 1)


def run():
    import env_extra_gpu  # noqa: F401  (registers GPU torch DLLs; research/env_extra_gpu.py)
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    os.makedirs(CK, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    tX, ty, sX = W.TM.ORIG()
    full = tX[tX.farm.isin(common.TARGET_FARMS)].set_index("row_id")
    for c in SUPV:
        lab["_s_" + c] = full[c].reindex(lab.row_id).to_numpy()
    lab["_s_hs"] = np.sin(2 * np.pi * lab.hour / 24); lab["_s_hc"] = np.cos(2 * np.pi * lab.hour / 24)
    SC = ["_s_" + c for c in SUPV] + ["_s_hs", "_s_hc"]
    R = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))
    rough = set(zip(R.farm, R.day.astype(int))); assert len(rough) == 30
    isr = np.array([(f, int(d)) in rough for f, d in zip(lab.farm, lab.day)])
    wbs = {"REF": wb, "R02": np.where(isr, np.minimum(wb, .2), wb), "R00": np.where(isr, 0.0, wb)}
    wps = {"REF": wp, "R02": np.where(isr, np.minimum(wp, .2), wp), "R00": np.where(isr, 0.0, wp)}
    FC = list(W.FEATURE_COLUMNS); X = pfn[FC].to_numpy(np.float32); y = pfn.sub_temp.to_numpy(); ixday = FC.index("day")
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt3_" + name, k)
            out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True)
            out["validator"], out["fold"] = name, k
            old = pd.read_csv(os.path.join(TT1, "%s_%d.csv" % (name, k)))
            assert (old.row_id.values == out.row_id.values).all()
            out["codex_REF"] = old.codex_REF.values
            saveFC = W.TM.FEATURE_COLUMNS
            for tag in ("REF", "R02", "R00"):
                for s in SEEDS:
                    out["base_%s_%d" % (tag, s)] = W.base_predict(lab[tm], lab[vm], wbs[tag][tm], ct, phc, s)
                if tag != "REF":
                    W.TM.FEATURE_COLUMNS = [c if c != "day" else "season" for c in FC]
                    try:
                        out["codex_%s" % tag] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wps[tag][tm], 726)
                    finally:
                        W.TM.FEATURE_COLUMNS = saveFC
            XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season
            for fam, ctxs in FAMS.items():
                ps = []
                for seed in ctxs:
                    idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
                    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                                   ignore_pretraining_limits=True, inference_precision=torch.float32)
                    m.fit(XS[idx], y[tm][idx]); ps.append(np.asarray(m.predict(VS), float)); del m; gc.collect(); torch.cuda.empty_cache()
                out["pfn_%s" % fam] = np.mean(ps, axis=0)
            out["s_support"] = support(lab.loc[tm, SC].to_numpy(float), lab.loc[vm, SC].to_numpy(float),
                                       (lab.loc[tm, "farm"] + lab.loc[tm, "day"].astype(str)).to_numpy(), None)
            assert out.drop(columns=["in_temp"]).notna().all().all()
            out.to_csv(path, index=False)
            print("%s/%d done" % (name, k), flush=True)


def blend(X, s, fam, cand, w3):
    g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1)).astype(float)
    B, C, P = X["base_REF_%d" % s], X.codex_REF, X["pfn_%s" % fam]
    if cand == "REF":
        return .4 * B + (.6 - .4 * g) * C + .4 * g * P
    if cand == "TT3":
        return .4 * B + (.6 - w3 * g) * C + w3 * g * P
    if cand == "TT4":
        g2 = g * X.s_support.values
        return .4 * B + (.6 - .4 * g2) * C + .4 * g2 * P
    tag = {"TT5a": "R02", "TT5b": "R00"}[cand]
    return .4 * X["base_%s_%d" % (tag, s)] + (.6 - .4 * g) * X["codex_%s" % tag] + .4 * g * P


def boot(X, a, b, seed=0):
    """share of farm x 5-day block resamples where candidate (b) is NOT better than reference (a)."""
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series(((b - X.sub_temp) ** 2 - (a - X.sub_temp) ** 2).values).groupby(cl).agg(["sum", "count"])
    sm, n = dd["sum"].values, dd["count"].values
    idx = np.random.default_rng(seed).integers(0, len(sm), (20000, len(sm)))
    return float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())


def summarize(w3=None):
    w3 = w3 if w3 is not None else choose_w()
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    print("folds:", G.groupby("validator").fold.nunique().to_dict())
    scope = {"TT3": ["DIAG10"], "TT4": ["EXT10"], "TT5a": ["DIAG10", "EXT10"], "TT5b": ["DIAG10", "EXT10"]}
    for cand in ("TT3", "TT4", "TT5a", "TT5b"):
        print("\n==== %s (primary %s)" % (cand, scope[cand])); ok = True
        for v in ("DIAG10", "EXT10", "EXT12", "EL1"):
            X = G[G.validator == v]
            cells, ps, d_all, better_share = [], [], [], []
            for s in SEEDS:
                for fam in FAMS:
                    a, b = blend(X, s, fam, "REF", w3), blend(X, s, fam, cand, w3)
                    cells.append(r(b - X.sub_temp) < r(a - X.sub_temp)); d_all.append(r(b - X.sub_temp) / r(a - X.sub_temp) - 1)
                    pw = boot(X, a, b); ps.append(pw); better_share.append(1 - pw)
            L = (X.day >= 179).values
            a2 = np.mean([blend(X, s, f, "REF", w3)[L] for s in SEEDS for f in FAMS], 0); b2 = np.mean([blend(X, s, f, cand, w3)[L] for s in SEEDS for f in FAMS], 0)
            line = "  %-6s %s  cells %d/4  change %+.2f%%..%+.2f%%  P(worse) max %.4f | pass-2 %+.2f%%" % (
                v, "PRIMARY" if v in scope[cand] else "other  ", sum(cells), 100 * min(d_all), 100 * max(d_all), max(ps),
                100 * (r(b2 - X.sub_temp[L]) / r(a2 - X.sub_temp[L]) - 1))
            if v in scope[cand]:
                ok &= all(cells) and max(ps) < ALPHA
            else:
                sig_worse = np.mean(d_all) > 0 and min(better_share) < ALPHA
                ok &= not sig_worse
                line += "  significantly worse: %s" % sig_worse
            print(line)
        X = G[G.validator == "DIAG10"]; cold = (X.in_temp <= 8).values
        print("  cold rows (<=8C, DIAG10, mean of seeds/families): REF %.4f  %s %.4f" % (
            r(np.mean([blend(X, s, f, "REF", w3)[cold] for s in SEEDS for f in FAMS], 0) - X.sub_temp[cold]), cand,
            r(np.mean([blend(X, s, f, cand, w3)[cold] for s in SEEDS for f in FAMS], 0) - X.sub_temp[cold])))
        print("  VERDICT %s: %s%s" % (cand, "PASS" if ok else "FAIL", " (w = .4 -> identical to reference)" if cand == "TT3" and w3 == .4 else ""))


if __name__ == "__main__":
    w = choose_w()
    if MODE != "sum":
        run()
    summarize(w)
