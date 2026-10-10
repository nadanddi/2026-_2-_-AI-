# -*- coding: utf-8 -*-
"""TT24: FINAL confirmation of the GRU member with the day gate T=10 on a never-used layout J (2026-10-10 집 클로드).
TS6e screen: blend = W40G-S + 0.1 * g * h10 * (GRU - W40G-S), h10 = clip((m_t - 10)/2, 0, 1), m_t = running min of in_temp
over hours 0..t of the record day; layouts F/G/H/I all -0.70..-0.76% (pass-2 -1.6..-2.0%), pooled P ~0, EXT10 +0.04%,
EXT12 +0.07%.  The threshold was chosen after seeing results -> this single run on layout J decides.  Fixed before running.
Layout DIAG10J (never used): per farm sorted training days, 6-DAY chunks (i // 6), fold = chunk % 10 (all five 5-day boundary
  shifts are already consumed by DIAG10/F/G/H/I); training excludes same farm +-1 and other farm +-5 days.
  New BASE seeds 1010 / 1111, CODEX 726, TabPFN season member refit on GPU with NEW context seeds 49..56 (A 49-52, B 53-56),
  GRU refit with the TS6 recipe and NEW seeds 3 / 4.
PASS iff on layout J all rows (rough days excluded): all 4 cells (BASE seed x PFN family) better and farm x 5-day block
  bootstrap P(worse) < .025 (k = 1).  Pass-2 rows, F13/F47 reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt24_gru_daygate_confirm_v1.py
"""
import os, sys, pathlib, gc
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
import importlib.util as _iu
_sp = _iu.spec_from_file_location('s6', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ts6_gru_member_v1.py')); S6 = _iu.module_from_spec(_sp); _sp.loader.exec_module(S6)
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt24_ckpt")
CK13 = os.path.join(env.LOCAL, "tt13_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt24_season"
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
COLS = ["in_co2", "in_hum", "in_temp"]; SEEDS = (1010, 1111); ALPHA = .0125
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def repaired_loader(orig, w=5):
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
    TM = W.TM; so, sc = TM.ORIG, common.load_raw
    TM.ORIG = loader; W.harness._CACHE.clear()
    try:
        return W.worlds()
    finally:
        TM.ORIG = so; common.load_raw = sc; W.harness._CACHE.clear()


def fresh_folds(lab):
    out = [{f: set() for f in ("F13", "F47")} for _ in range(10)]
    for f in ("F13", "F47"):
        for i, d in enumerate(sorted(lab[lab.farm == f].day.unique())):
            out[(i // 6) % 10][f].add(int(d))
    return out


def masks(lab, fd):
    val = np.zeros(len(lab), bool); bad = np.zeros(len(lab), bool); d = lab.day.values
    for f, ds in fd.items():
        isf = (lab.farm == f).values; ds = np.array(sorted(ds))
        val |= isf & np.isin(d, ds)
        bad |= (isf & np.isin(d, np.unique((ds[:, None] + np.arange(-1, 2)).ravel()))) | (~isf & np.isin(d, np.unique((ds[:, None] + np.arange(-5, 6)).ravel())))
    return ~bad & ~val, val


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    wb, wp = np.asarray(wb, float), np.asarray(wp, float)
    TF = W.TF
    flag_only = np.asarray(TF.row_weights(lab, .2, w_noisy=None), float)
    nd = TF.noisy_days(); qn = set(map(tuple, nd[nd.noisy][["farm", "day"]].values))
    w1 = flag_only * np.array([.2 if ((f, d) in qn and (f, d) not in RD) else 1.0 for f, d in zip(lab.farm, lab.day)])
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    FC1 = [c for c in FC if not c.endswith("_h0") and not (c == "act_heating" or c.startswith("act_heating_"))]

    def codex(tr, va, wtr, cols):
        save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = cols
        try:
            return W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wtr, 726)
        finally:
            W.TM.FEATURE_COLUMNS = save

    def members(tm, vm, tag, k, out):
        tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt24_%s" % tag, k)
        for s in SEEDS:
            out["base_R_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ct, phc, s)
        out["codex_R"] = codex(tr, va, wp[tm], FC)
        return tr, va
    # new layout DIAG10G with PFN refit (new context seeds)
    import env_extra_gpu  # noqa: F401
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4); assert torch.cuda.is_available()
    X = pfn[list(W.FEATURE_COLUMNS)].to_numpy(np.float32); y = pfn.sub_temp.to_numpy(); ixday = list(W.FEATURE_COLUMNS).index("day")
    SEQ = S6.sequences(lab)
    for k, fd in enumerate(fresh_folds(lab)):
        path = os.path.join(CK, "J_%d.csv" % k)
        if os.path.exists(path):
            continue
        tm, vm = masks(lab, fd)
        out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True)
        tr, va = members(tm, vm, "G", k, out)
        XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season
        ps = []
        for seed in range(49, 57):
            idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                           ignore_pretraining_limits=True, inference_precision=torch.float32)
            m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS)); del m; gc.collect(); torch.cuda.empty_cache()
        out["pfn_A"] = np.mean(ps[:4], 0); out["pfn_B"] = np.mean(ps[4:], 0)
        out["GRU"] = np.mean([S6.fit_predict(SEQ[tm], y[tm], wb[tm], SEQ[vm], sd, "cuda") for sd in (3, 4)], 0)
        out.to_csv(path, index=False); print("J %d done (val %d, train %d, pass-2 val rows %d)" % (k, vm.sum(), tm.sum(), int((lab.day[vm] >= 179).sum())), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261030)
    M = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.startswith("J_")], ignore_index=True)
    assert M.row_id.is_unique and len(M) == 9600, len(M)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True)
    g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1))
    mm = M.sort_values(["farm", "day", "hour"]).groupby(["farm", "day"]).in_temp.cummin().reindex(M.index)
    h = np.where(mm.isna(), 1, np.clip((mm - 10) / 2, 0, 1)); y = M.sub_temp.values
    rel, pw, r2 = [], [], []
    for s in SEEDS:
        for f in "AB":
            a = (.4 * M["base_R_%d" % s] + (.2 + .4 * (1 - g)) * M.codex_R + .4 * g * M["pfn_%s" % f]).values; b = a + .1 * g * h * (M.GRU.values - a)
            p = boot(M, a, b, rng); rel.append(r(b - y) / r(a - y) - 1); pw.append(p); L2 = M.day.values >= 179; r2.append(r((b - y)[L2]) / r((a - y)[L2]) - 1)
            fr = {fm: 100 * (r((b - y)[M.farm.values == fm]) / r((a - y)[M.farm.values == fm]) - 1) for fm in ("F13", "F47")}
            print("  seed %d PFN %s: %.4f -> %.4f (%+.2f%%) P %.4f | 2차 %+.2f%% | F13 %+.2f%% F47 %+.2f%%" % (s, f, r(a - y), r(b - y), 100 * rel[-1], p, 100 * r2[-1], fr["F13"], fr["F47"]))
    ok = all(x < 0 for x in rel) and max(pw) < .025
    print("  GRU alone %.3f | mean %+.2f%% P max %.4f | 2차 mean %+.2f%%" % (r(M.GRU - M.sub_temp), 100 * np.mean(rel), max(pw), 100 * np.mean(r2)))
    print("VERDICT GRU-DAYGATE10: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT24_SUM") != "1":
        run()
    summarize()
