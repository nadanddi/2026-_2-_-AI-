"""TT27: DECISIVE confirmation of the TC1 chain-aware early-hour corrector on a never-used layout L (2026-10-11 집 클로드).
TC1c (ablation, d72bba6): CHAIN features beat the no-chain corrector in every cell of DIAG10, EL1, F..K (pooled P .0102 <
.0125) while the no-chain corrector alone does nothing -> the gain is the chain history.  Fixed before running; this run decides.
Layout DIAG10L (never used): per farm sorted training days, 8-DAY chunks (i // 8), fold = chunk % 10; training excludes
  same farm +-1 and other farm +-5 days.  NEW BASE seeds 1414 / 1515, CODEX 726, TabPFN season member refit on GPU with NEW
  context seeds 65..72 (A 65-68, B 69-72).  No GRU.
Candidate: TC1c CHAIN corrector exactly as in tc1c (DIAG10 stored OOF training source with the L fold's days purged, same
  farm +-1 / other farm +-5; LightGBM 100 / .03 / 15 leaves / min_child 100, seed 1; hours 0-7; shrink 0.5), training target
  built from DIAG10 members with the same BASE-seed index and PFN family.
PASS iff on layout L all rows (rough days excluded): all 4 cells (BASE seed x PFN family) better than W40G-S AND seed-mean
  farm x 8-day block bootstrap P(worse) < .025 (k = 1).  Reported: hours 0-7, pass-2, F13/F47, in_temp <= 10, NOCHAIN.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt27_chain_corrector_layoutL_v1.py
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
_sp7 = _iu.spec_from_file_location('s7', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ts7_sequence_variants_v1.py')); S7 = _iu.module_from_spec(_sp7); _sp7.loader.exec_module(S7)
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt27_ckpt")
CK13 = os.path.join(env.LOCAL, "tt13_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt27_season"
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
COLS = ["in_co2", "in_hum", "in_temp"]; SEEDS = (1414, 1515); ALPHA = .0125
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
            out[(i // 8) % 10][f].add(int(d))
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
        tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt27_%s" % tag, k)
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
    for k, fd in enumerate(fresh_folds(lab)):
        path = os.path.join(CK, "L_%d.csv" % k)
        if os.path.exists(path):
            continue
        tm, vm = masks(lab, fd)
        out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True)
        tr, va = members(tm, vm, "G", k, out)
        XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season
        ps = []
        for seed in range(65, 73):
            idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                           ignore_pretraining_limits=True, inference_precision=torch.float32)
            m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS)); del m; gc.collect(); torch.cuda.empty_cache()
        out["pfn_A"] = np.mean(ps[:4], 0); out["pfn_B"] = np.mean(ps[4:], 0)
        out.to_csv(path, index=False); print("L %d done (val %d, train %d, pass-2 val rows %d)" % (k, vm.sum(), tm.sum(), int((lab.day[vm] >= 179).sum())), flush=True)




def summarize():
    import importlib.util as iu
    here = os.path.dirname(os.path.abspath(__file__))
    sp = iu.spec_from_file_location("tc1", os.path.join(here, "tc1_chain_history_early_hours_v1.py")); T = iu.module_from_spec(sp); sp.loader.exec_module(T)
    sp = iu.spec_from_file_location("tc1b", os.path.join(here, "tc1b_chain_history_common_train_v1.py")); B = iu.module_from_spec(sp); sp.loader.exec_module(B)
    sp = iu.spec_from_file_location("tc1c", os.path.join(here, "tc1c_chain_ablation_v1.py")); C = iu.module_from_spec(sp); sp.loader.exec_module(C)
    rng = np.random.default_rng(20261117)
    CT = T.chain_table(); D = T.layout_frame("DIAG10", CT)
    parts = []
    for k in range(10):
        Mk = pd.read_csv(os.path.join(CK, "L_%d.csv" % k)); Mk["fold"] = k; parts.append(Mk)
    M = pd.concat(parts, ignore_index=True); assert M.row_id.is_unique and len(M) == 9600, len(M)
    M = M.merge(CT, on=["farm", "day"], how="left"); M["jump"] = M.in_temp - M.ch_in23
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])].reset_index(drop=True); y = M.sub_temp.values
    cells, ok, rows = [], True, []
    for bi, s in enumerate(SEEDS):
        for fam in "AB":
            a = B.w40(M, "base_R_%d" % s, "codex_R", fam)
            c = C.corrected(M, a, D, bi, fam, C.FB); n = C.corrected(M, a, D, bi, fam, C.FA)
            rel = r(c - y) / r(a - y) - 1; cells.append(rel); ok &= rel < 0; rows.append((a, c))
            m7 = M.hour.values <= 7; m2 = M.day.values >= 179; cold = (M.in_temp.values <= 10)
            print("  BASE %d PFN %s: CHAIN %+.2f%% (NOCHAIN %+.2f%%) | hours0-7 %+.2f%% | pass-2 %+.2f%% | F13 %+.2f%% F47 %+.2f%% | in_temp<=10 %+.2f%%" % (
                s, fam, 100 * rel, 100 * (r(n - y) / r(a - y) - 1), 100 * (r((c - y)[m7]) / r((a - y)[m7]) - 1), 100 * (r((c - y)[m2]) / r((a - y)[m2]) - 1),
                *(100 * (r((c - y)[M.farm.values == f]) / r((a - y)[M.farm.values == f]) - 1) for f in ("F13", "F47")),
                100 * (r((c - y)[cold]) / r((a - y)[cold]) - 1)), flush=True)
    a = np.mean([x[0] for x in rows], 0); c = np.mean([x[1] for x in rows], 0)
    cl = (M.farm + "_" + (M.day // 8).astype(str)).values
    dd = pd.Series((c - y) ** 2 - (a - y) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd))); p = float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())
    print("seed-mean P(worse) %.4f (%d blocks) -> VERDICT TC1 chain corrector: %s" % (p, len(dd), "PASS" if ok and p < .025 else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT27_SUM") != "1":
        run()
    summarize()
