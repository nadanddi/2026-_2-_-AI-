# -*- coding: utf-8 -*-
"""TT23: CONFIRMATION of PW1 on a never-used layout (2026-10-10 집 클로드).  TS1 screen (layouts F/G/H, already consumed):
PW1 = CODEX with pass-2 training rows weighted x3 AND P1 features (no *_h0, no act_heating*) improved pass-2 rows in every
cell of all three layouts (-0.57..-0.66%), pooled P(worse) .0065 -> PROMOTED.  PW alone did nothing, so the effect is P1's.
Fixed before running; this run decides.
Layout DIAG10I (never used): per farm sorted training days, chunk = (i + 4) // 5, fold = (3 * chunk + 7) % 10; training
  excludes same farm +-1 and other farm +-5 days.  New BASE seeds 808 / 909, CODEX 726, TabPFN season member refit on GPU as
  TK2 with NEW context seeds 41..48 (families A = 41-44, B = 45-48).
Candidate PW1 (judged); P1 (= P1 features without the x3 weighting) reported only.
PASS iff on DIAG10I pass-2 rows (rough days excluded): all 4 cells (BASE seed x family) better and farm x 5-day block
  bootstrap P(worse) < .025 (k = 1).  All rows reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt23_pw1_confirm_v1.py
"""
import os, sys, pathlib, gc
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt23_ckpt")
CK13 = os.path.join(env.LOCAL, "tt13_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt23_season"
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
COLS = ["in_co2", "in_hum", "in_temp"]; SEEDS = (808, 909); ALPHA = .0125
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
            out[(3 * ((i + 4) // 5) + 7) % 10][f].add(int(d))
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
        tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt23_%s" % tag, k)
        for s in SEEDS:
            out["base_R_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ct, phc, s)
        out["codex_R"] = codex(tr, va, wp[tm], FC)
        out["codex_P1"] = codex(tr, va, wp[tm], FC1)
        out["codex_PW1"] = codex(tr, va, wp[tm] * np.where(lab.day.values[tm] >= 179, 3.0, 1.0), FC1)
        return tr, va
    # new layout DIAG10G with PFN refit (new context seeds)
    import env_extra_gpu  # noqa: F401
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4); assert torch.cuda.is_available()
    X = pfn[list(W.FEATURE_COLUMNS)].to_numpy(np.float32); y = pfn.sub_temp.to_numpy(); ixday = list(W.FEATURE_COLUMNS).index("day")
    for k, fd in enumerate(fresh_folds(lab)):
        path = os.path.join(CK, "I_%d.csv" % k)
        if os.path.exists(path):
            continue
        tm, vm = masks(lab, fd)
        out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True)
        tr, va = members(tm, vm, "G", k, out)
        XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season
        ps = []
        for seed in range(41, 49):
            idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                           ignore_pretraining_limits=True, inference_precision=torch.float32)
            m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS)); del m; gc.collect(); torch.cuda.empty_cache()
        out["pfn_A"] = np.mean(ps[:4], 0); out["pfn_B"] = np.mean(ps[4:], 0)
        out.to_csv(path, index=False); print("I %d done (val %d, train %d, pass-2 val rows %d)" % (k, vm.sum(), tm.sum(), int((lab.day[vm] >= 179).sum())), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261022)
    M = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.startswith("I_")], ignore_index=True)
    assert M.row_id.is_unique and len(M) == 9600, len(M)
    M = M[np.array([(f, d) not in RD for f, d in zip(M.farm, M.day)])]
    verdict = None
    for cand in ("PW1", "P1"):
        print("\n==== %s%s" % (cand, " (judged)" if cand == "PW1" else " (reported)"))
        for scope, sel in (("pass-2", M.day.values >= 179), ("all rows", np.ones(len(M), bool))):
            X = M[sel].copy(); g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1)); y = X.sub_temp.values
            rel, pw = [], []
            for s in SEEDS:
                for f in "AB":
                    bl = lambda c: (.4 * X["base_R_%d" % s] + (.2 + .4 * (1 - g)) * X[c] + .4 * g * X["pfn_%s" % f]).values
                    a, b = bl("codex_R"), bl("codex_%s" % cand); p = boot(X, a, b, rng)
                    rel.append(r(b - y) / r(a - y) - 1); pw.append(p)
                    fr = {fm: 100 * (r((b - y)[X.farm.values == fm]) / r((a - y)[X.farm.values == fm]) - 1) for fm in ("F13", "F47")}
                    print("  %-8s seed %d PFN %s: %+.2f%% P %.4f | F13 %+.2f%% F47 %+.2f%%" % (scope, s, f, 100 * rel[-1], p, fr["F13"], fr["F47"]))
            o = all(x < 0 for x in rel) and max(pw) < .025
            print("  %-8s rows %d mean %+.2f%% P max %.4f%s" % (scope, len(X), 100 * np.mean(rel), max(pw), (" -> " + ("pass" if o else "fail")) if cand == "PW1" and scope == "pass-2" else ""))
            if cand == "PW1" and scope == "pass-2":
                verdict = o
    print("\nVERDICT PW1: %s" % ("PASS" if verdict else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT23_SUM") != "1":
        run()
    summarize()
