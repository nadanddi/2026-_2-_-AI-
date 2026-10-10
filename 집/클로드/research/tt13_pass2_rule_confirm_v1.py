# -*- coding: utf-8 -*-
"""TT13: temperature PASS-2 protocol confirmation (2026-10-10 집 클로드, user: "2차 구간 판정 규칙으로 실험해봐").
User decision 10-10: temperature may use the EC-style pass-2 protocol (memory skill-not-luck).  Motivation (post hoc,
two runs): TT11 S5W1 improved pass-2 rows in all 16 cells; TT12 NOH0+NOHEAT pass-2 -0.9% on the fresh layout while
all-row change was 0.  Everything below is fixed BEFORE running; the outcome decides, whatever it is.
Protocol: candidate applied to pass-2 rows only (day >= 179; all 60 evaluation days are pass 2) - judged on pass-2
  rows only; the 30 strict CO2-rough days (6.437) are excluded from scoring (inputs corrupted, TD17).
Candidates (k = 2, alpha .0125):
  P1 NOH0+NOHEAT: CODEX LGB residual without *_h0 and act_heating* (BASE, PFN unchanged).
  P2 S5W1: training inputs in_co2/in_hum/in_temp on the 30 rough days repaired by a centred 5-row mean (training rows
     only; held-out rows keep original inputs; user ruling 10-10) + those days' weight back to 1; BASE and CODEX refit;
     PFN member unrepaired (as screened in TT11).
Validators:
  DIAG10G (NEW, never used, decisive): per farm sorted training days, chunk = (i + 1) // 5, fold = (7 * chunk + 3) % 10;
     training excludes same farm +-1 and other farm +-5 days.  BASE seeds 404 / 505 (new), CODEX 726, TabPFN season
     member refit on GPU as TK2 with NEW context seeds 9..16 (families A'' = 9-12, B'' = 13-16).
  DIAG10 (original) and EL1: BASE refit with seeds 404 / 505 (ref and P2), CODEX ref/P1/P2 refit, stored PFN A/B.
PASS iff (a) DIAG10G pass-2: all 4 cells (seed x family) better AND farm x 5-day block bootstrap P(worse) < .0125 in
  every cell; (b) DIAG10 pass-2 and EL1: all 4 cells better (direction only).  EXT10/EXT12 not run (reported in TT11/12).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt13_pass2_rule_confirm_v1.py
"""
import os, sys, pathlib, gc
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt13_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt13_season"
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
COLS = ["in_co2", "in_hum", "in_temp"]; SEEDS = (404, 505); ALPHA = .0125
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
            out[(7 * ((i + 1) // 5) + 3) % 10][f].add(int(d))
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
    labS, pfnS, *_ = build_world_with(repaired_loader(W.TM.ORIG, 5))
    assert (labS.row_id.values == lab.row_id.values).all()
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    FC1 = [c for c in FC if not c.endswith("_h0") and not (c == "act_heating" or c.startswith("act_heating_"))]

    def codex(tr, va, wtr, cols):
        save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = cols
        try:
            return W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wtr, 726)
        finally:
            W.TM.FEATURE_COLUMNS = save

    def members(tm, vm, tag, k, out):
        tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt13_%s" % tag, k)
        trS, _ = W.season_fold(pfnS[tm], pfn[vm], wv, "tt13S_%s" % tag, k)
        for s in SEEDS:
            out["base_R_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ct, phc, s)
            out["base_P2_%d" % s] = W.base_predict(labS[tm], lab[vm], w1[tm], ct, phc, s)
        out["codex_R"] = codex(tr, va, wp[tm], FC)
        out["codex_P1"] = codex(tr, va, wp[tm], FC1)
        out["codex_P2"] = codex(trS, va, w1[tm], FC)
        return tr, va
    # old validators (DIAG10, EL1): new BASE seeds, stored PFN
    for name, folds in sets:
        if name not in ("DIAG10", "EL1"):
            continue
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "old_%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            out = pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values})
            members(tm, vm, name, k, out)
            out.to_csv(path, index=False); print("old %s/%d done" % (name, k), flush=True)
    # new layout DIAG10G with PFN refit (new context seeds)
    import env_extra_gpu  # noqa: F401
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4); assert torch.cuda.is_available()
    X = pfn[list(W.FEATURE_COLUMNS)].to_numpy(np.float32); y = pfn.sub_temp.to_numpy(); ixday = list(W.FEATURE_COLUMNS).index("day")
    for k, fd in enumerate(fresh_folds(lab)):
        path = os.path.join(CK, "G_%d.csv" % k)
        if os.path.exists(path):
            continue
        tm, vm = masks(lab, fd)
        out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True)
        tr, va = members(tm, vm, "G", k, out)
        XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season
        ps = []
        for seed in range(9, 17):
            idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                           ignore_pretraining_limits=True, inference_precision=torch.float32)
            m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS)); del m; gc.collect(); torch.cuda.empty_cache()
        out["pfn_A"] = np.mean(ps[:4], 0); out["pfn_B"] = np.mean(ps[4:], 0)
        out.to_csv(path, index=False); print("G %d done (val %d, train %d, pass-2 val rows %d)" % (k, vm.sum(), tm.sum(), int((lab.day[vm] >= 179).sum())), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261015)
    Gn = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.startswith("G_")], ignore_index=True)
    assert Gn.row_id.is_unique and len(Gn) == 9600, len(Gn)
    Gn["validator"] = "DIAG10G"
    T1 = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    Old = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.startswith("old_")])
    Old = Old.merge(T1[["validator", "row_id", "farm", "day", "hour", "sub_temp", "in_temp", "pfn_A", "pfn_B"]], on=["validator", "row_id"], how="left")
    assert Old.pfn_A.notna().all()
    verdict = {}
    for cand, bc, cc in (("P1", "base_R_%d", "codex_P1"), ("P2", "base_P2_%d", "codex_P2")):
        print("\n==== %s (pass-2 rows, rough days excluded)" % cand); ok = True
        for v, D in (("DIAG10G", Gn), ("DIAG10", Old[Old.validator == "DIAG10"]), ("EL1", Old[Old.validator == "EL1"])):
            X = D[(D.day >= 179).values & np.array([(f, d) not in RD for f, d in zip(D.farm, D.day)])].copy()
            g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1)); y = X.sub_temp.values
            rel, pw = [], []
            for s in SEEDS:
                for f in "AB":
                    bl = lambda b_, c_: (.4 * X[b_ % s] + (.2 + .4 * (1 - g)) * X[c_] + .4 * g * X["pfn_%s" % f]).values
                    a, b = bl("base_R_%d", "codex_R"), bl(bc, cc); p = boot(X, a, b, rng)
                    rel.append(r(b - y) / r(a - y) - 1); pw.append(p)
                    fr = {fm: 100 * (r((b - y)[X.farm.values == fm]) / r((a - y)[X.farm.values == fm]) - 1) for fm in ("F13", "F47")}
                    print("  %-7s seed %d PFN %s: %.4f -> %.4f (%+.2f%%) P %.4f | F13 %+.2f%% F47 %+.2f%%" % (v, s, f, r(a - y), r(b - y), 100 * rel[-1], p, fr["F13"], fr["F47"]))
            o = all(x < 0 for x in rel) and (max(pw) < ALPHA if v == "DIAG10G" else True)
            ok &= o
            print("  %-7s rows %d days %d | seed-mean %+.2f%% P max %.4f -> %s" % (v, len(X), X.groupby(["farm", "day"]).ngroups, 100 * np.mean(rel), max(pw), "pass" if o else "fail"))
        verdict[cand] = ok
        print("VERDICT %s: %s" % (cand, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT13_SUM") != "1":
        run()
    summarize()
