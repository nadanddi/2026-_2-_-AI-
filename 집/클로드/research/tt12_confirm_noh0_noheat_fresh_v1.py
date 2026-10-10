# -*- coding: utf-8 -*-
"""TT12: CONFIRMATION of the combined candidate NOH0+NOHEAT on a FRESH layout (2026-10-10 집 클로드, user: "확인 실험 진행해봐").
Background: NOH0 (14 *_h0) and NOHEAT (8 act_heating*) removed from the CODEX LGB residual were each consistent but not
significant (TT9, 6.494).  Re-scored with the 30 strict CO2-rough days (6.437, rw1_rough_days_v1) removed from SCORING
(new validation protocol decided with the user 10-10; justification is training-internal: those days' indoor inputs
are corrupted - TD17 - not their error size), both improved on all four validators (td18, post hoc).  This run is
the one confirmation, fixed before running; it is decided by this rule whatever the outcome.
Candidate C: CODEX LGB residual features = FEATURE_COLUMNS minus *_h0 minus act_heating* (physics Ridge unchanged);
  BASE, TabPFN, gate, weights unchanged.  Reference = same pipeline with full CODEX features.
FRESH layout DIAG10F (never used): per farm, sorted training days, chunk index = (i + 3) // 5 (boundaries shifted by 3
  days), fold = (3 * chunk) % 10.  Training excludes same farm +-1 day (as before) AND the other farm +-5 days of every
  validation day (other-farm same-date leak, 6.454).  Per fold: BASE seeds 202 / 303 (new), CODEX ref + cand (726),
  TabPFN season member refit on GPU exactly as TK2 (2,000-row weighted context, n_estimators 4, float32, seeds 1..8);
  families A' = mean of seeds 1-4, B' = 5-8.
Other validators (already seen): EXT10 / EXT12 / EL1 / DIAG10 use the stored TT1 members (BASE 7/101, PFN A/B, CODEX
  ref) + candidate CODEX refit here.
Scoring: rows of the 30 strict rough days excluded everywhere.  k = 1, alpha .025.
PASS iff (a) DIAG10F: all 4 cells (BASE seed x family) better and farm x 5-day block bootstrap P(worse) < .025 for every
  cell, AND (b) EXT10: all 4 cells better, P(worse) < .025;  (c) EXT12 / EL1 / DIAG10: fail iff seed-mean worse and
  share(better) < .025.  Reported: pass-2 rows, F13/F47, in_temp < 6 rows.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt12_confirm_noh0_noheat_fresh_v1.py
"""
import os, sys, pathlib, gc
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt12_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt12_season"
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
ALPHA = .025
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def fresh_folds(lab):
    out = [{f: set() for f in ("F13", "F47")} for _ in range(10)]
    for f in ("F13", "F47"):
        days = sorted(lab[lab.farm == f].day.unique())
        for i, d in enumerate(days):
            out[(3 * ((i + 3) // 5)) % 10][f].add(int(d))
    return out


def masks(lab, fd):
    val = np.zeros(len(lab), bool); bad = np.zeros(len(lab), bool)
    d = lab.day.values
    for f, ds in fd.items():
        isf = (lab.farm == f).values
        ds = np.array(sorted(ds))
        val |= isf & np.isin(d, ds)
        near_same = np.isin(d, np.unique((ds[:, None] + np.arange(-1, 2)).ravel()))
        near_oth = np.isin(d, np.unique((ds[:, None] + np.arange(-5, 6)).ravel()))
        bad |= (isf & near_same) | (~isf & near_oth)
    return ~bad & ~val, val


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    wb, wp = np.asarray(wb, float), np.asarray(wp, float)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    FCC = [c for c in FC if not c.endswith("_h0") and not (c == "act_heating" or c.startswith("act_heating_"))]
    print("candidate CODEX features %d (removed %s)" % (len(FCC), sorted(set(FC) - set(FCC))), flush=True)

    def codex(tr, va, wtr, cols):
        save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = cols
        try:
            return W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wtr, 726)
        finally:
            W.TM.FEATURE_COLUMNS = save
    # (1) already-seen validators: candidate CODEX only
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "old_%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt12_" + name, k)
            pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values, "codex_C": codex(tr, va, wp[tm], FCC)}).to_csv(path, index=False)
            print("old %s/%d done" % (name, k), flush=True)
    # (2) fresh layout: everything refit
    import env_extra_gpu  # noqa: F401
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(4); assert torch.cuda.is_available()
    X = pfn[list(W.FEATURE_COLUMNS)].to_numpy(np.float32); y = pfn.sub_temp.to_numpy(); ixday = list(W.FEATURE_COLUMNS).index("day")
    for k, fd in enumerate(fresh_folds(lab)):
        path = os.path.join(CK, "fresh_%d.csv" % k)
        if os.path.exists(path):
            continue
        tm, vm = masks(lab, fd)
        tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt12_F", k)
        out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True)
        for s in (202, 303):
            out["base_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ct, phc, s)
        out["codex_R"] = codex(tr, va, wp[tm], FC)
        out["codex_C"] = codex(tr, va, wp[tm], FCC)
        XS = X[tm].copy(); VS = X[vm].copy(); XS[:, ixday] = tr.season; VS[:, ixday] = va.season
        ps = []
        for seed in range(1, 9):
            idx = np.random.default_rng(seed).choice(int(tm.sum()), min(2000, int(tm.sum())), replace=False, p=wp[tm] / wp[tm].sum())
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                           ignore_pretraining_limits=True, inference_precision=torch.float32)
            m.fit(XS[idx], y[tm][idx]); ps.append(m.predict(VS))
            del m; gc.collect(); torch.cuda.empty_cache()
        out["pfn_A"] = np.mean(ps[:4], 0); out["pfn_B"] = np.mean(ps[4:], 0)
        out.to_csv(path, index=False); print("fresh %d done (val rows %d, train rows %d)" % (k, vm.sum(), tm.sum()), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def judge(X, seeds, fams, bcol, rcol, ccol, rng, label):
    X = X[[(f, d) not in RD for f, d in zip(X.farm, X.day)]].copy()
    g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1)); y = X.sub_temp.values
    rel, pw, lines = [], [], []
    for s in seeds:
        for f in fams:
            bl = lambda c: (.4 * X[bcol % s] + (.2 + .4 * (1 - g)) * X[c] + .4 * g * X["pfn_%s" % f]).values
            a, b = bl(rcol), bl(ccol); p = boot(X, a, b, rng)
            rel.append(r(b - y) / r(a - y) - 1); pw.append(p)
            L = X.day.values >= 179; c6 = X.in_temp.values < 6
            fr = {fm: 100 * (r((b - y)[X.farm.values == fm]) / r((a - y)[X.farm.values == fm]) - 1) for fm in ("F13", "F47")}
            lines.append("seed %d PFN %s: %+.2f%% P %.4f | 2차 %+.2f%% | F13 %+.2f%% F47 %+.2f%% | <6C %s" % (
                s, f, 100 * rel[-1], p, 100 * (r((b - y)[L]) / r((a - y)[L]) - 1), fr["F13"], fr["F47"],
                "%.3f->%.3f" % (r((a - y)[c6]), r((b - y)[c6])) if c6.any() else "-"))
    print("\n-- %s (rough days excluded, %d rows)" % (label, len(X)))
    for l in lines:
        print("   " + l)
    return rel, pw


def summarize():
    rng = np.random.default_rng(20261014)
    Fr = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.startswith("fresh_")], ignore_index=True)
    assert Fr.row_id.is_unique and len(Fr) == 9600, len(Fr)
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    Cc = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.startswith("old_")])
    G = G.merge(Cc, on=["validator", "row_id"], how="left"); assert G.codex_C.notna().all()
    ok = True
    rel, pw = judge(Fr, (202, 303), "AB", "base_%d", "codex_R", "codex_C", rng, "DIAG10F (fresh, PRIMARY)")
    okf = all(x < 0 for x in rel) and max(pw) < ALPHA; ok &= okf
    print("   => %s" % ("pass" if okf else "fail"))
    for v in ("EXT10", "EXT12", "EL1", "DIAG10"):
        rel, pw = judge(G[G.validator == v], (7, 101), "AB", "base_REF_%d", "codex_REF", "codex_C", rng, v + (" (PRIMARY)" if v == "EXT10" else ""))
        if v == "EXT10":
            o = all(x < 0 for x in rel) and max(pw) < ALPHA
        else:
            o = not (np.mean(rel) > 0 and (1 - min(pw)) < ALPHA)
        ok &= o; print("   => seed-mean %+.2f%%, P(worse) max %.4f, %s" % (100 * np.mean(rel), max(pw), "pass" if o else "fail"))
    print("\nVERDICT NOH0+NOHEAT: %s" % ("PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT12_SUM") != "1":
        run()
    summarize()
