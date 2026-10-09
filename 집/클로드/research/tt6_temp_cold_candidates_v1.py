# -*- coding: utf-8 -*-
"""TT6: two cold-side temperature candidates from TD7-TD10 (2026-10-10 집 클로드, user: "지금까지 나온 의견 중 해볼 만한 거
전부 돌려보자").  Fixed before running (after plan-stage critic; LC residual-corrector dropped = 6b.21 structure).
Reference W40G-S = 0.4 BASE + (0.2 + 0.4(1-g)) CODEX + 0.4 g PFN,  g = clip((in_temp-8)/2, 0, 1)  (TT1 checkpoints, REF
members = TK2-identical).
 CW  (scope COLD, no refit): (0.4 - 0.2(1-g)) BASE + (0.2 + 0.6(1-g)) CODEX + 0.4 g PFN  (cold rows 0.2/0.8 instead of
     0.4/0.6; trees overpredict cold rows 6b.25, TD8 bias +.25~+.46 at 4-8 C).  Caveat: G_C2 share was chosen post hoc on
     EXT10/EXT12 (6.70) and 6.75 supported 0.4/0.6 - this is a further step in the same direction.
 CWT (scope ALL, refit BASE seeds 7/101 and CODEX seed 726): F13/F47 training-row weights x3 where in_temp < 10 (g < 1),
     on top of the existing weights (wb, wp).  PFN unchanged.
k = 2 -> alpha .0125.
USER RULE (memory skill-not-luck, temperature scope rule):
  COLD: every seed x PFN family better on EXT10 + EXT10 farm x 5-day block bootstrap P(worse) < .0125;
  ALL : same on DIAG10 AND EXT10;
  others (DIAG10/EXT12/EL1 as applicable): fail iff seed-mean worse AND bootstrap share(candidate better) < .0125.
STRICT (critic additions, reported alongside): also EXT12 every combo better with P(worse) < .0125, and no other
  validator worse on the seed-mean.  Reported: pass-2 rows, rows in_temp < 6, number/day-concentration of up-weighted rows.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt6_temp_cold_candidates_v1.py   (sum = summary only)
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
sys.argv = ["x"]
TKDIR = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1")
sys.path.insert(0, TKDIR)
import world as W  # noqa: E402  (Codex TK2 world, read-only)
import common  # noqa: E402
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt6_ckpt")
SEEDS = (7, 101); FAMS = ("A", "B"); VALS = ("DIAG10", "EXT10", "EXT12", "EL1"); ALPHA = .0125
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def refit():
    os.makedirs(CK, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    cold = (lab.in_temp < 10).to_numpy()
    wb2, wp2 = np.where(cold, 3.0, 1.0) * np.asarray(wb, float), np.where(cold, 3.0, 1.0) * np.asarray(wp, float)
    dd = lab[cold].groupby(["farm", "day"]).size()
    print("x3 rows %d (%.1f%%), days %d, top-10 days hold %.0f%% of them" % (cold.sum(), 100 * cold.mean(), len(dd), 100 * dd.nlargest(10).sum() / dd.sum()), flush=True)
    FC = list(W.FEATURE_COLUMNS)
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt1_" + name, k)
            out = lab.loc[vm, ["row_id"]].copy().reset_index(drop=True)
            out["validator"] = name
            for s in SEEDS:
                out["base_CWT_%d" % s] = W.base_predict(lab[tm], lab[vm], wb2[tm], ct, phc, s)
            saveFC = W.TM.FEATURE_COLUMNS
            W.TM.FEATURE_COLUMNS = [c if c != "day" else "season" for c in FC]
            try:
                out["codex_CWT"] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wp2[tm], 726)
            finally:
                W.TM.FEATURE_COLUMNS = saveFC
            out.to_csv(path, index=False)
            print("%s/%d done" % (name, k), flush=True)


def blend(X, s, f, cand):
    g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1))
    if cand == "REF":
        return .4 * X["base_REF_%d" % s] + (.2 + .4 * (1 - g)) * X.codex_REF + .4 * g * X["pfn_%s" % f]
    if cand == "CW":
        return (.4 - .2 * (1 - g)) * X["base_REF_%d" % s] + (.2 + .6 * (1 - g)) * X.codex_REF + .4 * g * X["pfn_%s" % f]
    return .4 * X["base_CWT_%d" % s] + (.2 + .4 * (1 - g)) * X.codex_CWT + .4 * g * X["pfn_%s" % f]


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series(((b - X.sub_temp) ** 2 - (a - X.sub_temp) ** 2).values).groupby(cl).agg(["sum", "count"])
    sm, n = dd["sum"].values, dd["count"].values
    idx = rng.integers(0, len(sm), (20000, len(sm)))
    return float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())


def summarize():
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    have_cwt = os.path.isdir(CK) and len(os.listdir(CK)) >= 24
    if have_cwt:
        C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
        G = G.merge(C, on=["validator", "row_id"], how="left")
        assert G.codex_CWT.notna().all()
    rng = np.random.default_rng(20261010)
    for cand, prim in (("CW", ("EXT10",)), ("CWT", ("DIAG10", "EXT10"))):
        if cand == "CWT" and not have_cwt:
            print("\nCWT: checkpoints incomplete"); continue
        print("\n==== %s (primary %s)" % (cand, "+".join(prim)))
        user_ok, strict_ok = True, True
        rel = {v: [] for v in VALS}; pw = {v: [] for v in VALS}
        for s in SEEDS:
            for f in FAMS:
                line = "seed %3d PFN %s |" % (s, f)
                for v in VALS:
                    X = G[G.validator == v]
                    a, b = blend(X, s, f, "REF"), blend(X, s, f, cand)
                    ra, rb = r(a - X.sub_temp), r(b - X.sub_temp); rel[v].append(rb / ra - 1)
                    L = (X.day >= 179).values; c6 = (X.in_temp < 6).values
                    p = boot(X, a, b, rng); pw[v].append(p)
                    line += " %s %+.2f%% P %.3f (p2 %+.2f%%, <6C %s) |" % (v, 100 * (rb / ra - 1), p, 100 * (r((b - X.sub_temp)[L]) / r((a - X.sub_temp)[L]) - 1),
                                                                         "%.3f->%.3f n%d" % (r((a - X.sub_temp)[c6]), r((b - X.sub_temp)[c6]), c6.sum()) if c6.sum() else "-")
                print(line)
        for v in VALS:
            m = np.mean(rel[v]); allb = all(x < 0 for x in rel[v]); pmax = max(pw[v]); pmin = min(pw[v])
            if v in prim:
                user_ok &= allb and pmax < ALPHA
            else:
                user_ok &= not (m > 0 and (1 - pmin) < ALPHA)
            if v == "EXT12":
                strict_ok &= allb and pmax < ALPHA
            strict_ok &= (m <= 0) if v not in prim else (allb and pmax < ALPHA)
            print("  %-6s seed-mean %+.2f%%  all better %s  P(worse) max %.4f" % (v, 100 * m, allb, pmax))
        print("VERDICT %s: user rule %s | strict %s" % (cand, "PASS" if user_ok else "FAIL", "PASS" if strict_ok else "FAIL"))


if __name__ == "__main__":
    if MODE != "sum":
        refit()
    summarize()
