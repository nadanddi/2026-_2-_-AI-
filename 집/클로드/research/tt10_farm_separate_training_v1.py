# -*- coding: utf-8 -*-
"""TT10: farm-separate training for temperature (2026-10-10 집 클로드, user: "F47만 학습 검증 나눠서 하면 어떻게 돼?").
Context: pass-2 validators show F47 RMSE ~2.3x F13 (td15); joint training (farm_id feature) never compared with
farm-only training for temperature (catalog grep).  Fixed before running.
 FS47: for F47 validation rows, BASE (seeds 7/101) and CODEX (726) refit on F47 training rows only; F13 rows unchanged.
 FS13: same for F13 (F47 rows unchanged).
 TabPFN member = stored joint predictions (refit too costly) -> partial separation; W40G-S blend otherwise identical.
 Caveat: joint REF can use the other farm's same-date labels (6.454 leak) -> comparison slightly tilted against FS.
k = 2 -> alpha .0125.  RULE (user temperature rule, scope ALL): every BASE seed x PFN family better on DIAG10 AND EXT10
  (all rows), farm x 5-day block bootstrap P(worse) < .0125 on both; EXT12/EL1 fail iff seed-mean worse and
  share(better) < .0125.  CODEX deterministic -> replicate axis = BASE seed x PFN family.
Reported: the changed farm's rows alone, pass-2 rows, in_temp < 6 rows.
Pre-result fix (crash, no output seen): season mapping needs both farms' training days -> computed once on the joint
fold (weather only, identical to REF), then rows split by farm.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt10_farm_separate_training_v1.py
"""
import os, sys, pathlib
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt10_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt10_season"
VALS = ("DIAG10", "EXT10", "EXT12", "EL1"); ALPHA = .0125
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    wb, wp = np.asarray(wb, float), np.asarray(wp, float)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm0, vm0 = common.split_mask(lab, fd)
            if not vm0.sum():
                continue
            # season index (training-day weather only, no labels) mapped once on the JOINT fold, as in REF
            tr0, va0 = W.season_fold(pfn[tm0], pfn[vm0], wv, "tt10_" + name, k)
            outs = []
            for farm in ("F13", "F47"):
                isf = (lab.farm == farm).to_numpy()
                tm, vm = tm0 & isf, vm0 & isf
                if not vm.sum():
                    continue
                tr = tr0[isf[tm0]]; va = va0[isf[vm0]]
                out = pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values})
                for s in (7, 101):
                    out["base_FS_%d" % s] = W.base_predict(lab[tm], lab[vm], wb[tm], ct, phc, s)
                save = W.TM.FEATURE_COLUMNS
                W.TM.FEATURE_COLUMNS = FC
                try:
                    out["codex_FS"] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wp[tm], 726)
                finally:
                    W.TM.FEATURE_COLUMNS = save
                outs.append(out)
            if outs:
                pd.concat(outs).to_csv(path, index=False); print("%s/%d done" % (name, k), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261012)
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))])
    G = G.merge(C, on=["validator", "row_id"], how="left"); assert G.codex_FS.notna().all()
    G["g"] = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    bl = lambda X, b, c, f: .4 * X[b] + (.2 + .4 * (1 - X.g)) * X[c] + .4 * X.g * X["pfn_%s" % f]
    for cand in ("F47", "F13"):
        print("\n==== FS%s (only %s rows use farm-only BASE/CODEX)" % (cand[1:], cand)); ok = True
        rel = {v: [] for v in VALS}; pw = {v: [] for v in VALS}
        for s in (7, 101):
            for f in "AB":
                line = "seed %3d PFN %s |" % (s, f)
                for v in VALS:
                    X = G[G.validator == v]; y = X.sub_temp.values; isf = (X.farm == cand).values
                    a = bl(X, "base_REF_%d" % s, "codex_REF", f).values
                    b = np.where(isf, bl(X, "base_FS_%d" % s, "codex_FS", f).values, a)
                    p = boot(X, a, b, rng); rel[v].append(r(b - y) / r(a - y) - 1); pw[v].append(p)
                    L = X.day.values >= 179
                    line += " %s %+.2f%% P %.3f (%s만 %.3f->%.3f, 2차 %s %.3f->%.3f) |" % (
                        v, 100 * rel[v][-1], p, cand, r((a - y)[isf]), r((b - y)[isf]), cand, r((a - y)[isf & L]), r((b - y)[isf & L]))
                print(line)
        for v in VALS:
            m = np.mean(rel[v])
            if v in ("DIAG10", "EXT10"):
                ok &= all(x < 0 for x in rel[v]) and max(pw[v]) < ALPHA
            else:
                ok &= not (m > 0 and (1 - min(pw[v])) < ALPHA)
            print("  %-6s seed-mean %+.2f%%  P(worse) max %.4f" % (v, 100 * m, max(pw[v])))
        print("VERDICT FS%s: %s" % (cand[1:], "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT10_SUM") != "1":
        run()
    summarize()
