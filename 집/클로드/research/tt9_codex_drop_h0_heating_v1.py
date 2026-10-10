# -*- coding: utf-8 -*-
"""TT9: step 2 of feature re-adjustment - confirm the two FA1 screening candidates (2026-10-10 집 클로드).
FA1 (CODEX-only group ablation, DIAG10/EXT10/EXT12, BASE seed 7 + PFN A) flagged H0 (all 14 *_h0 = value at hour 0,
crosses midnight stitching; DIAG10 -0.24%, EXT10 -0.23%, EXT12 -0.20%) and heating (8 act_heating* columns; -0.07 /
-0.79 / -0.03%).  Both were SELECTED on DIAG10/EXT10/EXT12 -> winner's curse; EL1 was NOT used in FA1.
Candidates (CODEX LGB residual features only, physics Ridge unchanged, exactly as screened):  NOH0, NOHEAT.  k = 2 ->
alpha .0125.  CODEX is deterministic (seed does not change it - TT8 critic), so the replicate axis is BASE seed {7,101}
(stored TT1 members, bagged LGB) x PFN family {A,B}.
RULE (fixed before running): user temperature rule, scope ALL: every BASE seed x family better on DIAG10 AND EXT10 with
  farm x 5-day block bootstrap P(worse) < .0125 on both; EXT12 fail iff seed-mean worse and share(better) < .0125;
  PLUS (selection guard) EL1 - unused in screening - must be better on the seed-mean with P(worse) < .05.
Reported: pass-2 rows, in_temp < 6 rows.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt9_codex_drop_h0_heating_v1.py
"""
import os, sys, pathlib
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt9_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt9_season"
VALS = ("DIAG10", "EXT10", "EXT12", "EL1"); ALPHA = .0125
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    wp = np.asarray(wp, float)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    drop = {"NOH0": [c for c in FC if c.endswith("_h0")], "NOHEAT": [c for c in FC if c == "act_heating" or c.startswith("act_heating_")]}
    print({k: len(v) for k, v in drop.items()}, flush=True)
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt9_" + name, k)
            tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
            out = pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values})
            save = W.TM.FEATURE_COLUMNS
            try:
                for nm, cols in drop.items():
                    W.TM.FEATURE_COLUMNS = [c for c in FC if c not in cols]
                    out["codex_" + nm] = W.TM.codex_fit_predict(tr, va, wp[tm], 726)
            finally:
                W.TM.FEATURE_COLUMNS = save
            out.to_csv(path, index=False); print("%s/%d done" % (name, k), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261011)
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))])
    G = G.merge(C, on=["validator", "row_id"], how="left"); assert G.codex_NOH0.notna().all()
    G["g"] = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    bl = lambda X, s, c, f: .4 * X["base_REF_%d" % s] + (.2 + .4 * (1 - X.g)) * X[c] + .4 * X.g * X["pfn_%s" % f]
    for cand in ("NOH0", "NOHEAT"):
        print("\n==== %s" % cand); ok = True; rel = {v: [] for v in VALS}; pw = {v: [] for v in VALS}
        for s in (7, 101):
            for f in ("A", "B"):
                line = "seed %3d PFN %s |" % (s, f)
                for v in VALS:
                    X = G[G.validator == v]; y = X.sub_temp.values
                    a, b = bl(X, s, "codex_REF", f).values, bl(X, s, "codex_" + cand, f).values
                    p = boot(X, a, b, rng); rel[v].append(r(b - y) / r(a - y) - 1); pw[v].append(p)
                    L = X.day.values >= 179; c6 = X.in_temp.values < 6
                    line += " %s %+.2f%% P %.3f (p2 %+.2f%%, <6C %s) |" % (v, 100 * rel[v][-1], p, 100 * (r((b - y)[L]) / r((a - y)[L]) - 1),
                                                                         "%.2f->%.2f" % (r((a - y)[c6]), r((b - y)[c6])) if c6.any() else "-")
                print(line)
        for v in VALS:
            m = np.mean(rel[v])
            if v in ("DIAG10", "EXT10"):
                ok &= all(x < 0 for x in rel[v]) and max(pw[v]) < ALPHA
            elif v == "EL1":
                ok &= m < 0 and max(pw[v]) < .05
            else:
                ok &= not (m > 0 and (1 - min(pw[v])) < ALPHA)
            print("  %-6s seed-mean %+.2f%%  P(worse) max %.4f" % (v, 100 * m, max(pw[v])))
        print("VERDICT %s: %s" % (cand, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if len(sys.argv) and os.environ.get("TT9_SUM") != "1":
        run()
    summarize()
