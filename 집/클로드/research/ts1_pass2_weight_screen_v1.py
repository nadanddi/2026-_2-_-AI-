# -*- coding: utf-8 -*-
"""TS1 (SCREEN, not adoption): CODEX trained with pass-2 training days up-weighted (2026-10-10 집 클로드, user: "그동안 다른 것도
같이 해봐").  Motivation: P1 changes behaved differently on pass-2 rows (TT12/13/17); all 60 evaluation days are pass 2.
Reuses the three already-consumed fresh layouts (members stored): F (TT12), G (TT13), H (TT17) - so this is a SCREEN; a
pass promotes the candidate to a confirmation on a NEW layout.  Fixed before running.
Candidates (k = 2):
  PW  : CODEX training weights x3 on rows with day >= 179 (all other weights unchanged), all CODEX features.
  PW1 : PW + P1 features (no *_h0, no act_heating*).
Score: pass-2 rows, rough days excluded, blend W40G-S with each layout's stored BASE (2 seeds) and PFN (families A/B).
PROMOTE iff on ALL three layouts every cell (2 BASE seeds x 2 families) is better and the pooled (3 layouts) farm x 5-day
  block bootstrap P(worse) < .0125.  All-row results reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ts1_pass2_weight_screen_v1.py
"""
import os, sys, pathlib, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


T12 = load("t12", "tt12_confirm_noh0_noheat_fresh_v1.py"); T13 = load("t13", "tt13_pass2_rule_confirm_v1.py"); T17 = load("t17", "tt17_p1_second_confirm_v1.py")
W.OUT = pathlib.Path(env.LOCAL) / "ts1_season"
CK = os.path.join(env.LOCAL, "ts1_ckpt")
RD = T13.RD; ALPHA = .0125
LAYOUTS = {"F": (T12.fresh_folds, T12.masks, os.path.join(env.LOCAL, "tt12_ckpt"), "fresh_", (202, 303), "base_%d", "codex_R"),
           "G": (T13.fresh_folds, T13.masks, os.path.join(env.LOCAL, "tt13_ckpt"), "G_", (404, 505), "base_R_%d", "codex_R"),
           "H": (T17.fresh_folds, T17.masks, os.path.join(env.LOCAL, "tt17_ckpt"), "H_", (606, 707), "base_R_%d", "codex_R")}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    wp = np.asarray(wp, float); wpw = wp * np.where(lab.day.values >= 179, 3.0, 1.0)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    FC1 = [c for c in FC if not c.endswith("_h0") and not (c == "act_heating" or c.startswith("act_heating_"))]

    def codex(tr, va, wtr, cols):
        save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = cols
        try:
            return W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wtr, 726)
        finally:
            W.TM.FEATURE_COLUMNS = save
    for L, (ff, mk, _, _, _, _, _) in LAYOUTS.items():
        for k, fd in enumerate(ff(lab)):
            path = os.path.join(CK, "%s_%d.csv" % (L, k))
            if os.path.exists(path):
                continue
            tm, vm = mk(lab, fd)
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "ts1_%s" % L, k)
            pd.DataFrame({"row_id": lab.row_id[vm].values, "codex_PW": codex(tr, va, wpw[tm], FC), "codex_PW1": codex(tr, va, wpw[tm], FC1)}).to_csv(path, index=False)
            print("%s/%d done" % (L, k), flush=True)


def boot_pooled(parts, rng):
    sm, n = [], []
    for X, a, b in parts:
        cl = (X.farm + "_" + (X.day // 5).astype(str)).values
        dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
        sm.append(dd["sum"].values); n.append(dd["count"].values)
    sm, n = np.concatenate(sm), np.concatenate(n)
    idx = rng.integers(0, len(sm), (20000, len(sm)))
    return float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261021)
    for cand in ("PW", "PW1"):
        print("\n==== %s" % cand); allbetter = True; parts = []
        for L, (_, _, ck, pre, seeds, bcol, rcol) in LAYOUTS.items():
            M = pd.concat([pd.read_csv(os.path.join(ck, f)) for f in sorted(os.listdir(ck)) if f.startswith(pre)], ignore_index=True)
            M = M.merge(pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.startswith(L + "_")]), on="row_id")
            M = M[[(f, d) not in RD for f, d in zip(M.farm, M.day)]]
            for scope, sel in (("pass-2", M.day.values >= 179), ("all", np.ones(len(M), bool))):
                X = M[sel].copy(); g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1)); y = X.sub_temp.values
                rel = []
                for s in seeds:
                    for f in "AB":
                        bl = lambda c: (.4 * X[bcol % s] + (.2 + .4 * (1 - g)) * X[c] + .4 * g * X["pfn_%s" % f]).values
                        a, b = bl(rcol), bl("codex_%s" % cand); rel.append(r(b - y) / r(a - y) - 1)
                        if scope == "pass-2" and s == seeds[0] and f == "A":
                            parts.append((X, a, b))
                if scope == "pass-2":
                    allbetter &= all(x < 0 for x in rel)
                print("  layout %s %-6s %+.2f%% [%+.2f, %+.2f]" % (L, scope, 100 * np.mean(rel), 100 * min(rel), 100 * max(rel)))
        p = boot_pooled(parts, rng)
        print("  pooled pass-2 P(worse) %.4f | all layouts all cells better (pass-2): %s -> %s" % (p, allbetter, "PROMOTE to confirmation" if allbetter and p < ALPHA else "stop"))


if __name__ == "__main__":
    if os.environ.get("TS1_SUM") != "1":
        run()
    summarize()
