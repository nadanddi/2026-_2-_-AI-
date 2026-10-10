# -*- coding: utf-8 -*-
"""TT18: fresh-layout checks of the pretrain feature (HP) and its combination with P1, reusing the DIAG10H members of TT17
(BASE 606/707, TabPFN 33-40).  Written and committed BEFORE TT17's results were seen (2026-10-10 집 클로드, user away).
Candidates (k = 2, alpha .0125):
  C1 HP-ALL : CODEX residual features + pre49 (49-greenhouse pretrained prediction, re13 recipe, pretrain seeds 202/303)
              on every row - confirmation of TT16 HP-ALL (32/32 cells better there, DIAG10 P .0195).
  C2 P1+HP  : CODEX residual features without *_h0 and act_heating*, plus pre49, on every row.
Judged on DIAG10H (fresh in TT17, scored here for the first time for these candidates) ALL rows, rough days excluded:
  PASS iff all 8 cells (BASE seed 606/707 x PFN family A/B x pretrain seed 202/303) better and farm x 5-day block bootstrap
  P(worse) < .0125 in every cell.  EXT10 (old validator, stored TT1 members; C2 CODEX refit here): reported, direction only.
  Also reported: DIAG10H pass-2 rows, F13/F47.
Run (after TT17 finished):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt18_hp_combo_fresh_v1.py
"""
import os, sys, pathlib
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
import importlib.util
spec = importlib.util.spec_from_file_location("t17", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tt17_p1_second_confirm_v1.py"))
T17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T17)
CK1 = os.path.join(env.LOCAL, "tt1_ckpt"); CK17 = os.path.join(env.LOCAL, "tt17_ckpt"); CK = os.path.join(env.LOCAL, "tt18_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt18_season"
RD = T17.RD; PS = (202, 303); ALPHA = .0125
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    wp = np.asarray(wp, float)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    FC1 = [c for c in FC if not c.endswith("_h0") and not (c == "act_heating" or c.startswith("act_heating_"))]
    import re13_pooled51_v1 as R13
    P = R13.build_panel(); feats = [c for c in P.columns if c not in ("farm", "day", "hour", "row_id", "sub_temp", "farm_cat")]
    oth = P[~P.farm.isin(R13.TARGETS)]; tgt = P[P.farm.isin(R13.TARGETS)].set_index("row_id").reindex(pfn.row_id)
    pre = {sd: R13.fit(R13.PARAMS, oth[feats], oth.sub_temp.values, sd).predict(tgt[feats]) for sd in PS}

    def codex(tr, va, wtr, cols):
        save = W.TM.FEATURE_COLUMNS; W.TM.FEATURE_COLUMNS = cols
        try:
            return W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wtr, 726)
        finally:
            W.TM.FEATURE_COLUMNS = save
    jobs = [("H", k, T17.masks(lab, fd)) for k, fd in enumerate(T17.fresh_folds(lab))]
    jobs += [("EXT10", k, common.split_mask(lab, fd)) for name, folds in sets if name == "EXT10" for k, fd in enumerate(folds)]
    for tag, k, (tm, vm) in jobs:
        path = os.path.join(CK, "%s_%d.csv" % (tag, k))
        if os.path.exists(path) or not vm.sum():
            continue
        out = pd.DataFrame({"row_id": lab.row_id[vm].values})
        for sd in PS:
            pf = pfn.copy(); pf["pre49"] = pre[sd]
            tr, va = W.season_fold(pf[tm], pf[vm], wv, "tt18_%s" % tag, k)
            out["codex_C1_%d" % sd] = codex(tr, va, wp[tm], FC + ["pre49"])
            out["codex_C2_%d" % sd] = codex(tr, va, wp[tm], FC1 + ["pre49"])
        out.to_csv(path, index=False); print("%s/%d done" % (tag, k), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261019)
    H = pd.concat([pd.read_csv(os.path.join(CK17, f)) for f in sorted(os.listdir(CK17)) if f.startswith("H_")], ignore_index=True)
    H = H.merge(pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.startswith("H_")]), on="row_id")
    T1 = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    E = T1[T1.validator == "EXT10"].merge(pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK)) if f.startswith("EXT10_")]), on="row_id")
    E = E.rename(columns={"base_REF_7": "base_R_7", "base_REF_101": "base_R_101", "codex_REF": "codex_R"})
    for cand in ("C1", "C2"):
        print("\n==== %s" % cand); ok = True
        for v, D, seeds, judged in (("DIAG10H", H, (606, 707), True), ("EXT10", E, (7, 101), False)):
            for scope, sel in (("all rows", lambda X: np.ones(len(X), bool)), ("pass-2", lambda X: X.day.values >= 179)):
                X = D[np.array([(f, d) not in RD for f, d in zip(D.farm, D.day)])]; X = X[sel(X)].copy()
                g = np.where(X.in_temp.isna(), 1, np.clip((X.in_temp - 8) / 2, 0, 1)); y = X.sub_temp.values
                rel, pw = [], []
                for ps in PS:
                    for s in seeds:
                        for f in "AB":
                            bl = lambda c: (.4 * X["base_R_%d" % s] + (.2 + .4 * (1 - g)) * X[c] + .4 * g * X["pfn_%s" % f]).values
                            a, b = bl("codex_R"), bl("codex_%s_%d" % (cand, ps)); p = boot(X, a, b, rng)
                            rel.append(r(b - y) / r(a - y) - 1); pw.append(p)
                fr = ""
                o = all(x < 0 for x in rel) and max(pw) < ALPHA
                if judged and scope == "all rows":
                    ok &= o
                print("  %-8s %-8s rows %5d | mean %+.2f%% [%+.2f, %+.2f] all better %s P max %.4f%s" % (
                    v, scope, len(X), 100 * np.mean(rel), 100 * min(rel), 100 * max(rel), all(x < 0 for x in rel), max(pw),
                    " -> " + ("pass" if o else "fail") if judged and scope == "all rows" else ""))
        print("VERDICT %s: %s" % (cand, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT18_SUM") != "1":
        run()
    summarize()
