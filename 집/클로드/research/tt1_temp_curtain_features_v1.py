# -*- coding: utf-8 -*-
"""TT1: temperature W40G-S + thermal-curtain schedule features.  (2026-10-09 집 클로드, user: "지금부터 개선 모델 만들어보자"
- improved temperature model to submit with submission_15 EC).  Fixed before running.
Base = W40G-S (9th submission temperature):  g = clip((in_temp - 8) / 2, 0, 1),
  W40G-S = 0.4 BASE + (0.2 + 0.4 (1 - g)) CODEX_season + 0.4 g PFN_season
  BASE  = .65 (LinearRegression physics + LGB residual) + .25 ridge + .10 Nystroem on the MASK-world features ct (with `day`),
          seeds 7 / 101 (temp_members);  CODEX_season = Ridge physics + LGB residual on FEATURE_COLUMNS with day -> season
          (TM.codex_fit_predict, seed 726);  PFN_season = stored TK2 TabPFN season predictions, family A = contexts 1-8,
          B = 17-24 (mean).  World, folds, weights, season mapping = Codex TK2 world.py (read-only import).
Candidate CUR = BASE and CODEX each get 5 curtain features (CT1 THS5: th_full_hours_td, th_night_closed, th_h9,
  th_h9_partial, th_sched_score; same farm, same day, hours 0..h only - unaffected by the MASK rule).  PFN unchanged.
Both baseline and candidate members are refitted in this run (same code path).
RULE (temperature rule as VW1, k = 1): PASS iff CUR better than W40G-S in all 12 cells (BASE seed {7,101} x PFN family
{A,B} x {DIAG10, EXT10, EXT12}) AND DIAG10 farm x 5-day block bootstrap (20,000) P(worse) < .025 for every combo.
Reported: EL1, pass-2 rows, cold rows (in_temp <= 8).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt1_temp_curtain_features_v1.py
"""
import os, sys, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ct1", os.path.join(HERE, "ct1_thermal_schedule_features_v1.py"))
ct1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct1)
TKDIR = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1")
sys.path.insert(0, TKDIR)
import world as W  # noqa: E402  (Codex TK2 world, read-only)
import common  # noqa: E402
CK = os.path.join(env.LOCAL, "tt1_ckpt")
THS5 = ["th_full_hours_td", "th_night_closed", "th_h9", "th_h9_partial", "th_sched_score"]
SEEDS = (7, 101)
FAM = {"A": range(1, 9), "B": range(17, 25)}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def main():
    os.makedirs(CK, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    tX, ty, sX = W.TM.ORIG()
    raw = tX[tX.farm.isin(common.TARGET_FARMS)][["row_id", "farm", "day", "hour", "act_thermal"]].copy()
    TF = ct1.thermal_features(raw)
    for D in (lab, pfn):
        for c in THS5:
            D[c] = TF.set_index("row_id").loc[D.row_id, c].to_numpy()
    assert (lab.row_id.values == pfn.row_id.values).all()
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
            out = lab.loc[vm, ["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy().reset_index(drop=True)
            out["validator"], out["fold"] = name, k
            for tag, cts in (("REF", ct), ("CUR", ct + THS5)):
                for s in SEEDS:
                    out["base_%s_%d" % (tag, s)] = W.base_predict(lab[tm], lab[vm], wb[tm], cts, phc, s)
                fcs = FC + (THS5 if tag == "CUR" else [])
                saveFC = W.TM.FEATURE_COLUMNS
                W.TM.FEATURE_COLUMNS = [c if c != "day" else "season" for c in fcs]
                try:
                    out["codex_%s" % tag] = W.TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), wp[tm], 726)
                finally:
                    W.TM.FEATURE_COLUMNS = saveFC
            for fam, ctxs in FAM.items():
                ps = []
                for c in ctxs:
                    zz = np.load(os.path.join(W.OUT, "pfn_%s_%d_%d.npz" % (name, k, c)), allow_pickle=True)
                    ps.append(pd.Series(zz["season"], index=zz["row_id"]))
                out["pfn_%s" % fam] = pd.concat(ps, axis=1).mean(axis=1).reindex(out.row_id).to_numpy()
            assert out.notna().drop(columns=["in_temp"]).all().all()
            out.to_csv(path, index=False)
            print("%s/%d done" % (name, k), flush=True)
    summarize()


def summarize():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    rng = np.random.default_rng(20261009)
    ok, cells = True, 0
    for s in SEEDS:
        for fam in FAM:
            for tag in ("REF", "CUR"):
                G["w_%s_%d_%s" % (tag, s, fam)] = .4 * G["base_%s_%d" % (tag, s)] + (.2 + .4 * (1 - g)) * G["codex_%s" % tag] + .4 * g * G["pfn_%s" % fam]
            line = "seed %3d PFN %s |" % (s, fam)
            for v in ("DIAG10", "EXT10", "EXT12", "EL1"):
                X = G[G.validator == v]
                if X.empty:
                    continue
                a, b = r(X["w_REF_%d_%s" % (s, fam)] - X.sub_temp), r(X["w_CUR_%d_%s" % (s, fam)] - X.sub_temp)
                L = X.day >= 179
                la, lb = r(X["w_REF_%d_%s" % (s, fam)][L] - X.sub_temp[L]), r(X["w_CUR_%d_%s" % (s, fam)][L] - X.sub_temp[L])
                line += " %s %.4f->%.4f (%+.2f%%, pass2 %+.2f%%)" % (v, a, b, 100 * (b / a - 1), 100 * (lb / la - 1))
                if v != "EL1":
                    cells += b < a; ok &= b < a
                if v == "DIAG10":
                    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
                    dd = pd.Series((X["w_CUR_%d_%s" % (s, fam)] - X.sub_temp) ** 2 - (X["w_REF_%d_%s" % (s, fam)] - X.sub_temp) ** 2).groupby(cl).agg(["sum", "count"])
                    sm, n = dd["sum"].values, dd["count"].values
                    idx = rng.integers(0, len(sm), (20000, len(sm)))
                    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()); ok &= p < .025
                    line += " P(worse) %.4f |" % p
            print(line)
    X = G[G.validator == "DIAG10"]; cold = X.in_temp <= 8
    print("DIAG10 cold rows (in_temp<=8): REF %.4f CUR %.4f (seed 7, PFN A)" % (r(X.w_REF_7_A[cold] - X.sub_temp[cold]), r(X.w_CUR_7_A[cold] - X.sub_temp[cold])))
    print("\nVERDICT: cells better %d/12 -> %s" % (cells, "PASS" if ok and cells == 12 else "FAIL"))


if __name__ == "__main__":
    if MODE == "sum":
        summarize()
    else:
        main()
