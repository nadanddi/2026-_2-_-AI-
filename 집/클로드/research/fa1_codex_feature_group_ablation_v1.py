# -*- coding: utf-8 -*-
"""FA1: CODEX feature-group ablation (DIAGNOSTIC, step 1 of feature re-adjustment).  (2026-10-10 집 클로드, user: "피처 재조정도
들어가나?" -> agreed plan: drop one feature group at a time, CODEX only first.)
For each group, remove its columns from the CODEX LGB residual features only (physics Ridge part unchanged), refit CODEX
(seed 726) on DIAG10 / EXT10 / EXT12 folds, and recompute W40G-S with BASE seed 7 + PFN family A (stored TT1 members).
Groups (89 columns): SEASON (day->season), TIME (hour, sin, cos), MISC (second, delta, farm_id), one group per raw
variable (out_temp, out_hum, out_rad, out_wspd, in_temp, in_hum, in_co2, vent, shade, thermal, heating, circfan, co2,
fog: all its transforms), and two transform groups across variables: H0 (all *_h0 = value at hour 0, crosses the
midnight stitching) and DAYSTAT (all *_mean / *_std running-day summaries).
Output: relative RMSE change vs REF per validator (all rows, pass-2 rows, in_temp<6 rows) + DIAG10 block bootstrap
P(worse).  NO adoption from this run: a group is only a CANDIDATE for step 2 if removal improves DIAG10 AND EXT10 and
EXT12 is not worse by > 0.3%; step 2 must re-test with BASE refit, new seeds and a fresh layout under the user rule.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u fa1_codex_feature_group_ablation_v1.py   (sum = summary only)
"""
import os, sys, pathlib
import env  # noqa: F401
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "fa1_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "fa1_season"
VALS = ("DIAG10", "EXT10", "EXT12")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
RAW = {"out_temp": "out_temp", "out_hum": "out_hum", "out_rad": "out_rad", "out_wspd": "out_wspd", "in_temp": "in_temp",
       "in_hum": "in_hum", "in_co2": "in_co2", "vent": "act_vent", "shade": "act_shade", "thermal": "act_thermal",
       "heating": "act_heating", "circfan": "act_circfan", "co2": "act_co2", "fog": "act_fog"}


def groups(FC):
    g = {"SEASON": ["season"], "TIME": ["hour", "sin", "cos"], "MISC": ["second", "delta", "farm_id"],
         "H0": [c for c in FC if c.endswith("_h0")], "DAYSTAT": [c for c in FC if c.endswith("_mean") or c.endswith("_std")]}
    for k, v in RAW.items():
        g[k] = [c for c in FC if c == v or c.startswith(v + "_")]
    return g


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    wp = np.asarray(wp, float)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    GR = groups(FC)
    print({k: len(v) for k, v in GR.items()}, flush=True)
    for name, folds in sets:
        if name not in VALS:
            continue
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "fa1_" + name, k)
            tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
            out = pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values})
            save = W.TM.FEATURE_COLUMNS
            try:
                for gname, cols in GR.items():
                    W.TM.FEATURE_COLUMNS = [c for c in FC if c not in cols]
                    out["codex_" + gname] = W.TM.codex_fit_predict(tr, va, wp[tm], 726)
            finally:
                W.TM.FEATURE_COLUMNS = save
            out.to_csv(path, index=False); print("%s/%d done" % (name, k), flush=True)


def summarize():
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))])
    G = G.merge(C, on=["validator", "row_id"], how="inner")
    gr = [c[6:] for c in C.columns if c.startswith("codex_")]
    rng = np.random.default_rng(20261010)
    bl = lambda X, c: .4 * X.base_REF_7 + (.2 + .4 * (1 - X.g)) * X[c] + .4 * X.g * X.pfn_A
    G["g"] = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    rows = []
    for gname in gr:
        row = {"group": gname}
        for v in VALS:
            X = G[G.validator == v]; y = X.sub_temp.values
            a, b = bl(X, "codex_REF").values, bl(X, "codex_" + gname).values
            row[v] = 100 * (r(b - y) / r(a - y) - 1)
            L = X.day.values >= 179
            row[v + "_p2"] = 100 * (r((b - y)[L]) / r((a - y)[L]) - 1)
            if v == "DIAG10":
                cl = (X.farm + "_" + (X.day // 5).astype(str)).values
                dd = pd.Series((b - y) ** 2 - (a - y) ** 2).groupby(cl).agg(["sum", "count"])
                idx = rng.integers(0, len(dd), (20000, len(dd)))
                row["P_worse"] = float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())
            if v == "EXT10":
                c6 = X.in_temp.values < 6
                row["EXT10_lt6"] = 100 * (r((b - y)[c6]) / r((a - y)[c6]) - 1)
        row["candidate"] = row["DIAG10"] < 0 and row["EXT10"] < 0 and row["EXT12"] < .3
        rows.append(row)
    T = pd.DataFrame(rows).set_index("group").sort_values("DIAG10")
    pd.set_option("display.width", 200)
    print("\n묶음을 뺐을 때 RMSE 변화(%%, 음수 = 빼면 좋아짐), W40G-S 시드 7·PFN A 기준")
    print(T.round(2).to_string())


if __name__ == "__main__":
    if MODE != "sum":
        run()
    summarize()
