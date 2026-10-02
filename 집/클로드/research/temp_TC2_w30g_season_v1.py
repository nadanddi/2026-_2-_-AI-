# -*- coding: utf-8 -*-
"""Temperature TC2a (fixed before running; 2026-10-03 집 클로드).
User: round 8 should test a NEW temperature model together with EC season v2,
not resubmit the known W30G.  TC1 (C6.167): season index in the MASK / Codex
members -> late -2.8~-3.0% (rule just missed).  Here the whole W30G blend:
  g = clip((in_temp - 8)/2, 0, 1)
  W30G = (0.5 - 0.1(1-g)) MASK + (0.2 + 0.4(1-g)) CODEX + 0.3 g PFN
variant BASE: MASK and CODEX with `day`; SEAS: both with the DC4 season index
instead of `day`.  PFN = stored temperature TabPFN OOF (web_tabpfn_v2_temp_*,
sample mean), identical in both variants (TabPFN with season = Codex TK2).
MASK seeds 7 and 101 (Codex seeds 726/727 are identical, 6b.27 -> 726).
Rule (user's): SEAS better than BASE for both seeds x DIAG10 / EXT10 / EXT12 and
DIAG10 paired block bootstrap (farm x 5-day, 20,000) P(worse) < .025 per seed.
Late (>= 179) and late & calendar < 70 reported.  Even if it fails the rule,
the user may submit it as an informative round-8 temperature (user's call).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u temp_TC2_w30g_season_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

import common
import harness
import cold_v5
import temp_mask_v1 as TM
from common import split_mask, TARGET_FARMS
from screen_v6 import temp_members
from anal_q1_errors import diag_folds
import train_flags_v6 as TF

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc4", os.path.join(HERE, "ec2_DC4_exact_twin_anchor_v1.py"))
dc4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc4)
SEEDS_T = (7, 101)


def main():
    _, labT, _ = harness.load()                     # TabPFN OOF arrays are aligned to this frame
    pfn_rows = labT.row_id.values
    common.load_raw = TM.masked_loader
    try:
        labM, ct, phc = TM.build_world()
    finally:
        common.load_raw = TM.ORIG
        harness._CACHE.clear()
    tX, ty, sX = TM.ORIG()
    CF = TM.build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    for c in TM.FEATURE_COLUMNS:
        if c not in labM.columns:
            labM[c] = CF.loc[labM.row_id, c].values
    w = TF.row_weights(labM, 0.2, w_noisy=0.2)
    wv = dc4.weather_vectors(tX[tX.row_id.str[:3].isin(["F13", "F47"])])
    calm = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv")).set_index(["farm", "day"]).cal
    labF, _, _ = TM.build_world()
    dmin = labF.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(labM))]
    for th in (10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    ctS = [c if c != "day" else "season" for c in ct]
    cxB = [c for c in TM.FEATURE_COLUMNS if c in labM.columns]
    out = []
    for s, fds in sets:
        pfn = pd.Series(np.load(os.path.join(env.LOCAL, "web_tabpfn_v2_temp_%s.npy" % s)).mean(0), index=pfn_rows)
        for k, fd in enumerate(fds):
            trm, vam = split_mask(labM, fd)
            if not vam.sum():
                continue
            tr, va = labM[trm].copy(), labM[vam].copy()
            tdays = tr[["farm", "day"]].drop_duplicates()
            vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
            season, vq = dc4.season_index(tdays, vdays, wv)
            tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
            vmap = dict(zip(zip(vdays.farm, vdays.day), vq))
            va["season"] = [vmap[(f, d)] for f, d in zip(va.farm, va.day)]
            vr = va.reset_index(drop=True)
            frame = vr[["row_id", "farm", "day", "hour", "sub_temp", "in_temp"]].copy()
            frame["validator"], frame["fold"] = s, k
            frame["pfn"] = pfn.reindex(vr.row_id).values
            frame["codex_base"] = TM.codex_fit_predict(tr, vr, w[trm], 726)
            frame["codex_seas"] = TM.codex_fit_predict(tr.assign(day=tr["season"]), vr.assign(day=vr["season"]), w[trm], 726)
            for sd in SEEDS_T:
                cold_v5.SEED = sd
                for tag, cols in (("base", ct), ("seas", ctS)):
                    M = temp_members(tr, vr, cols, phc, w[trm])
                    frame["mask_%s_%d" % (tag, sd)] = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
            out.append(frame)
            print("%s fold %d done" % (s, k), flush=True)
    cold_v5.SEED = 7
    O = pd.concat(out, ignore_index=True)
    O["cal"] = [calm.get((f, d), np.nan) for f, d in zip(O.farm, O.day)]
    g = np.where(O.in_temp.isna(), 1, np.clip((O.in_temp - 8) / 2, 0, 1))
    for tag in ("base", "seas"):
        for sd in SEEDS_T:
            O["w30_%s_%d" % (tag, sd)] = (0.5 - 0.1 * (1 - g)) * O["mask_%s_%d" % (tag, sd)] + \
                (0.2 + 0.4 * (1 - g)) * O["codex_" + tag] + 0.3 * g * O.pfn
    O.to_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv"), index=False)
    print("PFN coverage (non-NaN) by validator:", O.groupby("validator").pfn.apply(lambda x: round(x.notna().mean(), 3)).to_dict())
    O = O[O.pfn.notna()]
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261003)
    allb = True
    print("\nW30G form: BASE -> SEAS")
    for v in ("DIAG10", "EXT10", "EXT12"):
        G = O[O.validator == v]
        cells = []
        for sd in SEEDS_T:
            a, b = r(G["w30_base_%d" % sd] - G.sub_temp), r(G["w30_seas_%d" % sd] - G.sub_temp)
            allb &= b < a
            L = G.day >= 179
            cells.append("s%d %.4f->%.4f (%+.2f%%) late %.4f->%.4f" % (sd, a, b, 100 * (b / a - 1),
                         r((G["w30_base_%d" % sd] - G.sub_temp)[L]), r((G["w30_seas_%d" % sd] - G.sub_temp)[L])))
        print("  %-6s %s" % (v, " | ".join(cells)))
    D = O[O.validator == "DIAG10"].copy()
    D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for sd in SEEDS_T:
        dd = (D["w30_seas_%d" % sd] - D.sub_temp) ** 2 - (D["w30_base_%d" % sd] - D.sub_temp) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    LE = (D.day >= 179) & (D.cal < 70)
    print("  DIAG10 late&cal<70: %.4f -> %.4f" % (r((D.w30_base_7 - D.sub_temp)[LE]), r((D.w30_seas_7 - D.sub_temp)[LE])))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nTC2a decision:", "PASS" if allb and all(p < 0.025 for p in ps) else "FAIL")


if __name__ == "__main__":
    main()
