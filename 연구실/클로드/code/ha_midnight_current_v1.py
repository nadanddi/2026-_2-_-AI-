# -*- coding: utf-8 -*-
"""H-A: after the midnight seam, history features carry the previous (other-source)
day, so hours 0-2 are poorly predicted by every G_C2 member (catalog 6b.13).
Test: blend in a model that uses the CURRENT hour's inputs only, at hours 0-2 only.
(lab Claude, 2026-09-30)

CUR model   LightGBM huber (T_HUB params of rounds 3-6) on current-hour inputs
            only - no lag / EWM / rolling / daily-aggregate / fingerprint columns:
            outside & inside sensors, actuators, their instantaneous physics
            transforms, hour sin/cos, farm id.  Fitted on MASK-world training
            rows (catalog 6.30) with the round-5/6 row weights, in the same folds
            as the saved G_C2 members (DIAG10 round-robin, EXT10, EXT12 whole-day
            hold-out by min 3 h air temperature).
Candidate   hours 0,1,2:  0.5 * G_C2 + 0.5 * CUR ;  other hours: G_C2 unchanged.
            G_C2 = saved members (seed pairs averaged, TabPFN v2 samples 1-8).
            Mixing weight 0.5 and hours 0-2 are fixed here, not tuned.

PRE-SET ADOPTION RULE (fixed before running, one candidate -> no Bonferroni):
  ADOPT-AS-CANDIDATE only if RMSE(candidate) < RMSE(G_C2) for BOTH CUR seeds
  (7, 101) on ALL three validators (DIAG10, EXT10, EXT12), AND the DIAG10
  paired bootstrap P(worse) < 0.025 for both seeds.  Otherwise REJECT.
  Hours-0-2 RMSE is reported for diagnosis only.
  Nothing is submitted; a pass only makes it a candidate for the user.

Output: logs/ha_midnight_current_v1.log, local/ha_midnight_current_v1_oof.npz
Run:  PYTHONPATH="" <python> -u ha_midnight_current_v1.py   (from 연구실/클로드/code)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "집", "클로드", "research"))
import env  # noqa: E402,F401
import numpy as np  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402

import common  # noqa: E402
import harness  # noqa: E402
from common import split_mask, rmse, TARGET_FARMS  # noqa: E402
from anal_q1_errors import diag_folds  # noqa: E402
from screen_v6 import boot  # noqa: E402
from make_submission_v3 import T_HUB  # noqa: E402
import train_flags_v6 as TF  # noqa: E402
from temp_mask_v1 import masked_loader, build_world, ORIG  # noqa: E402

CUR_COLS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2",
            "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog",
            "vpd_in", "vpd_out", "dt_in_out", "rad_eff", "screen_ins", "heat_screen", "transp_pm",
            "root_dh", "heat_input", "hr_sin", "hr_cos", "farm_id"]
HOURS = (0, 1, 2)
MIX = 0.5
SEEDS = (7, 101)
OUT_LOCAL = os.path.join(HERE, "..", "local")


def main():
    labF, _, _ = build_world()                       # FULL world only for the EXT day selection (as temp_mask_v1)
    common.load_raw = masked_loader
    try:
        labM, _, _ = build_world()
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()
    assert (labM.row_id.values == labF.row_id.values).all()
    w = TF.row_weights(labM, 0.2, w_noisy=0.2)
    y = labM.sub_temp.values
    t = labM.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    night = labM.hour.isin(HOURS).values
    z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (z["row_id"] == labM.row_id.values).all()

    dmin = labF.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(labM))]
    for th in (10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    X = labM[CUR_COLS].astype(float)
    save, passes = {}, []
    for s, fds in sets:
        base = np.mean([z["%s__MASK__7" % s], z["%s__MASK__101" % s]], axis=0)
        cx = np.mean([z["%s__CODEX__726" % s], z["%s__CODEX__727" % s]], axis=0)
        pfn = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % s).mean(0)
        gc2 = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
        for sd in SEEDS:
            cur = np.full(len(labM), np.nan)
            for fd in fds:
                trm, vam = split_mask(labM, fd)
                m = LGBMRegressor(**T_HUB, random_state=sd, deterministic=True, force_col_wise=True,
                                  n_jobs=4, verbose=-1)
                m.fit(X[trm], y[trm], sample_weight=w[trm])
                cur[vam] = m.predict(X[vam])
            cand = np.where(night, (1 - MIX) * gc2 + MIX * cur, gc2)
            ok = ~np.isnan(gc2) & ~np.isnan(cur)
            r0, r1 = rmse(gc2[ok], y[ok]), rmse(cand[ok], y[ok])
            nn = ok & night
            _, lo, hi, pw = boot(labM[ok].reset_index(drop=True), "sub_temp", gc2[ok], cand[ok])
            print("%-6s CUR seed %3d | G_C2 %.5f -> cand %.5f (%+.2f%%) CI [%+.4f, %+.4f] P(worse)=%.3f | "
                  "hours 0-2: G_C2 %.3f CUR %.3f cand %.3f | CUR alone all-hours %.3f"
                  % (s, sd, r0, r1, 100 * (r1 / r0 - 1), lo, hi, pw, rmse(gc2[nn], y[nn]), rmse(cur[nn], y[nn]),
                     rmse(cand[nn], y[nn]), rmse(cur[ok], y[ok])), flush=True)
            passes.append((s, sd, r1 < r0, pw))
            save["%s__CUR__%d" % (s, sd)] = cur
    os.makedirs(OUT_LOCAL, exist_ok=True)
    np.savez(os.path.join(OUT_LOCAL, "ha_midnight_current_v1_oof.npz"), row_id=labM.row_id.values, **save)

    all_better = all(p[2] for p in passes)
    diag_ok = all(p[3] < 0.025 for p in passes if p[0] == "DIAG10")
    print("\nall 6 cells improve: %s | DIAG10 P(worse) < 0.025 for both seeds: %s" % (all_better, diag_ok))
    print("H-A verdict: %s" % ("ADOPT-AS-CANDIDATE" if all_better and diag_ok else "REJECT"))


if __name__ == "__main__":
    main()
