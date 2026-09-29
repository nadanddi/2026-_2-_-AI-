# -*- coding: utf-8 -*-
"""Round-6 temperature candidates from inspector 4-3, in the MASK world
(training-row features computed without test inputs, temp_mask_v1.py).

Candidates
  MID    midnight carry-over features: the model drags the previous record
         day's (usually another source's) evening into the new day.
           g1 = mean in_temp(d-1, 20-23 h) - in_temp(d, 0 h)
           g2 = mean in_temp(d-2, 20-23 h) - in_temp(d, 0 h)   (d-2 = same source more often)
           g1*exp(-h/6), g2*exp(-h/6)
         Inputs of earlier days and hour 0 of the same day only.
  NIGHT  noisy-day score recomputed on night hours (19-06 h) WITHOUT CO2
         dosing (act_co2 == 0): dosing on sealed days makes a saw-tooth that
         the all-hours score mistook for noise (39 of 96 days).  Same size
         (top quarter, cold-exempt) so only the definition changes.
  BOTH   MID + NIGHT.

Pre-set adoption rule (fixed before running):
  improve DIAG10 AND EXT10 for BOTH seeds (7, 101) vs the MASK base of the
  same seed, and the DIAG10 block-bootstrap CI excludes 0.
  Secondary (reported, not deciding): EXT12, EXT8, and the effect after the
  fixed 0.8/0.2 Codex blend.

Run:  cd research && PYTHONPATH="" <python> -u temp_v7_candidates.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

import common
import harness
import cold_v5
from common import rmse, TARGET_FARMS
from screen_v6 import temp_members, collect, boot
from anal_q1_errors import diag_folds
import train_flags_v6 as TF
from temp_mask_v1 import masked_loader, build_world, ORIG

SEEDS = (7, 101)
MID = ["mid_g1", "mid_g2", "mid_g1d", "mid_g2d"]


def midnight_features():
    tX, _, sX = masked_loader()
    a = pd.concat([tX, sX])[["row_id", "farm", "day", "hour", "in_temp"]]
    a = a[a.farm.isin(TARGET_FARMS)]
    eve = a[a.hour.between(20, 23)].groupby(["farm", "day"]).in_temp.mean()
    h0 = a[a.hour == 0].set_index(["farm", "day"]).in_temp
    out = a[["row_id", "farm", "day", "hour"]].copy()
    k0 = list(zip(out.farm, out.day))
    k1 = list(zip(out.farm, out.day - 1))
    k2 = list(zip(out.farm, out.day - 2))
    h0v = h0.reindex(k0).values
    out["mid_g1"] = eve.reindex(k1).values - h0v
    out["mid_g2"] = eve.reindex(k2).values - h0v
    dec = np.exp(-out.hour.values / 6.0)
    out["mid_g1d"] = out.mid_g1 * dec
    out["mid_g2d"] = out.mid_g2 * dec
    return out.set_index("row_id")[MID]


def night_noisy_days():
    tX, _, _ = ORIG()
    a = tX[tX.farm.isin(TARGET_FARMS)].sort_values(["farm", "t"]).reset_index(drop=True)
    a["ewm3"] = a.groupby("farm").in_temp.transform(lambda s: s.ewm(halflife=3, ignore_na=True).mean())
    rows = []
    for (f, d), g in a.groupby(["farm", "day"]):
        g = g.sort_values("t")
        ok = ((g.hour >= 19) | (g.hour <= 6)) & (g.act_co2.fillna(0) == 0)
        d1 = g.in_co2.diff().where((g.t.diff() == 1) & ok & ok.shift(1, fill_value=False))
        d1 = d1.dropna()
        ac1 = d1.autocorr(lag=1) if len(d1) >= 6 else np.nan
        d2 = d1.diff().abs().median() if len(d1) >= 3 else np.nan
        rows.append(dict(farm=f, day=d, ac1=ac1, d2=d2, cold=bool((g.ewm3 < 8).any())))
    D = pd.DataFrame(rows)
    D["noise_score"] = (-D.ac1).rank(pct=True) + D.d2.rank(pct=True)
    thr = D.noise_score.quantile(0.75)
    D["noisy"] = (D.noise_score >= thr) & ~D.cold
    return D


def weights(lab, noisy_set):
    w = TF.row_weights(lab, 0.2)
    m = np.array([(f, d) in noisy_set for f, d in zip(lab.farm.values, lab.day.values)])
    w[m] = w[m] * 0.2
    return w


def main():
    common.load_raw = masked_loader
    try:
        lab, ct, phc = build_world()
    finally:
        common.load_raw = ORIG
        harness._CACHE.clear()
    mf = midnight_features()
    for c in MID:
        lab[c] = mf.loc[lab.row_id, c].values
    y = lab.sub_temp.values

    old = TF.noisy_days()
    old_set = set(map(tuple, old[old.noisy][["farm", "day"]].values))
    new = night_noisy_days()
    new_set = set(map(tuple, new[new.noisy][["farm", "day"]].values))
    print("noisy days: old %d, night-rule %d, overlap %d | night ac1 median noisy %.2f vs rest %.2f"
          % (len(old_set), len(new_set), len(old_set & new_set),
             new[new.noisy].ac1.median(), new[~new.noisy].ac1.median()), flush=True)
    w_old, w_new = weights(lab, old_set), weights(lab, new_set)
    assert np.allclose(w_old, TF.row_weights(lab, 0.2, w_noisy=0.2))

    V = {"BASE": (ct, w_old), "MID": (ct + MID, w_old), "NIGHT": (ct, w_new), "BOTH": (ct + MID, w_new)}
    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(lab))]
    for th in (8.0, 10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    zc = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    assert (zc["row_id"] == lab.row_id.values).all()
    res = {}
    for s, fds in sets:
        for vn, (cols, w) in V.items():
            for sd in SEEDS:
                key = "%s__MASK__%d" % (s, sd)
                if vn == "BASE" and key in zc.files:
                    res[(s, vn, sd)] = zc[key]
                    continue
                cold_v5.SEED = sd
                M = collect(lab, fds, lambda tr, va, m: temp_members(tr, va, cols, phc, w[m]))
                res[(s, vn, sd)] = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
        cold_v5.SEED = 7
        print("  %s done" % s, flush=True)
    np.savez(env.LOCAL + "/temp_v7_candidates_oof.npz", row_id=lab.row_id.values,
             **{"%s__%s__%d" % k: v for k, v in res.items()})

    codex = {s: zc["%s__CODEX__726" % s] for s, _ in sets if "%s__CODEX__726" % s in zc.files}
    print("\n== candidate vs BASE (same seed) ==")
    verdict = {}
    for vn in ("MID", "NIGHT", "BOTH"):
        ok = True
        for s, _ in sets:
            for sd in SEEDS:
                a, b = res[(s, "BASE", sd)], res[(s, vn, sd)]
                g = ~np.isnan(a)
                pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", a[g], b[g])
                ra, rb = rmse(a[g], y[g]), rmse(b[g], y[g])
                line = "  %-5s %-6s seed %3d  base %.5f  cand %.5f  %+.2f%% [%+.4f, %+.4f]" % (
                    vn, s, sd, ra, rb, 100 * (rb / ra - 1), lo, hi)
                if s in codex:
                    ba, bb = 0.8 * a + 0.2 * codex[s], 0.8 * b + 0.2 * codex[s]
                    line += " | with Codex blend %.5f -> %.5f (%+.2f%%)" % (
                        rmse(ba[g], y[g]), rmse(bb[g], y[g]), 100 * (rmse(bb[g], y[g]) / rmse(ba[g], y[g]) - 1))
                print(line)
                if s in ("DIAG10", "EXT10") and rb >= ra:
                    ok = False
                if s == "DIAG10" and hi >= 0:
                    ok = False
        verdict[vn] = ok
    print("\nPRE-SET RULE VERDICT:", {k: ("ADOPT" if v else "REJECT") for k, v in verdict.items()})


if __name__ == "__main__":
    main()
