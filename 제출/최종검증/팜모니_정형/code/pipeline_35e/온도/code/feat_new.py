# -*- coding: utf-8 -*-
"""New causal derived variables named in HANDOFF section 6.3, built and tested.

Usage:  PYTHONPATH="" python feat_new.py temp|ec [blocks|combo] [fast|full]

`fast` uses the cheap sub_temp screening surrogate of feat_lib (250 trees, one
seed); only deltas against its own baseline are quoted and the winners are
re-measured at the submitted setting by feat_confirm.py.

Every column here uses only the same greenhouse at the current or an earlier
interval (shift / rolling window that closes on t / cumulative-from-start), so
it satisfies the competition's section-5 causality rule by construction.

Blocks
------
hinge   day_hinge{k} = max(day - k, 0) for k on a 30-day grid.
        `day` is already the single strongest EC feature (r = +0.44) but a
        tree can only cut it into steps; a hinge basis lets a linear-ish
        response to crop stage be represented with far fewer splits, and lets
        the slope change at transplanting / first-harvest style breakpoints.
rep24   x_rep24 = 1 when x(t) equals x(t-24) to the sensor's resolution.
        HANDOFF 1.5: train_X contains statistically *restored* values while
        test_X does not, and one restoration signature is a value copied from
        the previous day.  This flags the rows where the X-y relation is
        artificial, which is exactly what Huber loss is currently absorbing
        implicitly.
event   act_*_hsince_on  = hours since that actuator was last active
        act_*_hsince_chg = hours since its value last changed
        A slab responds to the *time since* a heating/venting event, not only
        to its current level; and a long unchanged actuator is either a fixed
        set-point regime or a frozen sensor.
dew     dew_in / dew_out (Magnus) and the dew-point depression.
        Condensation on the slab surface and the driving force for evaporation
        from the substrate are set by the depression, not by RH alone.
duty    act_*_duty48 / act_*_duty168 = fraction of the last 48 h / 168 h in
        which the actuator was active.  This is the management *regime* on the
        multi-day timescale that sets irrigation frequency, and therefore the
        salt balance behind sub_ec.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

import env  # noqa: F401  MUST be first
import common
import features_v2 as F2
from common import USABLE, TARGET_FARMS
from harness import load, score, views
import feat_lib as L

HINGES = [30, 60, 90, 120, 150, 180, 210, 240]
ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan",
        "act_co2", "act_fog"]
REP_COLS = ["in_temp", "in_hum", "in_co2", "out_temp", "out_rad", "out_hum"]
# "active" thresholds.  The curtains are OPENNESS (HANDOFF 1.6), so for
# act_shade / act_thermal "active" means DEPLOYED, i.e. openness below 50.
ACT_ON = {"act_vent": 5.0, "act_heating": 5.0, "act_circfan": 5.0,
          "act_co2": 5.0, "act_fog": 5.0}
ACT_DEPLOYED = {"act_shade": 50.0, "act_thermal": 50.0}


def _hours_since(flag):
    """Hours since `flag` was last True, counting the current interval as 0."""
    n = len(flag)
    idx = np.where(np.asarray(flag, bool), np.arange(n, dtype=float), np.nan)
    last = pd.Series(idx).ffill().values
    return np.arange(n, dtype=float) - last          # NaN before the first event


def _magnus_dew(temp, rh):
    rh = np.clip(np.asarray(rh, float), 1.0, 100.0)
    t = np.asarray(temp, float)
    g = 17.27 * t / (t + 237.3) + np.log(rh / 100.0)
    return 237.3 * g / (17.27 - g)


def build_extra():
    """Per-farm causal panel of the new columns, keyed by row_id."""
    tX, ty, sX = common.load_raw()
    cols = ["row_id", "farm", "day", "hour", "t"] + USABLE
    allx = pd.concat([tX[cols], sX[cols]], ignore_index=True)
    allx = allx[allx.farm.isin(TARGET_FARMS)]

    out, blocks = [], {}
    for farm, g in allx.groupby("farm"):
        g = g.sort_values("t")
        full = np.arange(g.t.min(), g.t.max() + 1)
        p = g.set_index("t").reindex(full)
        p["day"] = p.index // 24
        f = {}

        # -- hinge basis on crop stage ---------------------------------------
        for k in HINGES:
            f["day_hinge%d" % k] = (p.day - k).clip(lower=0).astype(float)

        # -- repeat-of-24h-ago flag (restoration / stuck-sensor signature) ----
        for c in REP_COLS:
            s = p[c]
            tol = 1e-6 + 0.005 * float(np.nanstd(s.values))
            f["%s_rep24" % c] = ((s - s.shift(24)).abs() <= tol).astype(float)
            f["%s_rep24" % c][s.isna().values | s.shift(24).isna().values] = np.nan

        # -- actuator event clocks + multi-day duty cycles --------------------
        for c in ACTS:
            s = p[c]
            if c in ACT_ON:
                on = (s.fillna(0) > ACT_ON[c])
            else:
                on = (s.fillna(100) < ACT_DEPLOYED[c])
            f["%s_hsince_on" % c] = _hours_since(on.values)
            chg = (s.diff().abs() > 1e-9).fillna(False)
            f["%s_hsince_chg" % c] = _hours_since(chg.values)
            onf = on.astype(float)
            f["%s_duty48" % c] = onf.rolling(48, min_periods=12).mean().values
            f["%s_duty168" % c] = onf.rolling(168, min_periods=42).mean().values

        # -- dew point --------------------------------------------------------
        f["dew_in"] = _magnus_dew(p.in_temp, p.in_hum)
        f["dew_out"] = _magnus_dew(p.out_temp, p.out_hum)
        f["dew_dep_in"] = p.in_temp.values - f["dew_in"]
        f["dew_dep_out"] = p.out_temp.values - f["dew_out"]
        f["dew_gap"] = f["dew_in"] - f["dew_out"]

        d = pd.DataFrame(f, index=p.index)
        d["row_id"] = p.row_id
        out.append(d.dropna(subset=["row_id"]))

    ex = pd.concat(out, ignore_index=True)
    names = [c for c in ex.columns if c != "row_id"]
    blocks = {
        "hinge": [c for c in names if c.startswith("day_hinge")],
        "rep24": [c for c in names if c.endswith("_rep24")],
        "event": [c for c in names if "_hsince" in c],
        "dew": [c for c in names if c.startswith("dew")],
        "duty": [c for c in names if "_duty" in c],
    }
    return ex, blocks


def attach():
    panel, lab_t, lab_e = load()
    ex, blocks = build_extra()
    new = [c for c in ex.columns if c != "row_id"]
    lab_t = lab_t.merge(ex, on="row_id", how="left")
    lab_e = lab_e.merge(ex, on="row_id", how="left")
    return panel, lab_t, lab_e, blocks, new


def main(target, mode="blocks", speed="full"):
    panel, lab_t, lab_e, blocks, new = attach()
    v = views(panel)
    seeds = (7,) if speed == "fast" else (7, 101, 2024)
    if target == "temp":
        lab, tgt, cols = lab_t, "sub_temp", v["temp"]
        model = L.TEMP_FAST if speed == "fast" else L.TEMP_MODEL
    else:
        lab, tgt, cols, model = lab_e, "sub_ec", v["ec"], L.EC_MODEL
    print("model=%s seeds=%s" % (speed, seeds))

    print("new columns: %d" % len(new))
    for b, cs in blocks.items():
        print("  %-6s %2d  %s" % (b, len(cs), ", ".join(cs)))

    base = {}
    for kind in ("A", "B"):
        (r, sd, per), oof = score(lab, tgt, cols, model, kind=kind,
                                  seeds=seeds, return_oof=True)
        base[kind] = dict(rmse=r, std=sd, per=per, oof=oof)
        line = "BASE %s %.4f (foldstd %.4f)" % (kind, r, sd)
        if target == "ec":
            line += "  dayRMSE %.4f" % L.day_rmse(lab, tgt, oof)
        print(line, flush=True)

    if mode == "blocks":
        trials = [(b, cols + cs) for b, cs in blocks.items()]
        trials.append(("ALL_NEW", cols + new))
    else:                                            # combo: read winners
        with open(env.LOCAL + "/new_%s.json" % target) as fh:
            prev = json.load(fh)
        good = [r["name"] for r in prev["trials"]
                if r["name"] != "ALL_NEW" and r["d_A"] < 0 and r["d_B"] < 0]
        print("combo from blocks improving on BOTH fold sets:", good)
        add = sum([blocks[b] for b in good], [])
        trials = [("COMBO(%s)" % "+".join(good) if good else "COMBO(empty)",
                   cols + add)]

    rows = []
    for name, cs in trials:
        rec = dict(name=name, n=len(cs))
        t0 = time.time()
        for kind in ("A", "B"):
            (r, sd, per), oof = score(lab, tgt, cs, model, kind=kind,
                                      seeds=seeds, return_oof=True)
            d, lo, hi, pw = L.paired_block_boot(lab, tgt, base[kind]["oof"], oof)
            rec["rmse_" + kind], rec["std_" + kind] = r, sd
            rec["d_" + kind] = r - base[kind]["rmse"]
            rec["ci_" + kind] = [lo, hi]
            pf = [x - y for x, y in zip(per, base[kind]["per"])]
            rec["perfold_" + kind] = pf
            rec["nneg_" + kind] = int(sum(1 for x in pf if x < 0))
            if target == "ec":
                dd, dlo, dhi, _ = L.paired_block_boot(
                    lab, tgt, base[kind]["oof"], oof, level="day")
                rec["dday_" + kind] = dd
                rec["ciday_" + kind] = [dlo, dhi]
        rec["sec"] = time.time() - t0
        rows.append(rec)
        msg = ("add %-14s n=%3d | A %.4f (%+.4f CI %+.4f..%+.4f, %d/5) "
               "| B %.4f (%+.4f CI %+.4f..%+.4f, %d/5)"
               % (name, rec["n"], rec["rmse_A"], rec["d_A"], rec["ci_A"][0],
                  rec["ci_A"][1], rec["nneg_A"], rec["rmse_B"], rec["d_B"],
                  rec["ci_B"][0], rec["ci_B"][1], rec["nneg_B"]))
        if target == "ec":
            msg += " | dayA %+.4f dayB %+.4f" % (rec["dday_A"], rec["dday_B"])
        print(msg, flush=True)

    tag = ("new_%s" % target if mode == "blocks" else "newcombo_%s" % target)
    if speed == "fast":
        tag += "_fast"
    path = env.LOCAL + "/%s.json" % tag
    with open(path, "w") as fh:
        json.dump(dict(target=tgt, base={k: base[k]["rmse"] for k in base},
                       trials=rows), fh, indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    a = sys.argv[1:] + ["ec", "blocks", "full"]
    main(a[0], a[1], a[2])
