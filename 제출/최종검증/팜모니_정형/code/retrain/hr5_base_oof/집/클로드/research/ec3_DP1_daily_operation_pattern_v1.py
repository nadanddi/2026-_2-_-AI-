# -*- coding: utf-8 -*-
"""EC stage-3 DP1: within-day operation-sequence features (fixed before running;
2026-10-04 집 클로드).
Basis: high EC = cold-season sealed days (HC0 6.240); R3S sees hourly values, hour-0
values and expanding means/zero-shares, but not the ORDER of operations within the day.
Features at hour h (same record, hours 0..h only -> legal):
  seal_run   consecutive hours up to h with act_vent == 0
  vent_open_hours  hours with act_vent > 0 so far
  first_open_hour  first hour with act_vent > 0 today (24 if none yet)
  thermal_switches / shade_switches  number of state changes (0 <-> >0) so far
  since_curtain_change  hours since the last thermal or shade state change (h+1 if none)
  heat_run   consecutive hours up to h with act_heating > 0
  co2_hours  hours with act_co2 > 0 so far
  vent_max   max act_vent so far
Added to FS and BS of the DC5 R3 recipe (seeds 7/101/2024).
Judge (EC rule 2026-10-04, 6.247): every seed x DIAG10/A/B better, DIAG10 P(worse) <
.025; guard: EL1 or DIAG10 pass-2 rows worse by >= 2 % -> HOLD; EXT reported.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_DP1_daily_operation_pattern_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("db1", os.path.join(HERE, "ec3_DB1_dong_split_models_v1.py"))
db1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(db1)
dc5, dc4, p3, core = db1.dc5, db1.dc4, db1.p3, db1.core
SEEDS = db1.SEEDS
db1.CK = CK = os.path.join(env.LOCAL, "dp1_ckpt")
TAG = "dp"
NEW = ["seal_run", "vent_open_hours", "first_open_hour", "thermal_switches", "shade_switches", "since_curtain_change", "heat_run", "co2_hours", "vent_max"]


def run_len(s):
    out, c = [], 0
    for v in s:
        c = c + 1 if v else 0
        out.append(c)
    return out


def day_feats(g):
    g = g.sort_values("hour")
    v, th, sh, he, co = g.act_vent.fillna(0).values, g.act_thermal.fillna(0).values, g.act_shade.fillna(0).values, g.act_heating.fillna(0).values, g.act_co2.fillna(0).values
    h = g.hour.values
    f = pd.DataFrame(index=g.index)
    f["seal_run"] = run_len(v == 0)
    f["vent_open_hours"] = np.cumsum(v > 0)
    fo = np.where(np.cumsum(v > 0) > 0, np.minimum.accumulate(np.where(v > 0, h, 99)), 24)
    f["first_open_hour"] = fo
    ts = np.r_[0, np.abs(np.diff((th > 0).astype(int)))]; ss = np.r_[0, np.abs(np.diff((sh > 0).astype(int)))]
    f["thermal_switches"], f["shade_switches"] = np.cumsum(ts), np.cumsum(ss)
    last = -1; sc = []
    for k in range(len(h)):
        if ts[k] or ss[k]:
            last = k
        sc.append(k - last if last >= 0 else k + 1)
    f["since_curtain_change"] = sc
    f["heat_run"] = run_len(he > 0)
    f["co2_hours"] = np.cumsum(co > 0)
    f["vent_max"] = np.maximum.accumulate(v)
    return f


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    F = pd.concat([day_feats(g) for _, g in lab.groupby(["farm", "day"])])
    lab = lab.join(F)
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + NEW
    el = []
    for f in ("F13", "F47"):
        days = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(days), 5):
            el.append(("EL1", len(el), {(f, int(d)) for d in days[k:k + 5]}))
    for name, i, vd in list(fds) + el:
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["%s_%d" % (TAG, s)] = dc5.r3(tr, va, s, FS, BS)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    db1.judge_report(TAG, "DP1")


if __name__ == "__main__":
    main()
