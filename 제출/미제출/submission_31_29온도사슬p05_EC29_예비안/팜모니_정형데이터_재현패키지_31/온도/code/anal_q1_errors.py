# -*- coding: utf-8 -*-
"""Analysis Q1: where does the remaining temperature error live?

Round 4 showed the coldest test rows are already at about the right level, so
the ~11% gap to the leader sits somewhere else.  The calibrated validators
each see only part of the record (geometry A/B: days 63-180, EXTRAP: cold
days only), so for error LOCATION a diagnostic out-of-fold prediction that
covers every labelled day is built: 5-day chunks per greenhouse, dealt
round-robin into 10 folds, each labelled day held out exactly once with the
usual 1-day buffer.  Scored on clean rows only (contaminated rows +-3 h
removed; the test has few of them, see catalogue 4.9).

Configurations: round 3 (plain) and the contaminated-row down-weighting
(w02, 167 flagged rows at weight 0.2).  Saved to local/oof_temp_diag.npz.

Error is split by record section (1st pass <= day 178 / 2nd pass, where all
test days live), greenhouse, hour, daily heating regime, air-temperature
band, source cluster (analysis-only KMeans on daily actuator means) and
day-level vs within-day.

Run:  cd research && PYTHONPATH="" <python> -u anal_q1_errors.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from common import split_mask, rmse, TARGET_FARMS
from harness import load, views
import feat_temp74 as T74
import feat_new
import features_v4 as F4
from screen_v6 import temp_members, collect
from cleanw_v6 import weights

ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]


def diag_folds(lab, chunk=5, k=10):
    out = [{f: set() for f in TARGET_FARMS} for _ in range(k)]
    for f in TARGET_FARMS:
        days = sorted(lab[lab.farm == f].day.unique())
        for i in range(0, len(days), chunk):
            for d in days[i:i + chunk]:
                out[(i // chunk) % k][f].add(int(d))
    return out


def seg(name, key, d):
    print("\n-- by %s --" % name)
    print("%-16s %5s | %7s %7s | %7s %7s | %8s" % ("", "n", "r3", "w02", "bias r3", "bias w02", "SSE% r3"))
    tot = float((d.e_a ** 2).sum())
    for k, g in d.groupby(key, observed=True):
        print("%-16s %5d | %7.3f %7.3f | %+7.3f %+7.3f | %7.1f%%"
              % (str(k)[:16], len(g), np.sqrt((g.e_a ** 2).mean()), np.sqrt((g.e_b ** 2).mean()),
                 g.e_a.mean(), g.e_b.mean(), 100 * float((g.e_a ** 2).sum()) / tot))


def main():
    panel, lab0, _ = load()
    v = views(panel)
    ex, blocks = feat_new.build_extra()
    sg, ph, fp = F4.seg_features(), F4.phys_features(), F4.fp_features()
    lab = (lab0.merge(ex, on="row_id", how="left").merge(sg, on="row_id", how="left")
               .merge(ph, on="row_id", how="left").merge(fp, on="row_id", how="left")).copy()
    f93 = T74.base74(v["temp"]) + list(blocks["dew"]) + list(blocks["event"])
    ct = f93 + F4.names(sg) + F4.names(fp)
    phc = F4.names(ph)
    y = lab.sub_temp.values
    W02 = weights(lab, 0, 0.2)
    clean = weights(lab, 3, 0.0) >= 1

    fds = diag_folds(lab)
    res = {}
    for tag, W in (("r3", None), ("w02", W02)):
        M = collect(lab, fds, lambda tr, va, m: temp_members(tr, va, ct, phc, None if W is None else W[m]))
        res[tag] = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
        print("  %s done" % tag, flush=True)
    np.savez(env.LOCAL + "/oof_temp_diag.npz", row_id=lab.row_id.values, r3=res["r3"], w02=res["w02"])

    g = ~np.isnan(res["r3"]) & clean
    d = lab.loc[g, ["farm", "day", "hour", "ph_in_temp_3", "act_heating"] + ACTS[:3]].copy()
    d = d.loc[:, ~d.columns.duplicated()]
    d["y"] = y[g]
    d["e_a"] = res["r3"][g] - y[g]
    d["e_b"] = res["w02"][g] - y[g]
    print("\nclean scored rows %d of %d | r3 %.4f -> w02 %.4f"
          % (int(g.sum()), len(lab), np.sqrt((d.e_a ** 2).mean()), np.sqrt((d.e_b ** 2).mean())))

    dm = d.groupby(["farm", "day"]).e_a.transform("mean")
    lvl = float(np.sqrt((dm ** 2).mean()))
    wit = float(np.sqrt(((d.e_a - dm) ** 2).mean()))
    print("r3 error split: day-level %.3f | within-day %.3f | day-level share of SSE %.1f%%"
          % (lvl, wit, 100 * lvl ** 2 / (lvl ** 2 + wit ** 2)))

    d["section"] = np.where(d.day <= 178, "1st pass <=178", "2nd pass >=179")
    d["hgrp"] = pd.cut(d.hour, [-1, 0, 6, 12, 18, 23], labels=["00", "01-06", "07-12", "13-18", "19-23"])
    heat = lab.groupby(["farm", "day"]).act_heating.apply(lambda s: float((s >= 99).mean()))
    d["heat_day"] = pd.cut(d.set_index(["farm", "day"]).index.map(heat).values, [-0.01, 0.0, 0.3, 1.0],
                           labels=["no heating", "heat <30%", "heat >=30%"])
    d["band"] = pd.cut(d.ph_in_temp_3, [-5, 8, 10, 12, 15, 18, 40],
                       labels=["<8", "8-10", "10-12", "12-15", "15-18", ">=18"])
    daym = lab.groupby(["farm", "day"])[ACTS].mean().fillna(0)
    km = KMeans(4, n_init=20, random_state=0).fit(StandardScaler().fit_transform(daym))
    cl = pd.Series(km.labels_, index=daym.index)
    d["source_cl"] = ["cl%d" % c for c in d.set_index(["farm", "day"]).index.map(cl).values]

    for name, key in (("record section", "section"), ("greenhouse", "farm"), ("hour", "hgrp"),
                      ("daily heating", "heat_day"), ("air ewm3 band", "band"),
                      ("source cluster (analysis only)", "source_cl")):
        seg(name, key, d)
    seg("greenhouse x section", ["farm", "section"], d)

    print("\n-- worst 12 greenhouse-days (r3, clean rows) --")
    w = d.groupby(["farm", "day"]).agg(n=("e_a", "size"), rmse=("e_a", lambda s: np.sqrt((s ** 2).mean())),
                                       bias=("e_a", "mean"), rmse_w02=("e_b", lambda s: np.sqrt((s ** 2).mean())),
                                       air=("ph_in_temp_3", "mean"), cl=("source_cl", "first"))
    print(w.sort_values("rmse", ascending=False).head(12).round(3).to_string())


if __name__ == "__main__":
    main()
