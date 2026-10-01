# -*- coding: utf-8 -*-
"""EC source search C2: anomalies in the TEST inputs themselves that could
carry label-side information.  Rules fixed in
EC_정보원_입력쪽_사전고정_2026-10-02.md (commit 81ce501).  2026-10-02 집 클로드.

(a) values that are rare in F13/F47 train (<= 0.1% of rows) per channel
(b) 7-actuator combinations per row
(c) row order / duplicates / id gaps in test_X and sample_submission
(d) decimal precision per channel
(e) stuck runs (same value >= 6 h) per channel, by hour
A pattern is "enriched" if its test row share >= 2x the train share and >= 1%
of test rows.  For every enriched pattern, train days containing it vs not are
compared on daily mean EC (Mann-Whitney); GO if any p < 0.01 / m (m = number
of enriched patterns).  Locked 40 days' labels are not read.

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_src_C2_test_anomaly_v1.py
"""
import env  # noqa: F401
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ACT = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
NUM = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2"] + ACT
LOCK = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")


def split(df):
    p = df.row_id.str.split("_", expand=True)
    df["farm"], df["day"], df["hour"] = p[0], p[1].astype(int), p[2].astype(int)
    return df


def main():
    tX = split(pd.read_csv(os.path.join(env.DATA, "train_X.csv")))
    sX = split(pd.read_csv(os.path.join(env.DATA, "test_X.csv")))
    ss = pd.read_csv(os.path.join(env.DATA, "sample_submission.csv"))
    ty = split(pd.read_csv(os.path.join(env.DATA, "train_y.csv")))
    lock = {(s["farm"], int(s["day"])) for s in json.load(open(LOCK, encoding="utf-8"))["selected"]}
    tr = tX[tX.farm.isin(["F13", "F47"])].sort_values(["farm", "day", "hour"]).reset_index(drop=True)
    te = sX.copy()
    y = ty[ty.farm.isin(["F13", "F47"]) & ty.sub_ec.notna()]
    y = y[[(f, d) not in lock for f, d in zip(y.farm, y.day)]]
    ec = y.groupby(["farm", "day"]).sub_ec.mean()

    pats = {}  # name -> (train row mask, test row mask)
    print("(a) train-rare values (<= 0.1% of train rows)")
    for c in NUM:
        vc = tr[c].value_counts(normalize=True)
        rare = set(vc[vc <= 0.001].index)
        mtr = tr[c].isin(rare)
        unseen = ~te[c].isin(set(vc.index)) & te[c].notna()
        mte = te[c].isin(rare) | unseen
        pats["rare_" + c] = (mtr.values, mte.values)
        print("   %-12s train %.2f%%  test %.2f%% (of which unseen in train %.2f%%)"
              % (c, 100 * mtr.mean(), 100 * mte.mean(), 100 * unseen.mean()))

    print("(b) actuator combinations")
    ktr = tr[ACT].astype(str).agg("|".join, axis=1)
    kte = te[ACT].astype(str).agg("|".join, axis=1)
    ftr, fte = ktr.value_counts(normalize=True), kte.value_counts(normalize=True)
    for k, sh in fte.items():
        if sh >= 0.01 and sh >= 2 * ftr.get(k, 0):
            pats["combo_" + k] = ((ktr == k).values, (kte == k).values)
            print("   %s  test %.2f%%  train %.3f%%" % (k, 100 * sh, 100 * ftr.get(k, 0)))

    print("(c) order / duplicates / gaps")
    srt = te.sort_values(["farm", "day", "hour"]).row_id.tolist()
    print("   test_X sorted by (farm, day, hour): %s; duplicates %d; sample_submission same order as test_X: %s"
          % (srt == te.row_id.tolist(), te.row_id.duplicated().sum(), ss.row_id.tolist() == te.row_id.tolist()))
    for f, g in te.groupby("farm"):
        t = (g.day * 24 + g.hour).sort_values().values
        print("   %s: hours %d, internal hour gaps %s, days per block %s" % (
            f, len(t), int((np.diff(t) > 1).sum()),
            [int(x) for x in np.diff(np.r_[0, np.where(np.diff(sorted(g.day.unique())) > 1)[0] + 1, g.day.nunique()])]))
    nn = te[NUM].isna().sum()
    print("   test missing per channel:", nn[nn > 0].to_dict() or "none")

    print("(d) decimal precision (max decimals) train | test")
    for c in NUM:
        def mx(s):
            s = s.dropna().astype(str)
            return int(s.str.split(".").str[1].fillna("").str.rstrip("0").str.len().max())
        print("   %-12s %d | %d" % (c, mx(tr[c]), mx(te[c])))

    print("(e) stuck runs (same value >= 6 consecutive hours)")
    for c in ["in_temp", "in_hum", "in_co2", "out_temp", "out_hum", "out_wspd"]:
        def stuck(df):
            m = np.zeros(len(df), bool)
            for _, g in df.groupby("farm"):
                v = g[c].values
                idx = g.index.values
                i = 0
                while i < len(v):
                    j = i
                    while j + 1 < len(v) and v[j + 1] == v[i]:
                        j += 1
                    if j - i + 1 >= 6 and not np.isnan(v[i]):
                        m[np.searchsorted(df.index.values, idx[i:j + 1])] = True
                    i = j + 1
            return m
        a = stuck(tr.reset_index(drop=True))
        b = stuck(te.sort_values(["farm", "day", "hour"]).reset_index(drop=True))
        pats["stuck_" + c] = (a, b)
        print("   %-9s train %.2f%%  test %.2f%%" % (c, 100 * a.mean(), 100 * b.mean()))

    # enriched patterns and EC association (train days)
    enr = {k: v for k, v in pats.items() if v[1].mean() >= 0.01 and v[1].mean() >= 2 * v[0].mean()}
    m = max(len(enr), 1)
    print("\nenriched patterns (test >= 2x train, >= 1%% of test rows): %d -> threshold %.4f" % (len(enr), 0.01 / m))
    trd = tr[["farm", "day"]].copy()
    go = False
    for k, (a, b) in enr.items():
        dd = trd[a].drop_duplicates()
        has = set(map(tuple, dd.values))
        e1 = [v for kk, v in ec.items() if kk in has]
        e0 = [v for kk, v in ec.items() if kk not in has]
        if len(e1) >= 3 and len(e0) >= 3:
            p = mannwhitneyu(e1, e0).pvalue
        else:
            p = np.nan
        go |= (p < 0.01 / m) if not np.isnan(p) else False
        print("   %-40s test %.1f%% train %.2f%% | train days with %d (EC median %.3f) vs without %d (%.3f) p %.4g"
              % (k[:40], 100 * b.mean(), 100 * a.mean(), len(e1), np.median(e1) if e1 else np.nan, len(e0),
                 np.median(e0), p))
    print("\nC2 decision:", "GO" if go else "STOP")


if __name__ == "__main__":
    main()
