# -*- coding: utf-8 -*-
"""RD4: were F13/F47 processed like the 48 integer farms (precision cut +
arbitrary fills)?  Test inputs are stated unprocessed, so every check compares
F13/F47 TRAIN with TEST.  2026-10-02 집 클로드 (user question).

Reading of the 48 farms (rd1, IF1-IF3): rounding first, fills afterwards, so
every fill keeps a decimal; 43% of long fills are straight lines, the rest are
smoother than real data (model-like fills).

Checks
  1. precision: last-digit distribution of decimal columns, train vs test
     (a cut/re-scaling would leave missing or uneven last digits)
  2. labels: straight-line stretches (|2nd diff| <= resolution) in sub_temp,
     sub_ec of F13/F47 and sub_temp of the 48 farms
  3. model-like fills: 6-h window roughness (mean |2nd diff|) of in_temp,
     in_hum, in_co2.  Share of TRAIN windows smoother than the TEST 1st/5th
     percentile of the same hour-of-day block (excess over 1%/5% = candidate
     smooth fills).  Train is noisier overall (6.18), so a smooth excess is not
     caused by that.
  4. where the smooth-excess windows sit (days) and the G_C2 / EC v2 errors there

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u rd4_f13f47_processing_v1.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd

from harness import load

D = env.DATA


def split(df):
    p = df.row_id.str.split("_", expand=True)
    df["farm"], df["day"], df["hour"] = p[0], p[1].astype(int), p[2].astype(int)
    df["t"] = df.day * 24 + df.hour
    return df


tX = split(pd.read_csv(os.path.join(D, "train_X.csv")))
sX = split(pd.read_csv(os.path.join(D, "test_X.csv")))
ty = split(pd.read_csv(os.path.join(D, "train_y.csv")))
tX["set"], sX["set"] = "train", "test"
X = pd.concat([tX, sX], ignore_index=True)
T = X[X.farm.isin(["F13", "F47"])].sort_values(["farm", "t"]).reset_index(drop=True)

# 1. last digits
print("1. last-digit share (%) train | test")
for c, k in (("in_temp", 10), ("out_temp", 10), ("out_wspd", 10)):
    for s in ("train", "test"):
        v = T.loc[T.set == s, c].dropna()
        d = np.round(np.abs(v) * k).astype(int) % 10
        print("   %-8s %-5s " % (c, s) + " ".join("%4.1f" % x for x in 100 * d.value_counts(normalize=True).reindex(range(10), fill_value=0)))
Y = ty[ty.farm.isin(["F13", "F47"])]
for c, k in (("sub_temp", 100), ("sub_ec", 1000)):
    v = Y[c].dropna()
    d = np.round(np.abs(v) * k).astype(int) % 10
    print("   %-8s train " % c + " ".join("%4.1f" % x for x in 100 * d.value_counts(normalize=True).reindex(range(10), fill_value=0)))


# 2. straight label stretches
def straight_len(v, t, tol):
    n = len(v)
    ok = np.zeros(n, bool)
    for i in range(1, n - 1):
        if t[i + 1] - t[i] == 1 and t[i] - t[i - 1] == 1 and not np.isnan(v[i - 1:i + 2]).any():
            ok[i] = abs(v[i + 1] - 2 * v[i] + v[i - 1]) <= tol + 1e-9
    L = np.zeros(n, int)
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        L[i - 1:j + 2] = j - i + 3
        i = j + 1
    return L


print("\n2. straight-line label stretches (rows in stretches of >= 6 / >= 8 h)")
Y = ty.sort_values(["farm", "t"])
for name, sub, c, tol in (("F13/F47 sub_temp", Y.farm.isin(["F13", "F47"]), "sub_temp", 0.01),
                          ("F13/F47 sub_ec", Y.farm.isin(["F13", "F47"]), "sub_ec", 0.001),
                          ("48 farms sub_temp", ~Y.farm.isin(["F13", "F47", "F32"]), "sub_temp", 1.0)):
    Ls = []
    for f, g in Y[sub].groupby("farm"):
        Ls.append(straight_len(g[c].values, g.t.values, tol))
    L = np.concatenate(Ls)
    nn = Y.loc[sub, c].notna().sum()
    print("   %-18s %5d rows >=6h, %5d rows >=8h  of %d" % (name, (L >= 6).sum(), (L >= 8).sum(), nn))

# 3. smooth windows
print("\n3. 6-h roughness: share of TRAIN windows smoother than TEST percentile (same 6-h hour block)")
W = 6
rec = []
for c in ("in_temp", "in_hum", "in_co2"):
    for f, g in T.groupby("farm"):
        v, t, s, h = g[c].values, g.t.values, g.set.values, g.hour.values
        for i in range(0, len(g) - W + 1):
            w = v[i:i + W]
            if np.isnan(w).any() or t[i + W - 1] - t[i] != W - 1 or len(set(s[i:i + W])) > 1:
                continue
            rec.append((c, f, int(t[i]), s[i], h[i] // 6, float(np.abs(np.diff(w, 2)).mean()),
                        float(np.ptp(w))))
R = pd.DataFrame(rec, columns=["ch", "farm", "t0", "set", "blk", "rough", "ptp"])
smooth_rows = {}
for c in ("in_temp", "in_hum", "in_co2"):
    out = []
    flagged = []
    for b in range(4):
        r = R[(R.ch == c) & (R.blk == b)]
        te, tr = r[r.set == "test"], r[r.set == "train"]
        # only windows that actually move (ptp >= test median ptp/2) so flat nights are not "smooth fills"
        mv = te.ptp.median() / 2
        te2, tr2 = te[te.ptp >= mv], tr[tr.ptp >= mv]
        q1, q5 = np.quantile(te2.rough, .01), np.quantile(te2.rough, .05)
        out.append("blk%d(%02d-%02dh) <q1 %.1f%% <q5 %.1f%% (test n=%d)"
                   % (b, 6 * b, 6 * b + 5, 100 * (tr2.rough < q1).mean(), 100 * (tr2.rough < q5).mean(), len(te2)))
        flagged.append(tr2[tr2.rough < q1])
    print("   %-8s " % c + " | ".join(out))
    smooth_rows[c] = pd.concat(flagged)

# 4. where, and errors
_, lab, _ = load()
z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
tt = lab.in_temp.values
gg = np.where(np.isnan(tt), 1, np.clip((tt - 8) / 2, 0, 1))
s = "DIAG10"
base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], 0)
cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], 0)
pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{s}.npy")).mean(0)
lab = lab.assign(e=(0.6 - 0.2 * (1 - gg)) * base + (0.2 + 0.4 * (1 - gg)) * cx + 0.2 * gg * pfn - lab.sub_temp)
lab["t"] = lab.day * 24 + lab.hour
E = lab[["farm", "t", "e"]].dropna()
o = pd.read_csv(os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1",
                             "oof_predictions.csv"), encoding="utf-8-sig")
o = o[o.validator == "DIAG10"]
o["t"] = o.day * 24 + o.hour
o["ee"] = o.v2 - o.sub_ec
r = lambda e: float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan
print("\n4. rows inside TRAIN windows smoother than test 1st percentile")
for c, F in smooth_rows.items():
    keys = {(f, t0 + k) for f, t0 in zip(F.farm, F.t0) for k in range(W)}
    m = np.array([(f, t) in keys for f, t in zip(E.farm, E.t)])
    me = np.array([(f, t) in keys for f, t in zip(o.farm, o.t)])
    days = pd.Series([(f, t // 24) for f, t in keys]).value_counts()
    print("   %-8s rows %d, days %d (top %s) | G_C2 RMSE %.3f vs %.3f | EC v2 RMSE %.3f vs %.3f"
          % (c, len(keys), days.size, days.head(5).to_dict(), r(E.e[m]), r(E.e[~m]), r(o.ee[me]), r(o.ee[~me])))
