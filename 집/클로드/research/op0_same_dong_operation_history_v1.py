# -*- coding: utf-8 -*-
"""OP0 (diagnostic, fixed before running; 2026-10-04 집 클로드).
Question: is the SIZE error of high-EC days explained by the same 동's OPERATION
history over previous dates (inputs only; legal at test time as same-greenhouse
previous inputs)?  HC5 found the same 동's previous EC labels relate to size (+.41),
but labels are absent in pass 2; this asks whether inputs can stand in for them.
Not tested before: MW1 used previous-day WEATHER, AG0 used predicted-high streaks.
Records: all train+test records of F13/F47 (460), calendar date and 동 from
local/st_dong_assign_v1.csv (pair role / input classifier, no labels).
Daily inputs per record (nan-mean over hours): sealed fraction (act_vent==0),
heating, co2 supply, thermal curtain, in_temp, in_hum, in_co2, out_rad.
Features: mean over previous 3 and 7 calendar dates of the same 동 (today excluded),
plus sealed streak (consecutive previous same-동 dates with sealed fraction >= .8).
Target: DIAG10 R3S seed-mean day residual (label mean - prediction mean).
Sets: high days (label >= 1, n ~31) primary; pass-2 labelled days secondary.
Benchmark: same statistics for the same 동's previous labelled EC (HC5 style).
Clue (fixed): on high days, the best feature's family-wise permutation p
(residuals permuted within farm, 10000, max |Spearman| over all features) < .05
AND the same sign in both farms.  Otherwise no model test."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

R = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).sort_values(["farm", "day"]).reset_index(drop=True)
R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")], ignore_index=True)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int)
X["sealed"] = np.where(X.act_vent.isna(), np.nan, (X.act_vent == 0).astype(float))
for c in ("act_heating", "act_co2", "act_thermal"):
    X[c + "_on"] = np.where(X[c].isna(), np.nan, (X[c] > 0).astype(float))
V = ["sealed", "act_heating_on", "act_co2_on", "act_thermal_on", "in_temp", "in_hum", "in_co2", "out_rad"]
DX = X.groupby(["farm", "day"])[V].mean().reset_index()
A = R.merge(DX, on=["farm", "day"], how="left")
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy(); Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
A = A.merge(Y.groupby(["farm", "day"]).sub_ec.mean().rename("ec").reset_index(), on=["farm", "day"], how="left")
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy()
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
A = A.merge(O.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean")).reset_index(), on=["farm", "day"], how="left")
A["res"] = A.y - A.p

rows = []
for _, r in A.iterrows():
    S = A[(A.farm == r.farm) & (A.dong == r.dong) & (A.date < r.date)].sort_values("date")
    f = dict(farm=r.farm, day=r.day)
    for K in (3, 7):
        W = S[S.date >= r.date - K]
        for v in V:
            f["%s_%d" % (v, K)] = W[v].mean()
    st = 0
    for _, q in S.iloc[::-1].iterrows():
        if pd.notna(q.sealed) and q.sealed >= .8:
            st += 1
        else:
            break
    f["seal_streak"] = st
    lab = S[S.date >= r.date - 7].dropna(subset=["ec"])
    f["bench_prev_ec_mean"] = lab.ec.mean() if len(lab) else np.nan
    f["bench_prev_ec_min"] = lab.ec.min() if len(lab) else np.nan
    rows.append(f)
A = A.merge(pd.DataFrame(rows), on=["farm", "day"])
A.to_csv(os.path.join(env.LOCAL, "op0_features_v1.csv"), index=False)
FE = ["%s_%d" % (v, K) for K in (3, 7) for v in V] + ["seal_streak"]
BE = ["bench_prev_ec_mean", "bench_prev_ec_min"]


def table(S, title):
    print("\n%s: n %d (mean label %.3f, pred %.3f, residual %.3f)" % (title, len(S), S.y.mean(), S.p.mean(), S.res.mean()))
    out = {}
    for c in FE + BE:
        z = S[[c, "res", "farm"]].dropna()
        rho = spearmanr(z[c], z.res).correlation
        fs = []
        for fm in ("F13", "F47"):
            zz = z[z.farm == fm]
            fs.append(spearmanr(zz[c], zz.res).correlation if len(zz) >= 5 else np.nan)
        out[c] = (rho, fs, len(z))
        print("   %-22s rho %+.2f  F13 %+.2f  F47 %+.2f  n %d" % (c, rho, fs[0], fs[1], len(z)))
    return out


H = A[A.y >= 1.0].dropna(subset=["res"]).copy()
P2 = A[(A.day >= 179) & A.res.notna()].copy()
oh = table(H, "HIGH-EC days (label >= 1): residual vs same-dong previous operation")
table(P2, "PASS-2 labelled days: residual vs same-dong previous operation")

# family-wise permutation on high days (operation features only)
Z = H[FE + ["res", "farm"]].copy()
Zr = Z[FE].rank()
def maxabs(res):
    rr = pd.Series(res, index=Z.index).rank()
    return np.nanmax(np.abs([Zr[c].corr(rr) for c in FE]))
obs = maxabs(Z.res.values)
rng = np.random.default_rng(20261004); cnt = 0; NP = 10000
for _ in range(NP):
    perm = Z.res.values.copy()
    for fm in ("F13", "F47"):
        m = (Z.farm == fm).values
        perm[m] = rng.permutation(perm[m])
    cnt += maxabs(perm) >= obs
pfw = (cnt + 1) / (NP + 1)
best = max(FE, key=lambda c: abs(oh[c][0]) if np.isfinite(oh[c][0]) else -1)
rho, fs, n = oh[best]
same = np.all(np.sign(fs) == np.sign(rho))
print("\nbest operation feature on high days: %s rho %+.2f (F13 %+.2f, F47 %+.2f), family-wise p %.4f" % (best, rho, fs[0], fs[1], pfw))
print("benchmark (labels, not usable in pass 2): prev_ec_mean rho %+.2f, prev_ec_min rho %+.2f" % (oh["bench_prev_ec_mean"][0], oh["bench_prev_ec_min"][0]))
print("\nOP0 clue:", bool(pfw < .05 and same))
