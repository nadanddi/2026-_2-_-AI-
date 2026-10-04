# -*- coding: utf-8 -*-
"""FZ1 (forensics, fixed before running; 2026-10-05 집 클로드).  Follows FZ0 (6.320: false gated
days cannot be told from true high days by any single input; multivariate upper bound AUC .74).
New information source = the user's domain material (domain_2026-10-03/코멘트_변수별_영향.txt):
  shading curtain closed -> substrate water up, temp down, EC DOWN; thermal curtain: same;
  fogging -> EC DOWN; inside humidity up -> EC DOWN; inside temperature up -> EC UP;
  CO2 up -> photosynthesis up, substrate water DOWN (-> EC up).
(A) ONE domain score fixed from those signs (no data-chosen weights), day level, z per farm over
    all labelled days:   S_low = z(in_hum_mean) + z(act_shade_mean) + z(act_thermal_mean)
                                 + z(act_fog_mean) - z(in_temp_mean) - z(in_co2_mean)
    Prediction: FALSE gated days (model and anchors say high, label < 1) have higher S_low than
    TRUE high days.  Gated days = FZ0 / HK1 broad gate (32 days: 23 true, 9 false).
    Pass (clue): AUC(false vs true) >= .80, one-sided permutation p < .05 (labels permuted within
    farm, 10000), same direction in both farms.  Full-day values = non-causal upper bound first.
    Also descriptive: S_low vs label among ALL sealed days (act_vent_zero >= .8), Spearman.
(B) Dip timing (grower correction after a high run? the student talk: feeding EC is set from
    drainage EC, target 1.2-1.5): same-동 labelled days in calendar order (st_dong_assign, record
    date); a DIP = label < 1.0 with the same 동's nearest earlier AND later labelled days within
    3 dates both >= 1.2.  Report dips, high-run length before each dip vs runs that did not dip,
    and dates since the previous dip (regularity).  Descriptive only."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

H = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv"))
V = ["in_hum_mean", "act_shade_mean", "act_thermal_mean", "act_fog_mean", "in_temp_mean", "in_co2_mean"]
SG = [1, 1, 1, 1, -1, -1]
for f in ("F13", "F47"):
    m = H.farm == f
    for v in V:
        sd = H.loc[m, v].std()
        H.loc[m, "z_" + v] = (H.loc[m, v] - H.loc[m, v].mean()) / (sd if sd > 0 else 1)
H["S_low"] = sum(s * H["z_" + v] for v, s in zip(V, SG))

O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"))
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
g = O.a1.notna() & (O.pm >= .8) & (O.a1 >= 1.0)
G = O[g].groupby(["farm", "day"]).dm.first().reset_index()
G["false"] = (G.dm < 1).astype(int)
G = G.merge(H[["farm", "day", "S_low"] + ["z_" + v for v in V]], on=["farm", "day"], how="left")
auc = roc_auc_score(G.false, G.S_low)
fa = {f: roc_auc_score(G[G.farm == f].false, G[G.farm == f].S_low) for f in ("F13", "F47")}
rng = np.random.default_rng(20261005); cnt = 0
for _ in range(10000):
    y = G.false.values.copy()
    for f in ("F13", "F47"):
        mm = (G.farm == f).values; y[mm] = rng.permutation(y[mm])
    cnt += roc_auc_score(y, G.S_low) >= auc
p = (cnt + 1) / 10001
print("(A) gated days %d (true %d, false %d)" % (len(G), (G.false == 0).sum(), G.false.sum()))
print("    S_low median true %.2f vs false %.2f; AUC %.2f (F13 %.2f, F47 %.2f); one-sided p %.4f" % (
    G[G.false == 0].S_low.median(), G[G.false == 1].S_low.median(), auc, fa["F13"], fa["F47"], p))
print("    component AUCs (sign-adjusted, >.5 = domain direction):",
      {v: round(roc_auc_score(G.false, s * G["z_" + v]), 2) for v, s in zip(V, SG)})
print(G.sort_values("S_low", ascending=False)[["farm", "day", "dm", "false", "S_low"]].round(2).to_string(index=False))
S = H[H.act_vent_zero >= .8]
print("    all sealed days n %d: Spearman(S_low, label) %.2f; all labelled days %.2f" % (
    len(S), spearmanr(S.S_low, S.ec).correlation, spearmanr(H.S_low, H.ec).correlation))
clue = auc >= .80 and p < .05 and min(fa.values()) > .5
print("FZ1 (A) clue:", clue)

# (B) dip timing
R = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).sort_values(["farm", "day"])
R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
R = R.merge(H[["farm", "day", "ec"]], on=["farm", "day"], how="left")
print("\n(B) dips (same 동, label < 1 between earlier/later same-동 labels >= 1.2 within 3 dates):")
rows = []
for (f, dg), K in R.dropna(subset=["ec"]).groupby(["farm", "dong"]):
    K = K.sort_values("date").reset_index(drop=True)
    for i in range(1, len(K) - 1):
        a, b, c = K.iloc[i - 1], K.iloc[i], K.iloc[i + 1]
        if b.ec < 1 and a.ec >= 1.2 and c.ec >= 1.2 and b.date - a.date <= 3 and c.date - b.date <= 3:
            run = 0
            for j in range(i - 1, -1, -1):
                if K.iloc[j].ec >= 1.0 and (K.iloc[j + 1].date - K.iloc[j].date) <= 3:
                    run += 1
                else:
                    break
            rows.append(dict(farm=f, dong=dg, day=int(b.day), date=int(b.date), ec=b.ec, before=a.ec, after=c.ec, high_run_before=run))
Dp = pd.DataFrame(rows)
print(Dp.round(2).to_string(index=False) if len(Dp) else "    none")
if len(Dp) > 1:
    for (f, dg), K in Dp.groupby(["farm", "dong"]):
        if len(K) > 1:
            print("    %s %s dates between dips:" % (f, dg), np.diff(sorted(K.date)).tolist())
# run lengths of high episodes that ended without a dip vs with one
ep = []
for (f, dg), K in R.dropna(subset=["ec"]).groupby(["farm", "dong"]):
    K = K.sort_values("date"); run = 0; last = None
    for _, r in K.iterrows():
        if r.ec >= 1.0 and (last is None or r.date - last <= 3):
            run += 1
        else:
            if run:
                ep.append(run)
            run = 1 if r.ec >= 1.0 else 0
        last = r.date
    if run:
        ep.append(run)
print("    high-run lengths (same 동, labelled, gaps <= 3 dates):", sorted(ep))
