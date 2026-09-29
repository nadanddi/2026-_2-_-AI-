# -*- coding: utf-8 -*-
"""Inspector 4-1: robustness of the 'sealed / fan-off day' claims (catalog 6.11-6.13).
Read-only; writes nothing except stdout (caller tees to local/audit4_1_sealed.log)."""
import env  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from harness import load
from anal_e3_spikes import day_table
import train_flags_v6 as TF

panel, _, lab = load()
z = np.load(env.LOCAL + "/oof_ec_diag.npz", allow_pickle=True)
oof = pd.Series(z["oof"], index=z["row_id"])
lab = lab.copy(); lab["p"] = oof.loc[lab.row_id].values
tr = day_table(lab)
lv = lab.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean")).reset_index()
tr = tr.merge(lv, on=["farm", "day"])
te = day_table(panel[panel.is_test])
print("train labelled days %d | test days %d" % (len(tr), len(te)))

# 1. threshold grid
print("\n== threshold grid: fan_mean<F & vent0share>V ==")
print("%5s %5s | %4s %5s %5s %6s | %4s | %6s %6s" % ("F", "V", "nTr", "ECin", "ECout", "gap", "nTe", "rmseIn", "rmseOut"))
res = []
for F in (5, 10, 20, 30, 50):
    for V in (0.7, 0.8, 0.85, 0.9, 0.95):
        mt = (tr.act_circfan_mean < F) & (tr.act_vent_zero > V)
        me = (te.act_circfan_mean < F) & (te.act_vent_zero > V)
        m = lab.merge(tr[["farm", "day"]].assign(s=mt.values), on=["farm", "day"]).s.values
        e = lab.p.values - lab.sub_ec.values
        ri, ro = np.sqrt(np.mean(e[m] ** 2)), np.sqrt(np.mean(e[~m] ** 2))
        res.append((F, V, mt.sum(), tr.y[mt].mean(), tr.y[~mt].mean(), me.sum(), ri, ro))
        print("%5d %5.2f | %4d %5.2f %5.2f %6.2f | %4d | %6.3f %6.3f" % (F, V, mt.sum(), tr.y[mt].mean(), tr.y[~mt].mean(),
              tr.y[mt].mean() - tr.y[~mt].mean(), me.sum(), ri, ro))
# single-variable versions
for nm, mt, me in (("fan<10 only", tr.act_circfan_mean < 10, te.act_circfan_mean < 10),
                   ("vent0>0.85 only", tr.act_vent_zero > 0.85, te.act_vent_zero > 0.85)):
    print("%-16s nTr %d EC %.2f vs %.2f | nTe %d" % (nm, mt.sum(), tr.y[mt].mean(), tr.y[~mt].mean(), me.sum()))

# season confound: within late season only
late = tr.day.between(120, 178) | (tr.day >= 179)
s = (tr.act_circfan_mean < 10) & (tr.act_vent_zero > 0.85)
print("\nsealed share by season band:")
for lo, hi in ((0, 60), (60, 120), (120, 179), (179, 260)):
    m = tr.day.between(lo, hi - 1)
    print("  days %3d-%3d: n %3d sealed %3d (%.0f%%) EC sealed %.2f non %.2f" % (lo, hi - 1, m.sum(), (s & m).sum(),
          100 * (s & m).mean() / max(m.mean(), 1e-9), tr.y[s & m].mean() if (s & m).any() else np.nan, tr.y[~s & m].mean()))
# rank of 'sealed' vs a simple season-only predictor
from sklearn.metrics import roc_auc_score
spike = tr.y > 1.2
print("AUC spike: sealed flag %.3f | day number %.3f | out_temp_mean(-) %.3f | in_temp_mean(-) %.3f"
      % (roc_auc_score(spike, s), roc_auc_score(spike, tr.day), roc_auc_score(spike, -tr.out_temp_mean),
         roc_auc_score(spike, -tr.in_temp_mean)))

# 2. per-day error concentration & expected test weight of sealed days
tr["drmse"] = lab.assign(e2=(lab.p - lab.sub_ec) ** 2).groupby(["farm", "day"]).e2.mean().values ** .5 \
    if False else np.nan
e2 = lab.assign(e2=(lab.p - lab.sub_ec) ** 2).groupby(["farm", "day"]).e2.mean().rename("mse").reset_index()
tr = tr.merge(e2, on=["farm", "day"])
ms, mn = tr.mse[s].mean(), tr.mse[~s].mean()
print("\nDIAG10 OOF day MSE: sealed %.4f (rmse %.3f) | other %.4f (rmse %.3f)" % (ms, ms ** .5, mn, mn ** .5))
p2 = tr.day >= 179
ms2, mn2 = tr.mse[s & p2].mean(), tr.mse[~s & p2].mean()
print("pass2 only: sealed n=%d rmse %.3f | other n=%d rmse %.3f" % ((s & p2).sum(), ms2 ** .5, (~s & p2).sum(), mn2 ** .5))
for a, b, nm in ((ms, mn, "all"), (ms2, mn2, "pass2")):
    sh = 21 * a / (21 * a + 39 * b)
    print("expected share of test SSE from 21 sealed days (%s rates): %.0f%%" % (nm, 100 * sh))
# bootstrap the share
rng = np.random.RandomState(0)
S, N = tr.mse[s].values, tr.mse[~s].values
sh = []
for _ in range(2000):
    a = S[rng.randint(0, len(S), 21)].mean(); b = N[rng.randint(0, len(N), 39)].mean()
    sh.append(21 * a / (21 * a + 39 * b))
print("share bootstrap (21 sealed / 39 other days drawn from training OOF): median %.0f%% [%.0f%%, %.0f%%]"
      % tuple(100 * np.percentile(sh, [50, 2.5, 97.5])))

# 3. rough vs sealed
nd = TF.noisy_days()
thr = nd.noise_score.quantile(0.75)
rough = set(map(tuple, nd[nd.noise_score >= thr][["farm", "day"]].values))
tr["rough"] = [(f, d) in rough for f, d in zip(tr.farm, tr.day)]
a, b = (s & tr.rough).sum(), (s & ~tr.rough).sum()
c, d = (~s & tr.rough).sum(), (~s & ~tr.rough).sum()
print("\nrough|sealed %d/%d=%.0f%% | rough|other %d/%d=%.0f%% | Fisher p=%.2g" % (a, a + b, 100 * a / (a + b), c, c + d,
      100 * c / (c + d), fisher_exact([[a, b], [c, d]])[1]))
# season-matched: within days>=120
m = tr.day >= 120
a, b = (s & tr.rough & m).sum(), (s & ~tr.rough & m).sum()
c, d = (~s & tr.rough & m).sum(), (~s & ~tr.rough & m).sum()
print("days>=120: rough|sealed %d/%d=%.0f%% | rough|other %d/%d=%.0f%% | Fisher p=%.2g" % (a, a + b, 100 * a / max(a + b, 1), c, c + d,
      100 * c / max(c + d, 1), fisher_exact([[a, b], [c, d]])[1]))
# Is the CO2 roughness mechanically caused by sealing? CO2 difference ac1 in test sealed vs train sealed
te_s = (te.act_circfan_mean < 10) & (te.act_vent_zero > 0.85)
print("co2_ac1 median: train sealed %.2f | train other %.2f | test sealed %.2f | test other %.2f"
      % (tr.co2_ac1[s].median(), tr.co2_ac1[~s].median(), te.co2_ac1[te_s].median(), te.co2_ac1[~te_s].median()))
print("co2_ac1 median train sealed & NOT rough %.2f | train sealed & rough %.2f"
      % (tr.co2_ac1[s & ~tr.rough].median(), tr.co2_ac1[s & tr.rough].median()))
print("EC mean sealed&rough %.2f | sealed&clean %.2f ; OOF rmse sealed&rough %.3f | sealed&clean %.3f"
      % (tr.y[s & tr.rough].mean(), tr.y[s & ~tr.rough].mean(), tr.mse[s & tr.rough].mean() ** .5, tr.mse[s & ~tr.rough].mean() ** .5))
print("test sealed days:", sorted(zip(te.farm[te_s], te.day[te_s])))
