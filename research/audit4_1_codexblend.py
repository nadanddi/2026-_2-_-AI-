# -*- coding: utf-8 -*-
"""Inspector 4-1: probe Codex resid_reset 20% blend (F60ND weights) and the round-5 increments.
Re-fits the Codex model on EXT8/10/12 (DIAG10 taken from cache).  Blend weights other than 0.2 are
POST-HOC sensitivity only.  Saves local/audit4_1_codexblend.npz.  Modifies no existing file."""
import env  # noqa: F401
import numpy as np
import pandas as pd
import common
from common import split_mask, rmse, TARGET_FARMS
from harness import load
from anal_q1_errors import diag_folds
from screen_v6 import boot
import train_flags_v6 as TF
import verify_codex_resid_reset as V
from features_v4 import phys_features

tX, ty, sX = common.load_raw()
_, lab0, _ = load()
z = np.load(env.LOCAL + "/eval_v6_oof.npz", allow_pickle=True)
F = V.build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
lab = lab0[["row_id", "farm", "day", "hour", "t", "sub_temp"]].copy().join(F, on="row_id")
w = TF.row_weights(lab, 0.2, w_noisy=0.2)
y = lab.sub_temp.values
ph = phys_features().set_index("row_id").loc[lab.row_id]
dmin = ph.groupby([lab.farm.values, lab.day.values]).ph_in_temp_3.min()
new = {"DIAG10": np.load(env.LOCAL + "/verify_codex_resid_reset_diag.npz")["new"]}
for th in (8, 10, 12):
    cd = dmin[dmin < th]
    fd = {f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}
    trm, vam = split_mask(lab, fd)
    o = np.full(len(lab), np.nan)
    o[vam] = V.fit_predict(lab[trm], lab[vam], w[trm])
    new["EXT%d" % th] = o
    print("EXT%d refit done, days %d" % (th, len(cd)), flush=True)
np.savez(env.LOCAL + "/audit4_1_codexblend.npz", row_id=lab.row_id.values, **new)

L = lab[["farm", "day", "hour"]].reset_index(drop=True)
print("\n== blend-weight sensitivity (0.2 = pre-registered; others POST-HOC) ==")
for s in ("DIAG10", "EXT8", "EXT10", "EXT12"):
    b = z["F60ND__%s" % s]; g = ~np.isnan(b) & ~np.isnan(new[s])
    line = "%-6s n=%5d base %.5f new %.4f |" % (s, g.sum(), rmse(b[g], y[g]), rmse(new[s][g], y[g]))
    for a in (0.1, 0.2, 0.3, 0.4, 0.5):
        line += " %.1f:%.5f" % (a, rmse(((1 - a) * b + a * new[s])[g], y[g]))
    print(line)
    pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_temp", b[g], (0.8 * b + 0.2 * new[s])[g])
    print("       blend20 delta %+.5f [%+.5f,%+.5f] P(worse) %.3f  (%+.2f%%)" % (pr, lo, hi, pw, 100 * pr / rmse(b[g], y[g])))
    # null diversity control: blend with the plain (unweighted round-3) OOF of the same folds
    pl = z["plain__%s" % s]
    print("       control 0.8*F60ND+0.2*plain %.5f ; 0.8*F60ND+0.2*F60 %.5f" % (rmse((0.8 * b + 0.2 * pl)[g], y[g]),
          rmse((0.8 * b + 0.2 * z["F60__%s" % s])[g], y[g])))
    print("       corr(err base, err new) %.3f" % np.corrcoef((b - y)[g], (new[s] - y)[g])[0, 1])

# DIAG10 structure
s = "DIAG10"; b = z["F60ND__DIAG10"]; bl = 0.8 * b + 0.2 * new[s]
D = L.assign(eb=(b - y) ** 2, el=(bl - y) ** 2)
D["pass2"] = D.day >= 179
print("\n== DIAG10 by farm x pass ==")
for (f, p2), gg in D.groupby(["farm", "pass2"]):
    print("  %s pass%d days %3d: base %.4f blend %.4f (%+.2f%%)" % (f, 2 if p2 else 1, gg.day.nunique(), gg.eb.mean() ** .5,
          gg.el.mean() ** .5, 100 * ((gg.el.mean() / gg.eb.mean()) ** .5 - 1)))
print("== DIAG10 by hour band ==")
for hb in ((0, 3), (4, 9), (10, 15), (16, 23)):
    gg = D[D.hour.between(*hb)]
    print("  %02d-%02d base %.4f blend %.4f (%+.2f%%)" % (hb + (gg.eb.mean() ** .5, gg.el.mean() ** .5,
          100 * ((gg.el.mean() / gg.eb.mean()) ** .5 - 1))))
dd = D.groupby(["farm", "day"])[["eb", "el"]].sum()
dd["gain"] = dd.eb - dd.el
tot = dd.gain.sum()
srt = dd.gain.sort_values(ascending=False)
print("day-level SSE gain: total %.1f ; days improved %d/%d ; top5 days = %.0f%% of net, top10 = %.0f%%, top20 = %.0f%%"
      % (tot, (dd.gain > 0).sum(), len(dd), 100 * srt.head(5).sum() / tot, 100 * srt.head(10).sum() / tot, 100 * srt.head(20).sum() / tot))
print("net gain without top10 days: %.0f%% of total remains" % (100 * (tot - srt.head(10).sum()) / tot))
# per DIAG fold
fds = diag_folds(lab)
fr = []
for i, fd in enumerate(fds):
    _, vam = split_mask(lab, fd)
    fr.append((rmse(b[vam], y[vam]), rmse(bl[vam], y[vam])))
fr = np.array(fr)
print("DIAG10 per-fold improved %d/10; per-fold rel deltas %s" % ((fr[:, 1] < fr[:, 0]).sum(),
      np.round(100 * (fr[:, 1] / fr[:, 0] - 1), 2).tolist()))

# EXT dependence: EXT12-only days vs EXT10 days, inside the EXT12 run
b12 = z["F60ND__EXT12"]; n12 = new["EXT12"]; g12 = ~np.isnan(b12)
cd10 = set(dmin[dmin < 10].index)
in10 = np.array([(f, d) in cd10 for f, d in zip(L.farm, L.day)])
for nm, m in (("EXT12 run, EXT10 days", g12 & in10), ("EXT12 run, 10-12C-only days", g12 & ~in10)):
    bb = 0.8 * b12 + 0.2 * n12
    print("%-30s n=%5d base %.4f blend %.4f (%+.2f%%)" % (nm, m.sum(), rmse(b12[m], y[m]), rmse(bb[m], y[m]),
          100 * (rmse(bb[m], y[m]) / rmse(b12[m], y[m]) - 1)))
print("EXT10 day set is a subset of EXT12: %d of %d EXT12 days" % (len(cd10), (dmin < 12).sum()))

# 60-day sampling noise: how much does a relative change measured on 60 random days (30/farm) scatter?
rng = np.random.RandomState(1)
days = {f: sorted(L[L.farm == f].day.unique()) for f in TARGET_FARMS}
dsse = L.assign(p=(z["plain__DIAG10"] - y) ** 2, q=(z["F60ND__DIAG10"] - y) ** 2, r=(bl - y) ** 2).groupby(["farm", "day"])[["p", "q", "r"]].sum()
cnt = L.groupby(["farm", "day"]).size()
res5, resC = [], []
for _ in range(4000):
    pick = [(f, d) for f in TARGET_FARMS for d in rng.choice(days[f], 30, replace=False)]
    sub = dsse.loc[pick]; n = cnt.loc[pick].sum()
    res5.append((sub.q.sum() / sub.p.sum()) ** .5 - 1)
    resC.append((sub.r.sum() / sub.q.sum()) ** .5 - 1)
for nm, r_ in (("plain->F60ND (round 5 change)", res5), ("F60ND->blend20 (Codex)", resC)):
    r_ = 100 * np.array(r_)
    print("60-day subsample rel change %-32s median %+.2f%%, 90%% range [%+.2f%%, %+.2f%%], P(worse) %.2f"
          % (nm, np.median(r_), np.percentile(r_, 5), np.percentile(r_, 95), (r_ > 0).mean()))
# same restricted to pass-2 + cold-ish days (test-like): days with daily min 3h air < 12
cold12 = set(dmin[dmin < 12].index)
for nm, col in (("plain->F60ND", ("q", "p")), ("F60ND->blend20", ("r", "q"))):
    pool = [k for k in dsse.index if k in cold12]
    rr = []
    for _ in range(4000):
        idx = rng.choice(len(pool), 60, replace=False)
        sub = dsse.loc[[pool[i] for i in idx]]
        rr.append((sub[col[0]].sum() / sub[col[1]].sum()) ** .5 - 1)
    rr = 100 * np.array(rr)
    print("60 cold(<12C) DIAG10 days %-16s median %+.2f%%, 90%% [%+.2f%%, %+.2f%%], P(worse) %.2f" % (nm, np.median(rr),
          np.percentile(rr, 5), np.percentile(rr, 95), (rr > 0).mean()))
