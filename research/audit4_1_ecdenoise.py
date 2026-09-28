# -*- coding: utf-8 -*-
"""Inspector 4-1: is the EC input-smoothing gain (6.20, -0.9% on A 5/5) distinguishable from noise?
Re-runs base / Strain (as in ec_denoise_v1) + base with another seed (seed-noise yardstick).
Saves OOFs to local/audit4_1_ecdenoise.npz.  Does not modify any file."""
import env  # noqa: F401
import numpy as np
import pandas as pd
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
import ec_v6
from screen_v6 import boot
import train_flags_v6 as TF

NOISY = ["in_temp", "in_hum", "in_co2"]
panel, _, lab0 = load()
fp = F4.fp_features(); fpc = F4.names(fp)
lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
p = panel.sort_values(["farm", "t"])
S = pd.DataFrame({c + "_s": p.groupby("farm")[c].transform(lambda s: s.ewm(halflife=1.5, ignore_na=True).mean()).values
                  for c in NOISY}, index=p.row_id.values)
for c in NOISY:
    lab[c + "_s"] = S.loc[lab.row_id, c + "_s"].values
nd = TF.noisy_days()
rs = set(map(tuple, nd[nd.noise_score >= nd.noise_score.quantile(0.75)][["farm", "day"]].values))
lab["rough"] = [(f, d) in rs for f, d in zip(lab.farm, lab.day)]


def run(tr, va, seed, strain):
    ec_v6.SEED = seed
    fn = ec_v6.pipeline(f14 + fpc, f14)
    if strain:
        tr = tr.copy(); m = tr.rough.values
        for c in NOISY:
            tr.loc[m, c] = tr.loc[m, c + "_s"]
    return fn(tr, va)


y = lab.sub_ec.values
out = {}
for vset, fds in (("A", folds("A")), ("DIAG10", diag_folds(lab))):
    per = []
    for k in ("base7", "strain7", "base8", "strain8"):
        out[(vset, k)] = np.full(len(lab), np.nan)
    for i, fd in enumerate(fds):
        trm, vam = split_mask(lab, fd)
        tr, va = lab[trm], lab[vam].reset_index(drop=True)
        idx = np.where(vam)[0]
        r = {}
        for k, sd, st in (("base7", 7, False), ("strain7", 7, True), ("base8", 8, False), ("strain8", 8, True)):
            pr = run(tr, va, sd, st)
            out[(vset, k)][idx] = pr
            r[k] = rmse(pr, y[idx])
        if vset == "A":
            sub = lab.iloc[idx].reset_index(drop=True)
            d7 = boot(sub, "sub_ec", out[(vset, "base7")][idx], out[(vset, "strain7")][idx])
            ds = boot(sub, "sub_ec", out[(vset, "base7")][idx], out[(vset, "base8")][idx])
            print("A fold %d: base7 %.4f strain7 %.4f (%+.1f%%, CI [%+.4f,%+.4f]) | base8 %.4f (seed diff %+.1f%%) strain8 %.4f"
                  % (i, r["base7"], r["strain7"], 100 * (r["strain7"] / r["base7"] - 1), d7[1], d7[2], r["base8"],
                     100 * (r["base8"] / r["base7"] - 1), r["strain8"]), flush=True)
        per.append(r)
        print("  %s fold %d done" % (vset, i), flush=True)
    P = pd.DataFrame(per)
    print("\n== %s == fold-mean: %s" % (vset, P.mean().round(4).to_dict()))
    print("   strain vs base: seed7 %+.2f%% (better %d/%d) | seed8 %+.2f%% (better %d/%d) | base8 vs base7 %+.2f%%"
          % (100 * (P.strain7.mean() / P.base7.mean() - 1), (P.strain7 < P.base7).sum(), len(P),
             100 * (P.strain8.mean() / P.base8.mean() - 1), (P.strain8 < P.base8).sum(), len(P),
             100 * (P.base8.mean() / P.base7.mean() - 1)))
    if vset == "DIAG10":
        for a, b in (("base7", "strain7"), ("base8", "strain8"), ("base7", "base8")):
            pr, lo, hi, pw = boot(lab, "sub_ec", out[(vset, a)], out[(vset, b)])
            print("   pooled %s->%s %+.4f [%+.4f,%+.4f] P(worse) %.3f (%+.2f%%)" % (a, b, pr, lo, hi, pw,
                  100 * pr / rmse(out[(vset, a)], y)))
        avg_b = (out[(vset, "base7")] + out[(vset, "base8")]) / 2
        avg_s = (out[(vset, "strain7")] + out[(vset, "strain8")]) / 2
        pr, lo, hi, pw = boot(lab, "sub_ec", avg_b, avg_s)
        print("   pooled 2-seed avg base->strain %+.4f [%+.4f,%+.4f] P(worse) %.3f" % (pr, lo, hi, pw))
np.savez(env.LOCAL + "/audit4_1_ecdenoise.npz", row_id=lab.row_id.values, **{"%s__%s" % k: v for k, v in out.items()})
