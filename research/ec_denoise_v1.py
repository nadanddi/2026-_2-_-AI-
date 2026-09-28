# -*- coding: utf-8 -*-
"""EC: does removing the injected indoor-sensor noise help?

Findings (anal_e3/e4, catalog 6.11-6.13): the injected noise sits in in_temp,
in_hum and in_co2 on ~25% of F13/F47 training days (outdoor inputs clean,
other greenhouses 9% rough, test 13%); 61% of the sealed / fan-off training
days - the regime that dominates the EC error - are rough, while the test's
sealed days are clean.  Down-weighting rough days hurt EC (it discards the
sealed days), so here the days are kept and the inputs are cleaned instead.

Variants (round-3 EC pipeline otherwise; seed 7):
  base     round 3
  Strain   TRAINING rows on rough days: in_temp / in_hum / in_co2 replaced by
           a causal EWM (halflife H h) of themselves; held-out rows raw, as
           the test will be.  (fp features untouched.)
  Sall     the three columns replaced by the causal EWM for every row
           (train, held-out, and later the test) - symmetric smoothing.
Rough days = training noise score top quarter (train_flags_v6, training
inputs only).  EWMs run per greenhouse along t over train + test inputs and
look backwards only.

Validators: geometry A (the EC validator whose direction matched all three
real EC changes so far; folds overlap, so each fold is scored on its own and
averaged) and the non-overlapping diagnostic folds.  Scores: all held-out
rows, held-out rows on clean days (test-like inputs), sealed days.

Run:  cd research && PYTHONPATH="" <python> -u ec_denoise_v1.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd

from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from anal_e3_spikes import day_table
from anal_e4_sealed_level import sealed
from ec_v6 import pipeline
import train_flags_v6 as TF

NOISY = ["in_temp", "in_hum", "in_co2"]
H = 1.5


def main():
    panel, _, lab0 = load()
    fp = F4.fp_features()
    fpc = F4.names(fp)
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]

    p = panel.sort_values(["farm", "t"])
    sm = {}
    for c in NOISY:
        sm[c] = p.groupby("farm")[c].transform(lambda s: s.ewm(halflife=H, ignore_na=True).mean())
    S = pd.DataFrame({c + "_s": sm[c].values for c in NOISY}, index=p.row_id.values)
    for c in NOISY:
        lab[c + "_s"] = S.loc[lab.row_id, c + "_s"].values

    nd = TF.noisy_days()
    rough_set = set(map(tuple, nd[nd.noise_score >= nd.noise_score.quantile(0.75)][["farm", "day"]].values))
    lab["rough"] = [(f, d) in rough_set for f, d in zip(lab.farm, lab.day)]
    dt = day_table(lab)
    sealed_set = set(map(tuple, dt[sealed(dt)][["farm", "day"]].values))
    lab["sealed"] = [(f, d) in sealed_set for f, d in zip(lab.farm, lab.day)]
    print("rough rows %d | sealed rows %d | sealed & clean rows %d"
          % (lab.rough.sum(), lab.sealed.sum(), (lab.sealed & ~lab.rough).sum()))

    fn = pipeline(f14 + fpc, f14)

    def v_base(tr, va):
        return fn(tr, va)

    def v_strain(tr, va):
        tr = tr.copy()
        m = tr.rough.values
        for c in NOISY:
            tr.loc[m, c] = tr.loc[m, c + "_s"]
        return fn(tr, va)

    def v_sall(tr, va):
        tr, va = tr.copy(), va.copy()
        for c in NOISY:
            tr[c] = tr[c + "_s"]
            va[c] = va[c + "_s"]
        return fn(tr, va)

    V = {"base": v_base, "Strain": v_strain, "Sall": v_sall}
    y = lab.sub_ec.values
    masks = {"all": np.ones(len(lab), bool), "clean": ~lab.rough.values, "sealed": lab.sealed.values,
             "sealed&clean": (lab.sealed & ~lab.rough).values}
    for vset, fds in (("A", folds("A")), ("DIAG10", diag_folds(lab))):
        per = {v: [] for v in V}
        pooled = {v: np.full(len(lab), np.nan) for v in V}
        for i, fd in enumerate(fds):
            trm, vam = split_mask(lab, fd)
            tr, va = lab[trm], lab[vam].reset_index(drop=True)
            idx = np.where(vam)[0]
            for v, f in V.items():
                pr = f(tr, va)
                pooled[v][idx] = pr
                per[v].append({k: rmse(pr[m[idx]], y[idx][m[idx]]) if m[idx].sum() else np.nan
                               for k, m in masks.items()})
            print("  %s fold %d done" % (vset, i), flush=True)
        print("\n== %s ==" % vset)
        for k in masks:
            line = "%-13s" % k
            b = np.array([r[k] for r in per["base"]])
            for v in V:
                a = np.array([r[k] for r in per[v]])
                if vset == "A":
                    line += " | %s %.4f" % (v, np.nanmean(a))
                    if v != "base":
                        line += " (%+.1f%%, better %d/%d)" % (100 * (np.nanmean(a) / np.nanmean(b) - 1),
                                                              int(np.nansum(a < b)), int(np.sum(~np.isnan(a))))
                else:
                    m = masks[k] & ~np.isnan(pooled[v])
                    line += " | %s %.4f" % (v, rmse(pooled[v][m], y[m]))
                    if v != "base":
                        line += " (%+.1f%%)" % (100 * (rmse(pooled[v][m], y[m]) / rmse(pooled["base"][m], y[m]) - 1))
            print(line)


if __name__ == "__main__":
    main()
