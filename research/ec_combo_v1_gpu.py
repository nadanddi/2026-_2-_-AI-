# -*- coding: utf-8 -*-
"""EC combination, one pre-set arm on FRESH seeds.  Two EC signals were
consistent in direction but short of significance:
  E40_8 (weight 0.4, 8-sample bag): 12/12 cells better (6.64, 6.71)
  day-so-far indoor means in the TabPFN view: 6/6 cells better (6.73)
The combination was chosen after seeing them (post-hoc), so it is judged on
never-used seeds only: samples 301..308 / 311..318, round-3 seeds 11 / 12.

COMBO = 0.6*round-3 + 0.4*mean(8 samples, TabPFN on c_et + 3 day-so-far means)
v2    = 0.8*round-3 + 0.2*mean(first 4 samples, TabPFN on c_et)
Shrink 0.5 + clip.  Validators A, B, DIAG10 (rule); EC EXT10 / EXT12 printed
for the cold-extrapolation caution (6.68), not part of the rule.
Rule (fixed): COMBO better than v2 in all 6 rule cells, DIAG10 p_worse < 0.025.

Run:  cd research && PYTHONPATH="" <python> -u ec_combo_v1_gpu.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
import torch

import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from screen_v6 import boot
from anal_ec_noncausal_tabpfn import tp

assert torch.cuda.is_available()
SERIES = {11: tuple(range(301, 309)), 12: tuple(range(311, 319))}


def main():
    _, _, lab0 = load()
    fp = F4.fp_features()
    ph = F4.phys_features()
    lab = lab0.merge(fp, on="row_id", how="left").merge(ph[["row_id", "ph_in_temp_3"]], on="row_id", how="left")
    lab = lab.reset_index(drop=True).sort_values(["farm", "day", "hour"])
    for v in ("in_temp", "in_hum", "in_co2"):
        lab[v + "_tdmean"] = lab.groupby(["farm", "day"])[v].transform(lambda s: s.expanding().mean())
    lab = lab.sort_index()
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    c_plus = c_et + ["in_temp_tdmean", "in_hum_tdmean", "in_co2_tdmean"]
    y = lab.sub_ec.values
    X, Xp = lab[c_et].values.astype(np.float32), lab[c_plus].values.astype(np.float32)
    dmin = lab.groupby(["farm", "day"]).ph_in_temp_3.min()
    ext = []
    for th in (10, 12):
        cd = dmin[dmin < float(th)]
        ext.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    ok = True
    for vset, fds in [("A", folds("A")), ("B", folds("B")), ("DIAG10", diag_folds(lab))] + ext:
        for r3, seeds in SERIES.items():
            ec_v6.SEED = r3
            per, pooled = [], {}
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                tr, va = lab[trm], lab[vam].reset_index(drop=True)
                yt = tr.sub_ec.values
                raw = (0.6 * ec_v6.et().fit(tr[c_et], yt).predict(va[c_et])
                       + 0.3 * ec_v6.ltw().fit(tr[f14], yt).predict(va[f14])
                       + 0.1 * ec_v6.mlp().fit(tr[f14], yt).predict(va[f14]))
                m4 = np.mean([tp(X[trm], y[trm], X[vam], s) for s in seeds[:4]], axis=0)
                m8 = np.mean([tp(Xp[trm], y[trm], Xp[vam], s) for s in seeds], axis=0)
                idx = np.where(vam)[0]
                row = {}
                for k, p in (("v2", 0.8 * raw + 0.2 * m4), ("combo", 0.6 * raw + 0.4 * m8)):
                    q = np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46)
                    pooled.setdefault(k, np.full(len(lab), np.nan))[idx] = q
                    row[k] = rmse(q, y[idx])
                per.append(row)
            if vset in ("A", "B"):
                a, b = np.mean([r["v2"] for r in per]), np.mean([r["combo"] for r in per])
                ok = ok and b < a
                print("%-6s seed %d | v2 %.4f combo %.4f (%+.2f%%)" % (vset, r3, a, b, 100 * (b / a - 1)), flush=True)
            else:
                g = ~np.isnan(pooled["v2"])
                a, b = rmse(pooled["v2"][g], y[g]), rmse(pooled["combo"][g], y[g])
                if vset == "DIAG10":
                    pw = boot(lab[g].reset_index(drop=True), "sub_ec", pooled["v2"][g], pooled["combo"][g])[3]
                    ok = ok and b < a and pw < 0.025
                    print("DIAG10 seed %d | v2 %.4f combo %.4f (%+.2f%%, p_worse %.4f)"
                          % (r3, a, b, 100 * (b / a - 1), pw), flush=True)
                else:
                    print("%s (info) seed %d | v2 %.4f combo %.4f (%+.2f%%)" % (vset, r3, a, b, 100 * (b / a - 1)),
                          flush=True)
        if vset == "DIAG10":
            print("EC COMBO PRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    main()
