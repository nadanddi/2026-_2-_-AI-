# -*- coding: utf-8 -*-
"""EC TabPFN member with causal day-so-far indoor means.  EC is a day-level
quantity (95% of variance between days); the fingerprint gives indoor
sensors only at hour 0 (in_*_h0) and actuators as expanding means.  Add the
expanding mean of in_temp / in_hum / in_co2 over hours 0..h of the row's own
day (current and earlier inputs only).

Both members use the same fresh samples (201..204 / 211..214).
Arm (fixed): EC blend 0.8*round-3 + 0.2*member_plus  vs  0.8*round-3 + 0.2*member (v2 view).
Validators A, B, DIAG10; round-3 seeds 7/8.
Rule: better in all 6 cells, DIAG10 p_worse < 0.025.

Run:  cd research && PYTHONPATH="" <python> -u ec_pfn_daymean_gpu.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
import torch

import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from screen_v6 import boot
from anal_ec_noncausal_tabpfn import tp

assert torch.cuda.is_available()
SERIES = {7: (201, 202, 203, 204), 8: (211, 212, 213, 214)}


def main():
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    lab = lab.sort_values(["farm", "day", "hour"])
    for v in ("in_temp", "in_hum", "in_co2"):
        lab[v + "_tdmean"] = lab.groupby(["farm", "day"])[v].transform(lambda s: s.expanding().mean())
    lab = lab.sort_index()
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    c_plus = c_et + ["in_temp_tdmean", "in_hum_tdmean", "in_co2_tdmean"]
    y = lab.sub_ec.values
    X, Xp = lab[c_et].values.astype(np.float32), lab[c_plus].values.astype(np.float32)
    ok = True
    for vset, fds in (("A", folds("A")), ("B", folds("B")), ("DIAG10", diag_folds(lab))):
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
                m0 = np.mean([tp(X[trm], y[trm], X[vam], s) for s in seeds], axis=0)
                m1 = np.mean([tp(Xp[trm], y[trm], Xp[vam], s) for s in seeds], axis=0)
                idx = np.where(vam)[0]
                row = {}
                for k, p in (("v2", 0.8 * raw + 0.2 * m0), ("plus", 0.8 * raw + 0.2 * m1)):
                    q = np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46)
                    pooled.setdefault(k, np.full(len(lab), np.nan))[idx] = q
                    row[k] = rmse(q, y[idx])
                row["m0"], row["m1"] = rmse(m0, y[idx]), rmse(m1, y[idx])
                per.append(row)
            if vset in ("A", "B"):
                a, b = np.mean([r["v2"] for r in per]), np.mean([r["plus"] for r in per])
                ok = ok and b < a
                print("%-6s seed %d | v2 %.4f plus %.4f (%+.2f%%) | members %.4f / %.4f"
                      % (vset, r3, a, b, 100 * (b / a - 1), np.mean([r["m0"] for r in per]),
                         np.mean([r["m1"] for r in per])), flush=True)
            else:
                g = ~np.isnan(pooled["v2"])
                a, b = rmse(pooled["v2"][g], y[g]), rmse(pooled["plus"][g], y[g])
                pw = boot(lab[g].reset_index(drop=True), "sub_ec", pooled["v2"][g], pooled["plus"][g])[3]
                ok = ok and b < a and pw < 0.025
                print("DIAG10 seed %d | v2 %.4f plus %.4f (%+.2f%%, p_worse %.4f)" % (r3, a, b, 100 * (b / a - 1), pw),
                      flush=True)
    print("EC daymean PRE-SET RULE VERDICT:", "ADOPT" if ok else "REJECT", flush=True)


if __name__ == "__main__":
    main()
