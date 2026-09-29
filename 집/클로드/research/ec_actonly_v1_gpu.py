# -*- coding: utf-8 -*-
"""EC TabPFN member on noise-free inputs only.  Training sealed days are
mostly noisy in the indoor sensors (catalog 6.13) while test sealed days are
clean, and sealed days dominate the EC error (6.52).  A member that never
sees in_temp/in_hum/in_co2 (actuators + time + actuator fingerprints only)
cannot learn noise artefacts.

Arms (fixed):
  ACT : v2 with the TabPFN member replaced by the actuator-only bag
  ADD : 0.7*round-3 + 0.2*TabPFN(v2 features) + 0.1*actuator-only bag
Reference v2 = 0.8*round-3 + 0.2*TabPFN (2000 x 4).  Shrink + clip as always.
Validators A, B, DIAG10; round-3 seeds 7/8, samples 1../11...
Rule (2026-09-27): better than v2 in every validator x seed, DIAG10
p_worse < 0.025/2.  Also reported (info, since test sealed days are clean):
DIAG10 RMSE on clean days and on sealed days.

Run:  cd research && PYTHONPATH="" <python> -u ec_actonly_v1_gpu.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
import torch

import common
import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS, TARGET_FARMS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from screen_v6 import boot
from anal_ec_regime import sealed_days
from anal_ec_noncausal_tabpfn import tp
import train_flags_v6 as TF

assert torch.cuda.is_available()
INDOOR = ("in_temp", "in_hum", "in_co2")


def main():
    tX, _, _ = common.load_raw()
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    c_act = [c for c in c_et if not any(c.startswith(v) for v in INDOOR)]
    print("actuator-only features (%d): %s" % (len(c_act), c_act), flush=True)
    y = lab.sub_ec.values
    X = lab[c_et].values.astype(np.float32)
    Xa = lab[c_act].values.astype(np.float32)
    s_tr = sealed_days(tX[tX.farm.isin(TARGET_FARMS)])
    key = list(zip(lab.farm, lab.day))
    is_s = np.array([bool(s_tr.get(k, False)) for k in key])
    nd = TF.noisy_days()
    noisy = set(map(tuple, nd[nd.noisy][["farm", "day"]].values))
    clean = np.array([k not in noisy for k in key])
    ok = {"ACT": True, "ADD": True}
    for vset, fds in (("A", folds("A")), ("B", folds("B")), ("DIAG10", diag_folds(lab))):
        for r3, s0 in ((7, 1), (8, 11)):
            ec_v6.SEED = r3
            per, pooled = [], {}
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                tr, va = lab[trm], lab[vam].reset_index(drop=True)
                yt = tr.sub_ec.values
                raw = (0.6 * ec_v6.et().fit(tr[c_et], yt).predict(va[c_et])
                       + 0.3 * ec_v6.ltw().fit(tr[f14], yt).predict(va[f14])
                       + 0.1 * ec_v6.mlp().fit(tr[f14], yt).predict(va[f14]))
                bag = np.mean([tp(X[trm], y[trm], X[vam], s0 + i) for i in range(4)], axis=0)
                bga = np.mean([tp(Xa[trm], y[trm], Xa[vam], s0 + i) for i in range(4)], axis=0)
                idx = np.where(vam)[0]
                cand = {"v2": 0.8 * raw + 0.2 * bag, "ACT": 0.8 * raw + 0.2 * bga,
                        "ADD": 0.7 * raw + 0.2 * bag + 0.1 * bga}
                row = {}
                for k, p in cand.items():
                    q = np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46)
                    pooled.setdefault(k, np.full(len(lab), np.nan))[idx] = q
                    row[k] = rmse(q, y[idx])
                per.append(row)
            if vset in ("A", "B"):
                ref = np.mean([r["v2"] for r in per])
                txt = []
                for a in ok:
                    s = np.mean([r[a] for r in per])
                    ok[a] = ok[a] and s < ref
                    txt.append("%s %.4f (%+.2f%%)" % (a, s, 100 * (s / ref - 1)))
                print("%-6s seed %d | v2 %.4f | %s" % (vset, r3, ref, " | ".join(txt)), flush=True)
            else:
                gg = ~np.isnan(pooled["v2"])
                ref = rmse(pooled["v2"][gg], y[gg])
                txt = []
                for a in ok:
                    pr, lo, hi, pw = boot(lab[gg].reset_index(drop=True), "sub_ec", pooled["v2"][gg], pooled[a][gg])
                    s = rmse(pooled[a][gg], y[gg])
                    ok[a] = ok[a] and s < ref and pw < 0.025 / 2
                    txt.append("%s %.4f (%+.2f%%, p_worse %.4f; clean %.4f vs %.4f; sealed %.4f vs %.4f)"
                               % (a, s, 100 * (s / ref - 1), pw,
                                  rmse(pooled[a][gg & clean], y[gg & clean]), rmse(pooled["v2"][gg & clean], y[gg & clean]),
                                  rmse(pooled[a][gg & is_s], y[gg & is_s]), rmse(pooled["v2"][gg & is_s], y[gg & is_s])))
                print("DIAG10 seed %d | v2 %.4f | %s" % (r3, ref, " | ".join(txt)), flush=True)
    for a, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (a, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
