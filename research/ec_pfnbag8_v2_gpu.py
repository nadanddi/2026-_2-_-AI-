# -*- coding: utf-8 -*-
"""Confirmation of E40_8 (ec_pfnbag8_v1_gpu.py: better in all 6 cells, but
DIAG10 seed 7 p_worse 0.018 > 0.0125 Bonferroni -> rejected there; v4 E3
with 4 samples was also better in all 6 cells).  One arm only, fresh
everything: samples 71..78 / 81..88, round-3 seeds 9 / 10.

Reference v2 = 0.8*round-3 + 0.2*mean(first 4 samples of the series).
Arm E40_8 = 0.6*round-3 + 0.4*mean(8).  Shrink 0.5 + clip.
Rule (fixed): better than v2 in all 6 cells (A, B, DIAG10 x 2 seeds),
DIAG10 p_worse < 0.025 in both seeds.

Run:  cd research && PYTHONPATH="" <python> -u ec_pfnbag8_v2_gpu.py
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
SERIES = {9: tuple(range(71, 79)), 10: tuple(range(81, 89))}
ARMS = {"E40_8": 0.4}


def main():
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    y = lab.sub_ec.values
    X = lab[c_et].values.astype(np.float32)
    ok = {a: True for a in ARMS}
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
                P = np.vstack([tp(X[trm], y[trm], X[vam], s) for s in seeds])
                m4, m8 = P[:4].mean(0), P.mean(0)
                cand = {"v2": 0.8 * raw + 0.2 * m4}
                for a, w in ARMS.items():
                    cand[a] = (1 - w) * raw + w * m8
                idx = np.where(vam)[0]
                row = {}
                for k, p in cand.items():
                    q = np.clip(causal_shrink(p, va, 0.5), 0.062, 3.46)
                    pooled.setdefault(k, np.full(len(lab), np.nan))[idx] = q
                    row[k] = rmse(q, y[idx])
                per.append(row)
            if vset in ("A", "B"):
                ref = np.mean([r["v2"] for r in per])
                txt = []
                for a in ARMS:
                    s = np.mean([r[a] for r in per])
                    ok[a] = ok[a] and s < ref
                    txt.append("%s %.4f (%+.2f%%)" % (a, s, 100 * (s / ref - 1)))
                print("%-6s seed %d | v2 %.4f | %s" % (vset, r3, ref, " | ".join(txt)), flush=True)
            else:
                gg = ~np.isnan(pooled["v2"])
                ref = rmse(pooled["v2"][gg], y[gg])
                txt = []
                for a in ARMS:
                    pw = boot(lab[gg].reset_index(drop=True), "sub_ec", pooled["v2"][gg], pooled[a][gg])[3]
                    s = rmse(pooled[a][gg], y[gg])
                    ok[a] = ok[a] and s < ref and pw < 0.025 / len(ARMS)
                    txt.append("%s %.4f (%+.2f%%, p_worse %.4f)" % (a, s, 100 * (s / ref - 1), pw))
                print("DIAG10 seed %d | v2 %.4f | %s" % (r3, ref, " | ".join(txt)), flush=True)
    for a, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (a, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
