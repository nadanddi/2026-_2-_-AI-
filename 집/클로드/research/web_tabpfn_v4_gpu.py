# -*- coding: utf-8 -*-
"""EC TabPFN step 4.  v3 (catalog 6.46): a large context (8000) is WORSE than
several small ones averaged; raising the TabPFN weight helps on geometry A
but only ~1% on DIAG10 (CI incl. 0).  Three variants against the current
candidate v2 (context 2000 x 4 samples, weight 0.2):

  E1 : context 1000 x 8 samples, weight 0.2   (smaller + more samples)
  E2 : log target - TabPFN fitted on log(EC), exp() back, 2000 x 4, w 0.2
  E3 : 2000 x 4, weight 0.4   (POST-HOC: 0.4 was flat-best on v3 DIAG10;
       re-tested here on fresh samples and seeds, flagged as post-hoc)

Fresh seeds only: round-3 seeds 9 and 10, TabPFN sample seeds 21.. (never
used before).  Validators A, B (fold means), DIAG10 (pooled).
Rule (2026-09-27): a variant is adopted if it beats v2 in every validator x
both seed sets AND on DIAG10 p_worse < 0.025/3 in both seed sets (Bonferroni
for 3 variants; p_worse < 0.025 corresponds to the 95% CI excluding 0).

Run:  cd research && PYTHONPATH="" <python> -u web_tabpfn_v4_gpu.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

import ec_v6
from common import split_mask, rmse, USABLE, OUT_COLS
from harness import load, folds
import features_v4 as F4
from anal_q1_errors import diag_folds
from make_submission_v3 import causal_shrink
from screen_v6 import boot

assert torch.cuda.is_available()
K = 3
SETS = {1: (9, 21), 2: (10, 41)}          # round-3 seed, first TabPFN sample seed


def tp(Xtr, ytr, Xva, n_ctx, seed, log=False):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(n_ctx, len(Xtr)), replace=False)
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4,
                                                   random_state=seed, ignore_pretraining_limits=True,
                                                   inference_precision=torch.float32)
    yt = np.log(ytr[idx]) if log else ytr[idx]
    m.fit(Xtr[idx], yt)
    p = m.predict(Xva)
    return np.exp(p) if log else p


def members(Xtr, ytr, Xva, s0):
    v2 = np.mean([tp(Xtr, ytr, Xva, 2000, s0 + i) for i in range(4)], axis=0)
    e1 = np.mean([tp(Xtr, ytr, Xva, 1000, s0 + 10 + i) for i in range(8)], axis=0)
    e2 = np.mean([tp(Xtr, ytr, Xva, 2000, s0 + i, log=True) for i in range(4)], axis=0)
    return {"v2": (v2, 0.2), "E1": (e1, 0.2), "E2": (e2, 0.2), "E3": (v2, 0.4)}


def main():
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    y = lab.sub_ec.values
    X = lab[c_et].values.astype(np.float32)
    assert (y > 0).all()
    ok = {v: True for v in ("E1", "E2", "E3")}
    for vset, fds in (("A", folds("A")), ("B", folds("B")), ("DIAG10", diag_folds(lab))):
        for sid, (r3, s0) in SETS.items():
            ec_v6.SEED = r3
            per, pooled = [], {}
            for fd in fds:
                trm, vam = split_mask(lab, fd)
                tr, va = lab[trm], lab[vam].reset_index(drop=True)
                yt = tr.sub_ec.values
                raw = (0.6 * ec_v6.et().fit(tr[c_et], yt).predict(va[c_et])
                       + 0.3 * ec_v6.ltw().fit(tr[f14], yt).predict(va[f14])
                       + 0.1 * ec_v6.mlp().fit(tr[f14], yt).predict(va[f14]))
                M = members(X[trm], y[trm], X[vam], s0)
                idx = np.where(vam)[0]
                row = {}
                for k, (mem, wt) in M.items():
                    p = np.clip(causal_shrink((1 - wt) * raw + wt * mem, va, 0.5), 0.062, 3.46)
                    pooled.setdefault(k, np.full(len(lab), np.nan))[idx] = p
                    row[k] = rmse(p, y[idx])
                per.append(row)
            if vset in ("A", "B"):
                ref = np.mean([r["v2"] for r in per])
                txt = []
                for k in ok:
                    s = np.mean([r[k] for r in per])
                    ok[k] = ok[k] and s < ref
                    txt.append("%s %.4f (%+.2f%%)" % (k, s, 100 * (s / ref - 1)))
                print("%-6s set %d | v2 %.4f | %s" % (vset, sid, ref, " | ".join(txt)), flush=True)
            else:
                g = ~np.isnan(pooled["v2"])
                ref = rmse(pooled["v2"][g], y[g])
                txt = []
                for k in ok:
                    pr, lo, hi, pw = boot(lab[g].reset_index(drop=True), "sub_ec", pooled["v2"][g], pooled[k][g])
                    s = rmse(pooled[k][g], y[g])
                    ok[k] = ok[k] and s < ref and pw < 0.025 / K
                    txt.append("%s %.4f (%+.2f%%, p_worse %.4f)" % (k, s, 100 * (s / ref - 1), pw))
                print("%-6s set %d | v2 %.4f | %s" % (vset, sid, ref, " | ".join(txt)), flush=True)
    for k, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (k, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
