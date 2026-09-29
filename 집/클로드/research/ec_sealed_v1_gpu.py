# -*- coding: utf-8 -*-
"""EC on sealed days (catalog 6.52/6.53: sealed-day RMSE 0.40 vs 0.13 on
other days; sealed days are 15% of training days, 33% of test days).

Idea: a sealed-day SPECIALIST - TabPFN whose context holds only training rows
of sealed days - used only on rows that look sealed so far.  Codex's sealed
model (6.25) was blended into every row (+4.1% on the other days); here the
other rows keep the current candidate exactly.

Gate (causal, per row): within the row's own day, hours 0..h:
  share of act_vent == 0 > 0.85  and  mean act_circfan < 10
Specialist context: training-fold rows of sealed days (full-day definition of
catalog 6.11 - a property of TRAINING days only), up to 2000 rows, 4 samples.
No test statistic enters anything.

Arms (fixed): on gated rows
  G50 : 0.5*round-3 + 0.5*specialist      G20 : 0.8*round-3 + 0.2*specialist
  other rows: candidate v2 = 0.8*round-3 + 0.2*TabPFN bag (2000 x 4)
Reference: v2 on every row.  Then causal_shrink 0.5 and clip, as always.
Validators A, B (fold means) and DIAG10 (pooled; sealed-only also shown);
seeds: round-3 7/8 with sample seeds 1../11...
Rule (2026-09-27): better than v2 in every validator x seed, DIAG10
p_worse < 0.025/2 in both seeds.

Run:  cd research && PYTHONPATH="" <python> -u ec_sealed_v1_gpu.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np
import pandas as pd
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

assert torch.cuda.is_available()
ARMS = {"G50": 0.5, "G20": 0.2}


def gate(lab):
    d = lab[["farm", "day", "hour", "act_vent", "act_circfan"]].copy().sort_values(["farm", "day", "hour"])
    g = d.groupby(["farm", "day"])
    z = (d.act_vent == 0).astype(float).where(d.act_vent.notna())
    d["v0"] = z.groupby([d.farm, d.day]).transform(lambda s: s.expanding().mean())
    d["fan"] = g.act_circfan.transform(lambda s: s.expanding().mean())
    return ((d.v0 > 0.85) & (d.fan < 10)).reindex(lab.index).values


def main():
    tX, _, _ = common.load_raw()
    _, _, lab0 = load()
    fp = F4.fp_features()
    lab = lab0.merge(fp, on="row_id", how="left").reset_index(drop=True)
    f14 = [c for c in (list(USABLE) + ["day", "hr_sin", "hr_cos", "midnight"]) if c not in OUT_COLS]
    c_et = f14 + F4.names(fp)
    y = lab.sub_ec.values
    X = lab[c_et].values.astype(np.float32)
    s_tr = sealed_days(tX[tX.farm.isin(TARGET_FARMS)])
    is_s = np.array([bool(s_tr.get(k, False)) for k in zip(lab.farm, lab.day)])
    gt = gate(lab)
    print("gate: %.1f%% of rows | precision vs sealed days %.2f | recall %.2f"
          % (100 * gt.mean(), is_s[gt].mean(), gt[is_s].mean()), flush=True)
    ok = {a: True for a in ARMS}
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
                cm = trm & is_s
                spec = np.mean([tp(X[cm], y[cm], X[vam], s0 + 20 + i) for i in range(4)], axis=0)
                g = gt[vam]
                idx = np.where(vam)[0]
                cand = {"v2": 0.8 * raw + 0.2 * bag}
                for a, w in ARMS.items():
                    p = cand["v2"].copy()
                    p[g] = (1 - w) * raw[g] + w * spec[g]
                    cand[a] = p
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
                refs = rmse(pooled["v2"][gg & is_s], y[gg & is_s])
                txt = []
                for a in ARMS:
                    pr, lo, hi, pw = boot(lab[gg].reset_index(drop=True), "sub_ec", pooled["v2"][gg], pooled[a][gg])
                    s = rmse(pooled[a][gg], y[gg])
                    ss = rmse(pooled[a][gg & is_s], y[gg & is_s])
                    ok[a] = ok[a] and s < ref and pw < 0.025 / len(ARMS)
                    txt.append("%s %.4f (%+.2f%%, p_worse %.4f; sealed %.4f vs %.4f)"
                               % (a, s, 100 * (s / ref - 1), pw, ss, refs))
                print("DIAG10 seed %d | v2 %.4f | %s" % (r3, ref, " | ".join(txt)), flush=True)
    for a, v in ok.items():
        print("%s PRE-SET RULE VERDICT: %s" % (a, "ADOPT" if v else "REJECT"), flush=True)


if __name__ == "__main__":
    main()
