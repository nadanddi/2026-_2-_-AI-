# -*- coding: utf-8 -*-
"""H-C: do clean-row validators track the real leaderboard better than all-row ones?
(lab Claude, 2026-09-30; hypothesis from gc2_error_conditions_v1 / catalog 6b.12)

Background: 24% of labelled rows are flagged (restored rows + noisy days,
train_flags_v6 weight 0.2) and carry 56% of the G_C2 DIAG10 squared error, but
such days are rare in the test period.  If so, a validator scored on clean rows
only should predict real score changes better.

Real temperature scores (RMSE) of the submitted rounds:
  R1 0.7450  R2 0.6666  R3 0.5624  R4 0.5697  R5 0.5456  R6 0.5251
Transitions scored (new vs old): R2/R1, R3/R2, R4/R3, R5/R3, R6/R5.

Out-of-fold predictions (all saved earlier, nothing refitted):
  R1 R2 R3        audit1_calib_oof.npz  "DIAG|R*", "EXT10|R*", "EXT12|R*"
  R4              0.5 * R3 + 0.5 * HA (same file; round 4 = half cold-hinge blend)
  R5              temp_mask_v1_oof.npz FULL world, mean of seeds 7 / 101
  R6 (G_C2)       temp_mask_v1_oof.npz MASK base + Codex (seed pairs averaged)
                  + web_tabpfn_v2_temp_<V>.npy samples 1-8, gate on in_temp
  consistency     eval_v6 "plain" must match audit1 R3, eval_v6 "F60ND" must
                  match temp_mask FULL (same recipes) - printed, not used.
Views: ALL = every scored row; CLEAN = rows with train_flags_v6 weight 1.

PRE-SET DECISION RULE (fixed before running):
  For each validator V in {DIAG10, EXT10, EXT12} and view in {ALL, CLEAN}:
    pred ratio  q = RMSE_new / RMSE_old on the rows scored in V (and the view)
    log error   err = ln(q) - ln(q_real)
    MAE  = mean |err| over the 5 transitions
    SIGN = number of transitions with sign(q - 1) == sign(q_real - 1)
  H-C holds for V if MAE_CLEAN < MAE_ALL and SIGN_CLEAN >= SIGN_ALL.
  H-C is SUPPORTED if it holds for at least 2 of the 3 validators; otherwise
  NOT SUPPORTED.  5 transitions only: the verdict is descriptive, not a test.

Output: logs/hc_clean_validator_v1.log
Run:  PYTHONPATH="" <python> -u hc_clean_validator_v1.py   (from 연구실/클로드/code)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "집", "클로드", "research"))
import env  # noqa: E402,F401
import numpy as np  # noqa: E402
from harness import load  # noqa: E402
import train_flags_v6 as TF  # noqa: E402

REAL = {"R1": 0.7450, "R2": 0.6666, "R3": 0.5624, "R4": 0.5697, "R5": 0.5456, "R6": 0.5251}
TRANS = [("R2", "R1"), ("R3", "R2"), ("R4", "R3"), ("R5", "R3"), ("R6", "R5")]
VALS = {"DIAG10": "DIAG", "EXT10": "EXT10", "EXT12": "EXT12"}   # temp_mask name -> audit1 name


def rmse(p, y, m):
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def main():
    _, lab, _ = load()
    y = lab.sub_temp.values
    t = lab.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    clean = TF.row_weights(lab, 0.2, w_noisy=0.2) >= 1
    a1 = np.load(env.LOCAL + "/audit1_calib_oof.npz", allow_pickle=True)
    tm = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
    ev = np.load(env.LOCAL + "/eval_v6_oof.npz", allow_pickle=True)
    for f in (a1, tm, ev):
        assert (f["row_id"] == lab.row_id.values).all()
    print("clean rows %d / %d" % (clean.sum(), len(clean)))

    verdicts = []
    for V, av in VALS.items():
        P = {r: a1["%s|%s" % (av, r)] for r in ("R1", "R2", "R3")}
        P["R4"] = 0.5 * a1["%s|R3" % av] + 0.5 * a1["%s|HA" % av]
        P["R5"] = np.mean([tm["%s__FULL__7" % V], tm["%s__FULL__101" % V]], axis=0)
        base = np.mean([tm["%s__MASK__7" % V], tm["%s__MASK__101" % V]], axis=0)
        cx = np.mean([tm["%s__CODEX__726" % V], tm["%s__CODEX__727" % V]], axis=0)
        pfn = np.load(env.LOCAL + "/web_tabpfn_v2_temp_%s.npy" % V).mean(0)
        P["R6"] = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
        scored = np.all([np.isfinite(p) for p in P.values()], axis=0)
        print("=" * 90)
        print("%s  scored rows %d (clean %d)" % (V, scored.sum(), (scored & clean).sum()))
        # consistency checks (not used in the verdict)
        pl, f6 = ev["plain__%s" % V], ev["F60ND__%s" % V]
        m = scored & np.isfinite(pl)
        print("  check: RMSE audit1 R3 %.4f vs eval_v6 plain %.4f | temp_mask FULL %.4f vs eval_v6 F60ND %.4f"
              % (rmse(P["R3"], y, m), rmse(pl, y, m), rmse(P["R5"], y, m), rmse(f6, y, m)))
        print("  RMSE ALL  : " + "  ".join("%s %.4f" % (r, rmse(P[r], y, scored)) for r in P))
        print("  RMSE CLEAN: " + "  ".join("%s %.4f" % (r, rmse(P[r], y, scored & clean)) for r in P))
        res = {}
        for view, m in (("ALL", scored), ("CLEAN", scored & clean)):
            errs, sign = [], 0
            for new, old in TRANS:
                q = rmse(P[new], y, m) / rmse(P[old], y, m)
                qr = REAL[new] / REAL[old]
                errs.append(np.log(q) - np.log(qr))
                sign += int(np.sign(q - 1) == np.sign(qr - 1))
                print("  %-5s %s/%s  validator %.4f  real %.4f  log err %+.3f" % (view, new, old, q, qr, errs[-1]))
            res[view] = (float(np.mean(np.abs(errs))), sign)
            print("  %-5s MAE(log) %.4f  sign agreement %d/5" % (view, *res[view]))
        holds = res["CLEAN"][0] < res["ALL"][0] and res["CLEAN"][1] >= res["ALL"][1]
        verdicts.append(holds)
        print("  -> H-C holds for %s: %s" % (V, holds))
    print("=" * 90)
    print("H-C %s (holds on %d of 3 validators; rule: >= 2)"
          % ("SUPPORTED" if sum(verdicts) >= 2 else "NOT SUPPORTED", sum(verdicts)))


if __name__ == "__main__":
    main()
