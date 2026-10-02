# -*- coding: utf-8 -*-
"""EC stage-2 EL1: eval-like validation of the season index (fixed before running;
2026-10-02 집 클로드; statistical critic's priority 3, after R5/R5b).

Why: R5 replays round 5 (ET without `day`) on today's DIAG10 -> late RMSE
.3775 -> .3163, yet the real leaderboard got worse (+3.8%).  R5b: round 3
predicted the test's calendar-early days at 0.443 on average, i.e. the large
DIAG10 over-prediction (~0.85 vs truth ~0.39) is not visible on the test.
Hypothesis: in the test every block has labelled days 2 record days away on
both sides, so `day` interpolates local late labels; in DIAG10 the late
validation days lose their neighbours (contiguous fold blocks + lock purge),
so `day` points to the pass-1 high-EC epoch.  The `day` bias may be largely a
validation artefact.

Eval-like validator: per greenhouse, the non-locked labelled late days
(record day >= 179) in record order are cut into consecutive blocks of 5
labelled days; each block is held out alone (one fold), purging only +-1
record day around it (and the lock +-1), so labelled late neighbours 2 days
away stay in training, as in the test.
Models (Codex phase-3 R3 recipe, seeds 7/101/2024): R3 with `day` (as
3rd-round / v2's R3 part) vs R3S with the DC4 season index (DC5 recipe).
Report pooled RMSE on the held-out late days, per seed, and by calendar
(<70 / >=70).
Decision for this diagnostic: "season advantage survives eval-like geometry"
if R3S beats R3 for all 3 seeds AND by >= 5% pooled; "artefact" if R3S is
not better on >= 2 of 3 seeds; otherwise "small".
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_EL1_evallike_late_blocks_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc5)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
SEEDS = (7, 101, 2024)


def main():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    cal = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv"))[["farm", "day", "cal"]]
    lab = lab.merge(cal, on=["farm", "day"], how="left")
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    BS = [c for c in core.BASE if c != "day"] + ["season"]
    folds = []
    for f in ("F13", "F47"):
        days = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(days), 5):
            folds.append({(f, int(d)) for d in days[k:k + 5]})
    print("eval-like folds:", len(folds), [sorted(x)[0] for x in folds])
    rec = []
    for i, vd in enumerate(folds):
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates()
        vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
        vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec", "cal"]].copy()
        frame["fold"] = i
        lab_late = tr[(tr.farm == vdays.farm[0]) & (tr.day >= 179)].day.unique()
        frame["gap"] = [min(abs(lab_late - d)) for d in frame.day]
        for s in SEEDS:
            frame["r3_%d" % s] = dc5.r3(tr, va, s, list(core.FULL), list(core.BASE))
            frame["r3s_%d" % s] = dc5.r3(tr, va, s, FS, BS)
        rec.append(frame)
        print("fold %d done (%s, nearest training late label %d-%d days)" % (i, sorted(vd)[0], frame.gap.min(), frame.gap.max()), flush=True)
    O = pd.concat(rec, ignore_index=True)
    O.to_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    wins, gains = 0, []
    print("\neval-like late hold-out (%d days)" % O[["farm", "day"]].drop_duplicates().shape[0])
    for s in SEEDS:
        a, b = r(O["r3_%d" % s] - O.sub_ec), r(O["r3s_%d" % s] - O.sub_ec)
        wins += b < a; gains.append(b / a - 1)
        print("  seed %d: R3 %.4f  R3S %.4f (%+.1f%%) | cal<70 %.4f -> %.4f | cal>=70 %.4f -> %.4f" % (
            s, a, b, 100 * (b / a - 1),
            r((O["r3_%d" % s] - O.sub_ec)[O.cal < 70]), r((O["r3s_%d" % s] - O.sub_ec)[O.cal < 70]),
            r((O["r3_%d" % s] - O.sub_ec)[O.cal >= 70]), r((O["r3s_%d" % s] - O.sub_ec)[O.cal >= 70])))
    res = O.groupby(["farm", "day"]).apply(lambda g: pd.Series({"cal": g.cal.iloc[0], "bias_r3": (g.r3_7 - g.sub_ec).mean(),
                                                                 "bias_r3s": (g.r3s_7 - g.sub_ec).mean()}), include_groups=False)
    print("  day-level bias (seed 7): R3 cal<70 %+.3f / cal>=70 %+.3f | R3S %+.3f / %+.3f" % (
        res.bias_r3[res.cal < 70].mean(), res.bias_r3[res.cal >= 70].mean(), res.bias_r3s[res.cal < 70].mean(), res.bias_r3s[res.cal >= 70].mean()))
    verdict = "survives" if wins == 3 and max(gains) <= -0.05 else ("artefact" if wins <= 1 else "small")
    print("\nEL1 verdict:", verdict)


if __name__ == "__main__":
    main()
