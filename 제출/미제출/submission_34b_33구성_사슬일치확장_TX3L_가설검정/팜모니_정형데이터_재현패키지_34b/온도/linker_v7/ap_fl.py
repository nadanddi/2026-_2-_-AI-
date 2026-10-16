# -*- coding: utf-8 -*-
# submission_27 FIELD LINKER 'FL' (2026-10-15 집 클로드; research fl1/fl2/fl3_*.py).
# Same candidates and conditional-logit form as linker v7, but ONLY features that exist for any farm's mixed sensor logs:
#   oj   = outdoor (temp, hum, rad, wind) continuity from the candidate's 23 h to the target's 0 h
#   v0-7 = indoor / actuator continuity 23 h -> 0 h
#   excl = candidate already has a confirmed next day in the TRAINING data
#   xf   = candidate belongs to the other farm ID
# REMOVED vs v7: record-number differences (d1, d2w, d2c, d3, ldn) and the date-order features (isprev, dsdev, dsna) that come from
# 'record number = date' (pass 1) and 'pass 2 re-uses pass-1 outdoor weather' (pass 2).  Same-date grouping by identical outdoor values is
# kept only to exclude same-day candidates (a shared farm weather station gives identical values on the same date).
# Weights: fitted on TRAINING truth pairs (label continuity, train_groups7: target's same-farm +-1 hidden) - no evaluation data,
# no imitation of the evaluation layout.
# Modes (run each with its own LINKER_WORK):
#   train <out_dir> : LINKER_WORK built from data whose test_X inputs are ALL NaN (MASK world) -> fits weights (w_fl.npy) and
#                     writes fl_train_links.csv: for every training record, candidates = training records only, its same-farm +-1 hidden.
#   eval  <out_dir> <w_fl.npy> : LINKER_WORK built from the real data -> each evaluation record decides INDEPENDENTLY at hour 0
#                     (own hour-0 inputs, training records, same-farm earlier-day records; no other-farm evaluation inputs, no later inputs,
#                     no decisions carried between evaluation records) -> fl_eval_links.csv and fl_eval_anchor.csv
#                     (anchor rows: one step, labelled previous training record, p >= .8, |dT_air| <= 1.5; v7 anchor-file columns).
import sys, os
MODE = sys.argv[1]; OUTD = sys.argv[2]; os.makedirs(OUTD, exist_ok=True)
from feat7 import *
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]
def pick(b, lab, removed, w):
    df, _ = features7(b, 0, lab, removed=removed)
    if len(df) == 0: return None, 0.0
    s = df[FL].values.astype(float) @ w; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p)); return int(df.a.values[k]), float(p[k])
if MODE == "train":
    IFL = [FEATS7.index(c) for c in FL]; TRG = train_groups7()
    w = fit_clogit([(X_[:, IFL], y_) for b_, a_, X_, y_ in TRG], l2=0.5, iters=300); np.save(os.path.join(OUTD, "w_fl.npy"), w)
    out = []
    for b in range(N):
        if not TRAIN[b] or FARM[b] not in ("F13", "F47"): continue
        lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
        a0, p0 = pick(b, lab, set(TEST_IDX), w)
        if a0 is not None: out.append(dict(rec=rid(b), prev0=rid(a0), p0=round(p0, 4)))
    pd.DataFrame(out).to_csv(os.path.join(OUTD, "fl_train_links.csv"), index=False)
    print("FL train: weights", dict(zip(FL, np.round(w, 3))), "| links", len(out), "| p0>=.5", sum(o["p0"] >= .5 for o in out))
else:
    w = np.load(sys.argv[3]); out = []
    for f in ["F13", "F47"]:
        for r_ in sorted(np.where((FARM == f) & ~TRAIN)[0], key=lambda i: NUM[i]):
            a0, p0 = pick(r_, TRAIN.copy(), set(), w)
            if a0 is None: continue
            out.append(dict(row=rid(r_), rec=rid(r_), prev_h0=rid(a0), prev0=rid(a0), p_h0=round(p0, 3), p0=round(p0, 4), prev_is_labelled=bool(TRAIN[a0]), xf=FARM[a0] != f,
                            steps_to_anchor=1 if TRAIN[a0] else -1, anchor_ec23=round(EC[a0, 23], 3) if TRAIN[a0] else np.nan,
                            anchor_st23=round(ST[a0, 23], 2) if TRAIN[a0] else np.nan, dT_air=round(IN_[r_, 0, 0] - IN_[a0, 23, 0], 2)))
    O = pd.DataFrame(out); O.to_csv(os.path.join(OUTD, "fl_eval_links.csv"), index=False)
    C = O[(O.steps_to_anchor == 1) & O.prev_is_labelled & (O.p_h0 >= .8) & (O.dT_air.abs() <= 1.5)]
    C.to_csv(os.path.join(OUTD, "fl_eval_anchor.csv"), index=False, encoding="utf-8-sig")
    print("FL eval: links", len(O), "| p0>=.5", int((O.p0 >= .5).sum()), "| anchors", len(C), "| cross-farm anchors", int(C.xf.sum()))
