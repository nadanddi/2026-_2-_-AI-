# -*- coding: utf-8 -*-
# submission_28 draft FIELD LINKER with a 'no previous day' option 'FLN' (2026-10-15 집 클로드; research fln1_field_linker_none_option_v1.py, catalog 6.932).
# Same candidates/features as ap_fl.py (oj, v0-7, excl, xf; no record-number or date-order features) + one 'none' row per decision:
#   none-row features: isnone=1, min_oj (smallest candidate outdoor cost), min_vs (smallest candidate indoor/actuator distance), log #candidates.
# Weights: conditional logit on TRAINING truth pairs (train_groups7, target's same-farm +-1 hidden), each pair used twice:
#   (n) as is -> label = true previous day;  (d) true previous day removed from the candidates -> label = none   (field-style 'missing log' simulation).
# Modes (each with its own LINKER_WORK, as ap_fl.py):
#   train <out_dir>             : MASK world -> w_fln.npy + fln_train_links.csv (rec, prev0, p0, p_none) for every training record (same-farm +-1 hidden).
#   eval  <out_dir> <w_fln.npy> : real data -> fln_eval_links.csv (rec, prev0, p0, p_none, prev_is_labelled, xf, anchor_ec23, anchor_st23, dT_air);
#                                 each evaluation record decides independently at hour 0 (no later inputs, no other-farm evaluation inputs, no carried decisions).
import sys, os
MODE = sys.argv[1]; OUTD = sys.argv[2]; os.makedirs(OUTD, exist_ok=True)
from feat7 import *
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]; NX = ["isnone", "min_oj", "min_vs", "logn"]; COLS = FL + NX
def mat(df):
    C = np.zeros((len(df) + 1, len(COLS))); C[:-1, :len(FL)] = df[FL].values.astype(float)
    vs = df[[f"v{j}" for j in range(8)]].values.astype(float).sum(1); C[-1, len(FL):] = [1.0, df.oj.min(), vs.min(), np.log(len(df))]
    return C
def pick(b, lab, removed, w):
    df, _ = features7(b, 0, lab, removed=removed)
    if len(df) == 0: return None, 0.0, 1.0
    s = mat(df) @ w; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p[:-1])); return int(df.a.values[k]), float(p[k]), float(p[-1])
if MODE == "train":
    G = []
    for b, a in zip(TL7.b, TL7.a):
        lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
        for m in ("n", "d"):
            df, _ = features7(b, 0, lab, removed=set(TEST_IDX) | ({int(a)} if m == "d" else set()))
            if len(df) == 0: continue
            if m == "n":
                y = np.where(df.a.values == a)[0]
                if not len(y): continue
                y = int(y[0])
            else: y = len(df)
            G.append((mat(df), y))
    w = fit_clogit(G, l2=0.5, iters=300); np.save(os.path.join(OUTD, "w_fln.npy"), w); out = []
    for b in range(N):
        if not TRAIN[b] or FARM[b] not in ("F13", "F47"): continue
        lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
        a0, p0, pn = pick(b, lab, set(TEST_IDX), w)
        if a0 is not None: out.append(dict(rec=rid(b), prev0=rid(a0), p0=round(p0, 4), p_none=round(pn, 4)))
    pd.DataFrame(out).to_csv(os.path.join(OUTD, "fln_train_links.csv"), index=False)
    print("FLN train: groups", len(G), "| links", len(out), "| p0>=.5", sum(o["p0"] >= .5 for o in out))
else:
    w = np.load(sys.argv[3]); out = []
    for f in ["F13", "F47"]:
        for r_ in sorted(np.where((FARM == f) & ~TRAIN)[0], key=lambda i: NUM[i]):
            a0, p0, pn = pick(r_, TRAIN.copy(), set(), w)
            if a0 is None: continue
            out.append(dict(rec=rid(r_), prev0=rid(a0), p0=round(p0, 4), p_none=round(pn, 4), prev_is_labelled=bool(TRAIN[a0]), xf=FARM[a0] != f,
                            anchor_ec23=round(EC[a0, 23], 3) if TRAIN[a0] else np.nan, anchor_st23=round(ST[a0, 23], 2) if TRAIN[a0] else np.nan,
                            dT_air=round(IN_[r_, 0, 0] - IN_[a0, 23, 0], 2)))
    O = pd.DataFrame(out); O.to_csv(os.path.join(OUTD, "fln_eval_links.csv"), index=False)
    print("FLN eval: links", len(O), "| p0>=.5", int((O.p0 >= .5).sum()), "| labelled prev & p0>=.5", int(((O.p0 >= .5) & O.prev_is_labelled).sum()))
