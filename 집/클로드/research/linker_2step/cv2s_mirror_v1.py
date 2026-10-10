# 2-step linker, MIRROR CV (same geometry as critic/cvm.py, linker v7m): hide 10-record chunk in ID f + deleted neighbours
# (s-1, s+10) + other ID's mirror chunk (m-1..m+10, m = s-2 for F13, s+2 for F47). Weights refit without groups touching the
# hidden set; decisions at hour 0 for every TRAINING record of the chunk, each record independently.
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_2step")
from feat2s import *
import time
t0 = time.time()
TRG = train_groups2(0); print("training groups (MASK, h0): %d of %d truth 2-step pairs" % (len(TRG), len(TRUTH2)), flush=True)
w_all = fit_clogit([(X_, y_) for _, _, _, X_, y_ in TRG], l2=0.5, iters=300); print("w_all", dict(zip(FEATS2, np.round(w_all, 3))))
out = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh
        oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist())
        rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w = fit_clogit([(X_, y_) for b_, a1_, a2_, X_, y_ in TRG if not (touch[b_] or touch[a1_] or touch[a2_])], l2=0.5, iters=300)
        lab = TRAIN.copy(); lab[touch] = False
        for r_ in blk:
            if not TRAIN[r_]: continue
            a, p, nc = pick2(r_, 0, lab, rem2, w)
            t2 = TRUTH2.get(r_)
            out.append(dict(rec=rid(r_), pick=rid(a) if a is not None else "", p=round(p, 4), ncand=nc,
                            xf=(FARM[a] != f) if a is not None else False,
                            anchor_ec23=round(EC[a, 23], 3) if a is not None else np.nan,
                            truth_a2=rid(t2[0]) if t2 else "", truth_a1=rid(t2[1]) if t2 else "",
                            a1_labelled_in_fold=bool(lab[t2[1]]) if t2 else np.nan,
                            a2_available=bool(lab[t2[0]] and not (t2[0] in rem2)) if t2 else np.nan,
                            hit=(a == t2[0]) if (t2 and a is not None) else np.nan))
    print(f, "done %.0fs" % (time.time() - t0), flush=True)
O = pd.DataFrame(out); O.to_csv(S2S + r"\cv_mirror_choice_2step_v1.csv", index=False, encoding="utf-8-sig")
print("rows", len(O), "with pick", int((O.pick != "").sum()))
