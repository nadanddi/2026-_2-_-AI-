# MIRROR CV for linker v9 (geometry identical to linker_v7/cv400_v7m_critic.py; only the added fp feature differs). Decision hour 0.
# Stage-2a gate (fixed before running, descriptive): compare with v7m on the same records — overall hit, pass-2 hit, confident
# (p >= .8) 1-step labelled anchors: count and accuracy (truth known).  Go to downstream EV/TA scoring only if confident-correct
# anchors increase by >= 10 % with accuracy not lower than v7m.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
OUT = os.path.join(env.LOCAL, "linker_v9"); os.makedirs(OUT, exist_ok=True)
allv = np.ones(N, bool); COMP = components(allv)
TRG = train_groups9(0); out = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w = fit_clogit([(X_, y_) for b_, a_, X_, y_ in TRG if not touch[b_] and not touch[a_]], l2=0.5, iters=300)
        lab = TRAIN.copy(); lab[touch] = False
        for r_ in blk:
            if not TRAIN[r_]: continue
            a0, p0 = pick9(r_, 0, lab, rem2, w)
            if a0 is None: continue
            out.append(dict(rec=rid(r_), prev=rid(a0), p=round(p0, 4), prev_is_labelled=bool(lab[a0]), xf=FARM[a0] != f, steps_to_anchor=1 if lab[a0] else -1,
                            anchor_st23=round(ST[a0, 23], 2) if lab[a0] else np.nan, anchor_ec23=round(EC[a0, 23], 3) if lab[a0] else np.nan, dT_air=round(IN_[r_, 0, 0] - IN_[a0, 23, 0], 2)))
O = pd.DataFrame(out); O.to_csv(os.path.join(OUT, "cv_all400_choice_v9m.csv"), index=False, encoding="utf-8-sig")
w_all = fit_clogit([(X_, y_) for b_, a_, X_, y_ in TRG], l2=0.5, iters=300); print("weights:", dict(zip(FEATS9, np.round(w_all, 2))))
TLa = pd.read_pickle(S3 + r"\truth.pkl"); tm = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}
V7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
for nm, D in (("v7m", V7), ("v9m", O)):
    D = D.copy(); D["truth"] = D.rec.map(tm); T = D[D.truth.notna()]; T2 = T[T.rec.str[4:7].astype(int) >= 179]
    c = D[(D.p >= .8) & D.prev_is_labelled.astype(bool)]; ct = c[c.truth.notna()]
    print(f"{nm}: hit {(T.prev == T.truth).mean():.3f} (n {len(T)}) | pass-2 hit {(T2.prev == T2.truth).mean():.3f} (n {len(T2)}) | confident 1-step labelled {len(c)}, truth-known {len(ct)}, acc {(ct.prev == ct.truth).mean():.3f}, correct {(ct.prev == ct.truth).sum()} | pass-2 confident {int((c.rec.str[4:7].astype(int) >= 179).sum())}")
