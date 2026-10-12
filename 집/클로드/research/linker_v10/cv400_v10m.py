# MIRROR CV for linker v10 (geometry identical to linker_v7/cv400_v7m_critic.py) + eval decisions (ap) + comparison with v7.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
OUT = os.path.join(env.LOCAL, "linker_v10"); os.makedirs(OUT, exist_ok=True)
TRG = train_groups10(); print("train groups", len(TRG), "| truth links whose predecessor is same-ID later-numbered:", int(sum(FARM[a] == FARM[b] and NUM[a] > NUM[b] for b, a, _, _ in TRG)))
out = []
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
            a0, p0 = pick10(r_, 0, lab, rem2, w)
            if a0 is None: continue
            out.append(dict(rec=rid(r_), prev=rid(a0), p=round(p0, 4), prev_is_labelled=bool(lab[a0]), xf=FARM[a0] != f, lat=FARM[a0] == f and NUM[a0] > NUM[r_],
                            anchor_st23=round(ST[a0, 23], 2) if lab[a0] else np.nan, anchor_ec23=round(EC[a0, 23], 3) if lab[a0] else np.nan, dT_air=round(IN_[r_, 0, 0] - IN_[a0, 23, 0], 2)))
O = pd.DataFrame(out); O.to_csv(os.path.join(OUT, "cv_all400_choice_v10m.csv"), index=False, encoding="utf-8-sig")
TP = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
V7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
for nm, D in (("v7m", V7), ("v10m", O)):
    D = D.copy(); D["truth"] = D.rec.map(TP); T = D[D.truth.notna()]; T2 = T[T.rec.str[4:7].astype(int) >= 179]
    c = D[(D.p >= .8) & D.prev_is_labelled.astype(bool) & (D.dT_air.abs() <= 1.5)]; ct = c[c.truth.notna()]
    print(f"{nm}: hit {(T.prev == T.truth).mean():.3f} | pass-1 hit {(T[T.rec.str[4:7].astype(int) < 179].prev == T[T.rec.str[4:7].astype(int) < 179].truth).mean():.3f} | pass-2 hit {(T2.prev == T2.truth).mean():.3f} (n {len(T2)}) | confident anchors {len(c)} acc {(ct.prev == ct.truth).mean():.3f} correct {(ct.prev == ct.truth).sum()} (pass-2 {int((c.rec.str[4:7].astype(int) >= 179).sum())})")
# eval decisions
w_all = fit_clogit([(X_, y_) for b_, a_, X_, y_ in TRG], l2=0.5, iters=300); np.save(os.path.join(OUT, "w_v10.npy"), w_all)
print("weight of lat:", round(float(w_all[FEATS10.index("lat")]), 2))
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig").set_index("row")
ev = []
for f in ["F13", "F47"]:
    for r_ in sorted(np.where((FARM == f) & ~TRAIN)[0], key=lambda i: NUM[i]):
        a0, p0 = pick10(r_, 0, TRAIN.copy(), set(), w_all)
        ev.append(dict(row=rid(r_), prev_h0=rid(a0), p_h0=round(p0, 3), prev_is_labelled=bool(TRAIN[a0]), xf=FARM[a0] != f, lat=FARM[a0] == f and NUM[a0] > NUM[r_],
                       anchor_ec23=round(EC[a0, 23], 3) if TRAIN[a0] else np.nan, anchor_st23=round(ST[a0, 23], 2) if TRAIN[a0] else np.nan, dT_air=round(IN_[r_, 0, 0] - IN_[a0, 23, 0], 2)))
E = pd.DataFrame(ev); E.to_csv(os.path.join(OUT, "eval60_predecessor_v10.csv"), index=False, encoding="utf-8-sig")
anc = lambda d, pc: set(d[(d[pc] >= .8) & d.prev_is_labelled.astype(bool) & (d.dT_air.abs() <= 1.5)].iloc[:, 0])
a7 = set(E7[(E7.p_h0 >= .8) & E7.prev_is_labelled.astype(bool) & (E7.dT_air.abs() <= 1.5)].index); a10 = anc(E, "p_h0")
print(f"\neval anchors v7 {len(a7)} | v10 {len(a10)} | only v7 {sorted(a7 - a10)} | only v10 {sorted(a10 - a7)}")
Ei = E.set_index("row"); chg = [(r, E7.at[r, "prev_h0"], Ei.at[r, "prev_h0"], E7.at[r, "p_h0"], Ei.at[r, "p_h0"]) for r in Ei.index if E7.at[r, "prev_h0"] != Ei.at[r, "prev_h0"]]
print("eval records whose chosen predecessor changed:", len(chg)); [print("  ", c) for c in chg]
