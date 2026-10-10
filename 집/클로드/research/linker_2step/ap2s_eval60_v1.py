# 2-step linker on the 60 evaluation records: hour-0 decision, each record independently (no decision of any other evaluation
# record is used). Weights = all MASK-world 2-step truth groups. Records with a confident 1-step anchor (linker v7) excluded.
# Output: confident (p >= .8) 2-step anchors + change vs 12th-round EC (submission_19) under pred_h = p_h + .5 (A - pm_h).
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_2step")
from feat2s import *
TRG = train_groups2(0); w = fit_clogit([(X_, y_) for _, _, _, X_, y_ in TRG], l2=0.5, iters=300); np.save(S2S + r"\w_2step_v1.npy", w)
print("w", dict(zip(FEATS2, np.round(w, 3))))
one = set(pd.read_csv(S7 + r"\eval60_confident_1step_anchor_v7.csv").row)
out = []
for f in ["F13", "F47"]:
    for r_ in sorted(np.where((FARM == f) & ~TRAIN)[0], key=lambda i: NUM[i]):
        a, p, nc = pick2(r_, 0, TRAIN.copy(), set(), w)
        out.append(dict(row=rid(r_), has_1step_anchor=rid(r_) in one, pick_2step=rid(a) if a is not None else "", p_h0=round(p, 4), ncand=nc,
                        xf=(FARM[a] != f) if a is not None else False, anchor_ec23=round(EC[a, 23], 3) if a is not None else np.nan))
O = pd.DataFrame(out); O.to_csv(S2S + r"\eval60_2step_all_v1.csv", index=False, encoding="utf-8-sig")
C = O[~O.has_1step_anchor & (O.p_h0 >= .8) & O.anchor_ec23.notna()].copy()
S = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); S["rec"] = S.row_id.str[:7]; S["h"] = S.row_id.str[-2:].astype(int)
S = S.sort_values(["rec", "h"]); S["pm"] = S.groupby("rec").sub_ec.transform(lambda s: s.expanding().mean())
S["A"] = S.rec.map(dict(zip(C.row, C.anchor_ec23))); S = S[S.A.notna()]; S["new"] = S.sub_ec + 0.5 * (S.A - S.pm)
g = S.groupby("rec").agg(sub19_mean=("sub_ec", "mean"), new_mean=("new", "mean")); g["dmean"] = g.new_mean - g.sub19_mean
C = C.merge(g, left_on="row", right_index=True, how="left")
C.to_csv(S2S + r"\eval60_confident_2step_anchor_v1.csv", index=False, encoding="utf-8-sig")
print("no-1-step records:", int((~O.has_1step_anchor).sum()), "with any 2-step pick:", int(((~O.has_1step_anchor) & (O.pick_2step != "")).sum()))
print(O[~O.has_1step_anchor].to_string(index=False))
print("CONFIDENT 2-step anchors:", len(C)); print(C.round(4).to_string(index=False))
if len(S): print("EC change vs submission_19 on %d rows: RMS %.4f, mean %.4f" % (len(S), np.sqrt(np.mean((S.new - S.sub_ec) ** 2)), (S.new - S.sub_ec).mean()))
