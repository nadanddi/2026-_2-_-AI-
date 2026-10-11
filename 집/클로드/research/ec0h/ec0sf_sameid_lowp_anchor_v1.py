# -*- coding: utf-8 -*-
"""EC0-SF (2026-10-12 집 클로드; EXPLORATORY - condition chosen AFTER seeing EC0-S results (critic result note: same-farm links -4.8 vs other-farm +6.5); not a confirmatory test). EC0-S restricted to SAME-ID links (xf False).  Base design: EC0-S (critic-revised minimal design, ec0h/critic_plan_v2.md): use the linker's top candidate's 23 h EC
label when it is a LABELLED record but confidence is only p in [.5, .8) (currently discarded), shrunk by a p-bin coefficient.
Fixed before running:
  targets   : EV val records with NO EA2 anchor (anchored False) whose v7m choice is labelled with p in [.5, .65) or [.65, .8),
              prev not hidden in that EV fold (not 'val'/'del' of the same fold), |A1 - q0| <= .5 (gap cap).
  beta(bin) : LS slope of (y0 - q0) on (A1 - q0) over records of the OTHER EV folds with the same bin (DIAG10 OOF EC, column sg,
              seed mean, as q), clipped to [0, 1].
  spread    : c_h = carry of the 0 h residual measured on DIAG10 OOF pass-1 records (LS slope of e_h on e_0), constant.
  pred_h    = EA2_h + c_h * beta(bin) * (A1 - EA2_0)      (EA2 = stored EV final chain, BASE season, per seed)
  2-step agreement guard of the critic is NOT implemented (no EV-geometry 2-step choices available) -> stated as a limitation.
Judgement (critic): (A) EV pass-1 targets: final-chain RMSE better in all 3 seeds AND chunk-cluster bootstrap P(worse) < .0125;
  (B) EV pass-2 targets: declared 'not decidable' (few records) -> direction only.  Adopt only if A passes (date-run mimic C not run).
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
H = os.path.join(env.ROOT, u"집", u"클로드", "research")
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[R.arm == "BASE"]
F = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv"))
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; QD = O.groupby(["rec", "hour"]).sg.mean().unstack(); YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
# carry c_h from DIAG10 pass-1
p1 = [r for r in QD.index if int(r[4:7]) < 179]; e = (YD.loc[p1] - QD.loc[p1]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)])
print("carry c_h:", " ".join("%d:%.2f" % (h, v) for h, v in enumerate(c) if h % 3 == 0 or h == 23))
fold_of = F[F.role == "val"].set_index("rec").fold.to_dict()
hidden = {k: set(F[(F.fold == k)].rec) for k in F.fold.unique()}
binf = lambda p: 0 if .5 <= p < .65 else (1 if .65 <= p < .8 else -1)
# candidate records (any fold) with labelled prev and p in bins; A1 = anchor_ec23
C = L[L.prev_is_labelled.astype(bool) & L.anchor_ec23.notna() & ~L.xf.astype(bool)].copy(); C["bin"] = C.p.map(binf); C = C[C.bin >= 0]
C = C[C.index.isin(QD.index)]; C["gap"] = C.anchor_ec23 - QD.loc[C.index, 0].values; C["r0"] = YD.loc[C.index, 0].values - QD.loc[C.index, 0].values
C["fold"] = C.index.map(fold_of)
rows = []
for seed, G in R[R.seed.astype(str) != "avg"].groupby("seed"):
    G = G.sort_values(["rec", "hour"]).copy(); G["new"] = G.EA2
    for k in sorted(G.fold.unique()):
        oth = C[(C.fold != k)]
        beta = {b: float(np.clip((oth[oth.bin == b].gap * oth[oth.bin == b].r0).sum() / (oth[oth.bin == b].gap ** 2).sum(), 0, 1)) for b in (0, 1)}
        for rec in G[(G.fold == k) & (~G.anchored)].rec.unique():
            if rec not in C.index or C.at[rec, "fold"] != k: continue
            prev = L.at[rec, "prev"]
            if prev in hidden[k]: continue
            m = (G.rec == rec).values; ea0 = G.EA2.values[m][0]; gap = C.at[rec, "anchor_ec23"] - ea0
            if abs(C.at[rec, "anchor_ec23"] - QD.at[rec, 0]) > .5: continue
            G.loc[m, "new"] = G.EA2.values[m] + c[G.hour.values[m]] * beta[C.at[rec, "bin"]] * gap
            rows.append(dict(seed=seed, rec=rec, fold=k, bin=C.at[rec, "bin"], beta=beta[C.at[rec, "bin"]], gap=gap))
    R.loc[G.index, "new"] = G.new
T = pd.DataFrame(rows); print("targets per seed:", T.groupby("seed").rec.nunique().to_dict(), "| pass-2:", T[T.rec.str[4:7].astype(int) >= 179].rec.unique().tolist())
print("beta by bin (mean over folds):", T.groupby("bin").beta.mean().round(3).to_dict())
rng = np.random.default_rng(20261217); chunk = F[F.role == "val"].set_index("rec").chunk.to_dict()
for part, cond in (("pass-1 targets", lambda r: int(r[4:7]) < 179), ("pass-2 targets", lambda r: int(r[4:7]) >= 179)):
    recs = [r for r in T.rec.unique() if cond(r)]
    if not recs: print(part, "none"); continue
    cells = []; per = []
    for seed, G in R[R.seed.astype(str) != "avg"].groupby("seed"):
        m = G.rec.isin(recs); cells.append(rm((G.new - G.y)[m]) / rm((G.EA2 - G.y)[m]) - 1)
        per.append(((G.new - G.y) ** 2 - (G.EA2 - G.y) ** 2)[m].groupby(G.rec[m]).sum())
    P = pd.concat(per, axis=1).mean(1); cl = pd.Series({r: chunk.get(r, r) for r in P.index}); S = P.groupby(cl).sum()
    idx = rng.integers(0, len(S), (20000, len(S))); pw = float((S.values[idx].sum(1) >= 0).mean())
    print("%s: records %d chunks %d | cells %s | improved %d/%d | cluster P(worse) %.4f" % (part, len(recs), len(S), " ".join("%+.2f%%" % (100 * x) for x in cells), int((P < 0).sum()), len(P), pw))
    if part.startswith("pass-1"): print("VERDICT EC0-S (A):", "PASS" if all(x < 0 for x in cells) and pw < .0125 else "FAIL")
