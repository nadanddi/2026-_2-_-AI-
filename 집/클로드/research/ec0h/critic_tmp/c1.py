import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
H = os.path.join(env.ROOT, u"집", u"클로드", "research")
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
F = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); V = F[F.role == "val"].set_index("rec")
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; QD = O.groupby(["rec", "hour"]).sg.mean().unstack(); YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
print("O cols", list(O.columns), "seeds", sorted(O.seed.unique()), "DIAG10 recs", QD.shape, "NaN", int(QD.isna().sum().sum()))
p1 = [r for r in QD.index if int(r[4:7]) < 179]; e = (YD.loc[p1] - QD.loc[p1]).values
c = np.array([(e[:, 0] * e[:, h]).sum() / (e[:, 0] ** 2).sum() for h in range(24)]); print("c_h", np.round(c, 2))
anch = R.groupby("rec").anchored.first()
T = []
for t in V.index:
    if t not in L.index or bool(anch.get(t, True)): continue
    pr = L.loc[t]
    if pr.prev_is_labelled or pr.xf or pr.p < .8: continue
    p = pr.prev
    if p not in V.index or V.at[p, "fold"] != V.at[t, "fold"] or p[:3] != t[:3] or int(p[4:7]) >= int(t[4:7]): continue
    T.append(dict(t=t, p=p, fold=V.at[t, "fold"], arm="E1" if bool(anch.get(p, False)) else "SA", chunk=V.at[t, "chunk"]))
T = pd.DataFrame(T); A = T[T.arm == "SA"]; A1 = A[A.t.str[4:7].astype(int) < 179]
print("SA", len(A), "pass1", len(A1), "gap p->t:", (A.t.str[4:7].astype(int) - A.p.str[4:7].astype(int)).value_counts().to_dict())
Y = R.groupby(["rec", "hour"]).y.first().unstack()
print("Y vs YD max diff on common", float(np.nanmax(np.abs(Y.loc[QD.index.intersection(Y.index)] - YD.loc[QD.index.intersection(Y.index)]).values)))
pivs = {}
for col in ("R3", "SG2", "EA2"):
    for sd, G in R.groupby("seed"): pivs[(col, sd)] = G.pivot_table(index="rec", columns="hour", values=col)
for sd in sorted(O.seed.unique()): pivs[("DIAG_sg", str(sd))] = O[O.seed == sd].groupby(["rec", "hour"]).sg.mean().unstack()
pivs[("DIAG_sgmean", "m")] = QD
D=set(QD.index); A1c = A1[A1.t.isin(D) & A1.p.isin(D)]; recs = list(A1c.t); prv = list(A1c.p); print("common targets", len(A1c))
print("\n== errors on the %d SA pass-1 targets t (and their p) ==" % len(recs))
print("%-14s %8s %8s %8s %8s %8s %8s %10s" % ("pred", "t 0h", "t 1-23", "t all", "p 23h", "p all", "|p23-t0|", "E-oracle0h"))
for k, P in pivs.items():
    t0 = P.loc[recs, 0].values - Y.loc[recs, 0].values; tr = (P.loc[recs, list(range(1, 24))] - Y.loc[recs, list(range(1, 24))]).values
    ta = (P.loc[recs] - Y.loc[recs]).values; p23 = P.loc[prv, 23].values - Y.loc[prv, 23].values; pa = (P.loc[prv] - Y.loc[prv]).values
    oracle = P.loc[prv, 23].values - Y.loc[recs, 0].values  # error of E as estimate of y_t0
    print("%-14s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %10.4f" % ("%s/%s" % k, rm(t0), rm(tr), rm(ta), rm(p23), rm(pa), rm(P.loc[prv, 23].values - P.loc[recs, 0].values), rm(oracle)))
print("truth: rms(y_t0 - y_p23) = %.4f ; rms(y_t0 - mean_h y_t) = %.4f" % (rm(Y.loc[recs, 0].values - Y.loc[prv, 23].values), rm(Y.loc[recs, 0].values - Y.loc[recs].mean(1).values)))

def sa(P, A, c, b=False):
    new = P.copy(); ws = {}
    for k in A.fold.unique():
        oth = A[A.fold != k]
        x = P.loc[oth.p, 23].values - P.loc[oth.t, 0].values; yv = Y.loc[oth.t, 0].values - P.loc[oth.t, 0].values
        w = float(np.clip((x * yv).sum() / (x * x).sum(), 0, 1)); ws[k] = w
        for _, r in A[A.fold == k].iterrows():
            new.loc[r.t] = P.loc[r.t].values + c * w * (P.at[r.p, 23] - P.at[r.t, 0])
    return new, ws
print("\n== same SA rule (EV-fold leave-fold-out w, same c_h) applied to each prediction on the same targets ==")
Ac = A[A.t.isin(D) & A.p.isin(D)]
for part, AA in (("pass-1 common", A1c), ("pass-1 all", A1), ("pass-2", Ac[Ac.t.str[4:7].astype(int) >= 179])):
    rr = list(AA.t)
    for k, P in pivs.items():
        if not set(AA.t)<=set(P.index): continue
        new, ws = sa(P, Ac if k[0].startswith("DIAG") else A, c)
        h0 = rm(new.loc[rr, 0] - Y.loc[rr, 0]) / rm(P.loc[rr, 0] - Y.loc[rr, 0]) - 1
        al = rm((new.loc[rr] - Y.loc[rr]).values) / rm((P.loc[rr] - Y.loc[rr]).values) - 1
        print("%s %-14s 0h %+6.1f%% all %+6.1f%% | w %s" % (part, "%s/%s" % k, 100 * h0, 100 * al, " ".join("%.2f" % ws[f] for f in sorted(ws))))
