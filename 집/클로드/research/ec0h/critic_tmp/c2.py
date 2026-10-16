import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
H = os.path.join(env.ROOT, u"집", u"클로드", "research")
F = pd.read_csv(os.path.join(H, "ev_validator", "ev_folds_v1.csv")); V = F[F.role == "val"].set_index("rec")
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anch = R.groupby("rec").anchored.first()
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; recs = sorted(O.rec.unique()); ri = {r: i for i, r in enumerate(recs)}
Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack().reindex(recs).values
QE = [O[O.seed == sd].groupby(["rec", "hour"]).sg.mean().unstack().reindex(recs).values for sd in sorted(O.seed.unique())]
dfold = O.groupby("rec").fold.first().reindex(recs).values
S = L[(~L.prev_is_labelled.astype(bool)) & (~L.xf.astype(bool)) & (L.p >= .8) & L.rec.isin(ri) & L.prev.isin(ri)].copy()
S["tn"] = S.rec.str[4:7].astype(int); S["pn"] = S.prev.str[4:7].astype(int)
S["samefarm"] = S.rec.str[:3] == S.prev.str[:3]; S["earlier"] = S.pn < S.tn
S["evsame"] = [V.at[a, "fold"] == V.at[b, "fold"] if (a in V.index and b in V.index) else False for a, b in zip(S.rec, S.prev)]
S["t_anch"] = S.rec.map(anch).fillna(True).astype(bool); S["dsame"] = [dfold[ri[a]] == dfold[ri[b]] for a, b in zip(S.rec, S.prev)]
print("SA1 S", len(S), "| samefarm", S.samefarm.sum(), "earlier", S.earlier.sum(), "EV-same-fold", S.evsame.sum(), "EV-anchored t", S.t_anch.sum(), "DIAG10 same fold", S.dsame.sum())
print("gap dist", (S.tn - S.pn).value_counts().to_dict())
num = np.array([int(r[4:7]) for r in recs]); farm = np.array([r[:3] for r in recs]); p1 = num < 179
def carry(Q):
    e = Y - Q; return np.array([(e[p1, 0] * e[p1, h]).sum() / (e[p1, 0] ** 2).sum() for h in range(24)])
def rule(S, Q, useb, grp):
    t = np.array([ri[r] for r in S.rec]); pv = np.array([ri[r] for r in S.prev]); c = carry(Q); new = Q.copy(); g = grp(S)
    for k in np.unique(g):
        te = g == k; tr = ~te
        b = np.mean(Y[t[tr], 0] - Q[pv[tr], 23]) if useb else 0.
        x = Q[pv[tr], 23] + b - Q[t[tr], 0]; yv = Y[t[tr], 0] - Q[t[tr], 0]; w = float(np.clip((x * yv).sum() / (x * x).sum(), 0, 1))
        new[t[te]] = Q[t[te]] + c[None, :] * (w * (Q[pv[te], 23] + b - Q[t[te], 0]))[:, None]
    return new, t
lo = {f: num[farm == f].min() for f in ("F13", "F47")}
chunkg = lambda S: np.array(["%s%d" % (r[:3], (int(r[4:7]) - lo[r[:3]]) // 10) for r in S.rec])
evg = lambda S: np.array([V.at[r, "fold"] for r in S.rec])
for nm, SS in (("SA1 set", S), ("samefarm+earlier", S[S.samefarm & S.earlier]), ("EV-like (same EV fold, t unanch)", S[S.samefarm & S.earlier & S.evsame & ~S.t_anch]),
               ("DIAG10 same-fold subset", S[S.dsame]), ("DIAG10 diff-fold subset", S[~S.dsame])):
    for useb in (True, False):
        for gn, g in (("chunk", chunkg), ("EVfold", evg)):
            res = []
            for Q in QE:
                new, t = rule(SS, Q, useb, g)
                res.append(rm((new - Y)[t]) / rm((Q - Y)[t]) - 1)
            print("%-34s n=%3d b=%d grp=%-6s  %s" % (nm, len(SS), useb, gn, " ".join("%+.1f%%" % (100 * v) for v in res)))
print("dtypes", L.prev_is_labelled.dtype, L.xf.dtype)
S["p_anch"] = S.prev.map(anch).fillna(False).astype(bool)
print("p anchored (E1-type) in S:", S.p_anch.sum())
for nm, SS in (("SA only (p unanch)", S[~S.p_anch]), ("E1 only", S[S.p_anch])):
    for gn, g in (("chunk", chunkg), ("EVfold", evg)):
        res = []; res1 = []
        for Q in QE:
            new, t = rule(SS, Q, False, g); res.append(rm((new - Y)[t]) / rm((Q - Y)[t]) - 1)
        print("%-20s n=%3d b=0 grp=%-6s %s" % (nm, len(SS), gn, " ".join("%+.1f%%" % (100 * v) for v in res)))
# per-record contributions on full S, b=0 EVfold, seed mean
SS = S; t = np.array([ri[r] for r in SS.rec])
d = np.mean([(((rule(SS, Q, False, evg)[0] - Y) ** 2).sum(1) - ((Q - Y) ** 2).sum(1))[t] for Q in QE], 0)
SS = SS.assign(d=d); print(SS.sort_values("d", ascending=False)[["rec", "prev", "p", "p_anch", "d"]].head(12).to_string())
print("sum d", d.sum(), "SA-only sum", SS[~SS.p_anch].d.sum(), "E1 sum", SS[SS.p_anch].d.sum())
