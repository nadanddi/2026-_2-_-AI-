import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
exec(open(r"C:\work\farmai\집\클로드\research\ec0h\critic_tmp\c1.py", encoding="utf-8").read().split("pivs = {}")[0].split("print(\"SA\"")[0])
Y = R.groupby(["rec", "hour"]).y.first().unstack()
pv = {sd: G.pivot_table(index="rec", columns="hour", values="EA2") for sd, G in R.groupby("seed")}
# c_h from non-target pass-1 records only
tg = set(T.t) | set(T.p)
p1n = [r for r in QD.index if int(r[4:7]) < 179 and r not in tg]; e2 = (YD.loc[p1n] - QD.loc[p1n]).values
c2 = np.array([(e2[:, 0] * e2[:, h]).sum() / (e2[:, 0] ** 2).sum() for h in range(24)])
print("c_h(all pass-1) mean %.3f vs c_h(non-target, n=%d) mean %.3f, max |diff| %.3f" % (c.mean(), len(p1n), c2.mean(), np.abs(c - c2).max()))
def sa(P, A, cc):
    new = P.copy(); ws = {}
    for k in A.fold.unique():
        oth = A[A.fold != k]; x = P.loc[oth.p, 23].values - P.loc[oth.t, 0].values; yv = Y.loc[oth.t, 0].values - P.loc[oth.t, 0].values
        w = float(np.clip((x * yv).sum() / (x * x).sum(), 0, 1)); ws[k] = w
        for _, r in A[A.fold == k].iterrows(): new.loc[r.t] = P.loc[r.t].values + cc * w * (P.at[r.p, 23] - P.at[r.t, 0])
    return new, ws
r1 = list(A1.t)
for nm, cc in (("c_all", c), ("c_nontarget", c2)):
    print(nm, " ".join("%+.1f%%" % (100 * (rm((sa(P, A, cc)[0].loc[r1] - Y.loc[r1]).values) / rm((P.loc[r1] - Y.loc[r1]).values) - 1)) for P in pv.values()))
# per record (seed mean) contributions, SA pass-1
d = pd.Series(np.mean([(((sa(P, A, c)[0].loc[r1] - Y.loc[r1]) ** 2).sum(1) - ((P.loc[r1] - Y.loc[r1]) ** 2).sum(1)).values for P in pv.values()], 0), index=r1)
jump = pd.Series(np.abs(Y.loc[r1, 0].values - Y.loc[list(A1.p), 23].values), index=r1)
print("total d %.3f; top gains:" % d.sum()); print(pd.DataFrame(dict(d=d, jump=jump, p=L.loc[r1].p)).sort_values("d").head(6).round(3).to_string())
print("worst:"); print(pd.DataFrame(dict(d=d, jump=jump, p=L.loc[r1].p)).sort_values("d").tail(4).round(3).to_string())
print("without top record: %.1f%% of total kept; drop-top1 seed-mean rel:" % (100 * (d.sum() - d.min()) / d.sum()))
print("cross-midnight truth jump |y_t0-y_p23| in SA pass-1: >0.3: %d, >0.5: %d (median %.3f)" % ((jump > .3).sum(), (jump > .5).sum(), jump.median()))
# pass-2 detail
r2 = [r for r in A.t if int(r[4:7]) >= 179]
for r in r2:
    p = A.set_index("t").at[r, "p"]; P = pv["7"]
    print("p2", r, "<-", p, "pL=%.2f" % L.at[r, "p"], "y_t0 %.3f q_t0 %.3f E=q_p23 %.3f y_p23 %.3f | DIAG sg t0 %s p23 %s" % (Y.at[r, 0], P.at[r, 0], P.at[p, 23], Y.at[p, 23],
          "%.3f" % QD.at[r, 0] if r in QD.index else "-", "%.3f" % QD.at[p, 23] if p in QD.index else "-"))
# pass-1 vs pass-2 0h error: all EV val records (no anchor) R3 vs DIAG sg
allv = [r for r in V.index if not anch.get(r, True)]
for part, f in (("pass-1", lambda r: int(r[4:7]) < 179), ("pass-2", lambda r: int(r[4:7]) >= 179)):
    rr = [r for r in allv if f(r)]; rd = [r for r in rr if r in QD.index]
    print("unanchored EV val %s n=%d: EV EA2 0h %.3f 23h %.3f | DIAG sg (n=%d) 0h %.3f 23h %.3f" % (part, len(rr), rm(pv["7"].loc[rr, 0] - Y.loc[rr, 0]), rm(pv["7"].loc[rr, 23] - Y.loc[rr, 23]), len(rd), rm(QD.loc[rd, 0] - YD.loc[rd, 0]), rm(QD.loc[rd, 23] - YD.loc[rd, 23])))
# pseudo-SA pool size for pass-2 (DIAG10 OOF, prev any status, same farm, earlier, p>=.8)
LL = L.reset_index(); Q = LL[(LL.p >= .8) & ~LL.xf & (LL.rec.str[:3] == LL.prev.str[:3]) & (LL.rec.str[4:7].astype(int) > LL.prev.str[4:7].astype(int)) & LL.rec.isin(QD.index) & LL.prev.isin(QD.index)]
print("pseudo-SA pool (DIAG10 OOF, p>=.8 same-ID earlier prev, prev labelled or not): all %d, pass-2 %d (prev unlabelled %d)" % (len(Q), (Q.rec.str[4:7].astype(int) >= 179).sum(), ((Q.rec.str[4:7].astype(int) >= 179) & ~Q.prev_is_labelled).sum()))
gap = pd.Series((A1.t.str[4:7].astype(int) - A1.p.str[4:7].astype(int)).values, index=r1)
for gname, m in (("gap1", gap == 1), ("gap2", gap == 2), ("gap>=3", gap >= 3)):
    rr = list(gap[m].index)
    print(gname, "n=%d d_sum %.2f improved %d | rel %s" % (len(rr), d[m].sum(), (d[m] < 0).sum(), " ".join("%+.1f%%" % (100 * (rm((sa(P, A, c)[0].loc[rr] - Y.loc[rr]).values) / rm((P.loc[rr] - Y.loc[rr]).values) - 1)) for P in pv.values())))
dup = A.p.value_counts(); print("prev used by >1 target:", dup[dup > 1].to_dict())
