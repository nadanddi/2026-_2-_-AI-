# -*- coding: utf-8 -*-
"""YS1 (2026-10-11 집 클로드 agentY, descriptive): can a RULE-SAFE successor anchor exist?
For eval record t (ID X) at row hour h, the only rule-safe successor candidates are OTHER-ID TRAINING records (and same-ID
earlier-numbered training records), identified with t's own inputs at hours <= h.
 (2) oracle slope of residual on (S0 - q_h) for CROSS-ID truth links (TL cert, fa != fb) vs same-ID; LOO gain on rows h>=20.
 (3) identification top-1 accuracy vs truth on training, by h: date term (outdoor midnight continuity; at h<23 the class curve
     must come from OTHER members of t's date class, since t's own late outdoor is not yet available) + inside term
     (t's hour-h inside/actuator values vs candidate 0 h).
 (4) eval coverage counts (record ids only) and realised-gain estimate with the identified (not true) successor.
Run: cd 집/클로드/research && PYTHONPATH="" PYTHONIOENCODING=utf-8 py -3.12 -u agentY_succ/ys1_cross_id_successor_feasibility_v1.py
"""
import os, sys, importlib.util, pickle
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd
RES = r"C:\work\farmai\집\클로드\research"; sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(RES, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)

S3 = os.path.join(env.LOCAL, "linker_v3")
R = pickle.load(open(os.path.join(S3, "base.pkl"), "rb")); r = R["recs"].reset_index(drop=True)
rid = np.array([f"{f}_{d:03d}" for f, d in zip(r.farm, r.day)]); N = len(r)
FARM = r.farm.values; DAY = r.day.values; TR = (r.split == "train").values; G = r.G.values
OUT, IN, ACT = R["OUT"], R["IN"], R["ACT"]
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); TL = TL[TL.cert].reset_index(drop=True)
succ = dict(zip(TL.a1, TL.b))  # index a -> index b (b is a's next date, same house)

ty = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); ty["rec"] = ty.row_id.str[:7]; ty["h"] = ty.row_id.str[8:10].astype(int)
ST = ty.pivot_table(index="rec", columns="h", values="sub_temp"); EC = ty.pivot_table(index="rec", columns="h", values="sub_ec")
FR = A2.A1.frames(); M, cells = FR["DIAG10"]; M = M.assign(q=np.mean(cells, 0), rec=M.farm + "_" + M.day.map("{:03d}".format))
QT = M.pivot_table(index="rec", columns="hour", values="q")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; QE = O.groupby(["rec", "hour"]).sg.mean().unstack()
TGT = {"temp": (ST, QT), "EC": (EC, QE)}
# whole-validator totals (DIAG10 OOF) for scaling: all rows with both y and q
TOT = {}
for nm, (Y, Q) in TGT.items():
    ix = Y.index.intersection(Q.index); e = (Y.loc[ix].values - Q.loc[ix].values).ravel(); e = e[np.isfinite(e)]
    TOT[nm] = (float((e ** 2).sum()), len(e))


def slope(x, y):
    m = np.isfinite(x) & np.isfinite(y); return float((x[m] * y[m]).sum() / (x[m] ** 2).sum()) if m.sum() > 2 else np.nan


def xy(pairs, nm, h):
    Y, Q = TGT[nm]; xs, ys = [], []
    for a, b in pairs:
        A, B = rid[a], rid[b]
        if A in Q.index and A in Y.index and B in Y.index:
            xs.append(Y.at[B, 0] - Q.at[A, h]); ys.append(Y.at[A, h] - Q.at[A, h])
        else:
            xs.append(np.nan); ys.append(np.nan)
    return np.array(xs), np.array(ys)


cross = [(a, b) for a, b in zip(TL.a1, TL.b) if FARM[a] != FARM[b]]
same = [(a, b) for a, b in zip(TL.a1, TL.b) if FARM[a] == FARM[b]]
print("truth cert links %d: cross-ID %d, same-ID %d (of same-ID: successor number < target number %d)" % (
    len(TL), len(cross), len(same), sum(DAY[b] < DAY[a] for a, b in same)))
print("(2) oracle successor slope by hour (residual on S0-q):")
HH = list(range(18, 24))
for nm in ("temp", "EC"):
    for lab, P in (("cross", cross), ("same", same), ("all", cross + same)):
        sl = [slope(*xy(P, nm, h)) for h in HH]
        print("  %-4s %-5s n%3d: %s" % (nm, lab, len(P), " ".join("%d:%.2f" % (h, v) for h, v in zip(HH, sl))))
    # bootstrap CI cross slopes at 22,23
    rng = np.random.default_rng(0)
    for h in (21, 22, 23):
        x, y = xy(cross, nm, h); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x)); bs.append(slope(x[i], y[i]))
        print("    %s cross h%d slope 95%% CI [%.2f, %.2f]" % (nm, h, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)))

# upper-bound gain (successor known), rows h>=20, LOO slope fitted on ALL cert links except the target
print("(2b) upper bound: successor KNOWN, correction q + beta_h(S0-q), beta LOO over all cert links, rows h>=20")
UB = {}
for nm in ("temp", "EC"):
    allP = cross + same; rowsum = {}
    for lab, P in (("cross", cross), ("all", allP)):
        d_sse = 0.0; base_sse = 0.0; n = 0
        for h in range(20, 24):
            xa, ya = xy(allP, nm, h); x, y = xy(P, nm, h)
            sxx = np.nansum(np.where(np.isfinite(xa * ya), xa ** 2, 0)); sxy = np.nansum(np.where(np.isfinite(xa * ya), xa * ya, 0))
            for xi, yi in zip(x, y):
                if not (np.isfinite(xi) and np.isfinite(yi)): continue
                bet = (sxy - xi * yi) / (sxx - xi ** 2)
                base_sse += yi ** 2; d_sse += (yi - bet * xi) ** 2 - yi ** 2; n += 1
        rowsum[lab] = (d_sse, base_sse, n)
        tsse, tn = TOT[nm]
        print("  %-4s %-5s rows %4d: touched-row RMSE %.4f -> %.4f (%.1f%%); whole-validator RMSE change %.2f%% (if these rows were all of its h>=20 changes)" % (
            nm, lab, n, np.sqrt(base_sse / n), np.sqrt((base_sse + d_sse) / n), 100 * (np.sqrt((base_sse + d_sse) / base_sse) - 1),
            100 * (np.sqrt((tsse + d_sse) / tsse) - 1)))
    UB[nm] = rowsum

# ---------------- (3) identification ----------------
V_OUT = [0, 1, 3]  # out_temp, out_hum, out_wspd (rad ~0 at midnight)
XIN = np.concatenate([IN, ACT], axis=2)  # in_temp,in_hum,in_co2, vent,shade,thermal,heating,circfan
NAMES = ["in_temp", "in_hum", "in_co2", "vent", "shade", "thermal", "heating", "circfan"]
jo = np.array([OUT[b, 0, V_OUT] - OUT[a, 23, V_OUT] for a, b in zip(TL.a1, TL.b)]); s_out = np.nanstd(jo, 0) + 1e-6
ji = np.array([XIN[b, 0] - XIN[a, 23] for a, b in zip(TL.a1, TL.b)]); s_in = np.nanstd(ji, 0) + 1e-6
print("(3) scales: outdoor midnight jump sd", np.round(s_out, 3), "| inside 23->0 jump sd", dict(zip(NAMES, np.round(s_in, 3))))
# class curves (exact outdoor equality groups G)
classes = np.unique(G)


def class_curve(g, excl):
    mem = [i for i in np.where(G == g)[0] if i != excl]
    return OUT[mem[0]] if mem else None


def date_cost(o_prev, o_next):  # o_prev: 24x4 day curve of target date; o_next: candidate curve
    lvl = np.abs(o_next[0, V_OUT] - o_prev[23, V_OUT]) / s_out
    ext = np.abs(o_next[0, V_OUT] - (2 * o_prev[23, V_OUT] - o_prev[22, V_OUT])) / s_out
    return float(np.nansum(lvl) + 0.5 * np.nansum(ext))


# accuracy of next-date class from outdoor only (class level), on truth links
rk = []
for a, b in zip(TL.a1, TL.b):
    c = {g: date_cost(OUT[a], OUT[np.where(G == g)[0][0]]) for g in classes if g != G[a]}
    order = sorted(c, key=c.get); rk.append(order.index(G[b]) + 1 if G[b] in order else 999)
rk = np.array(rk); print("  next-date class by outdoor midnight continuity (all %d truth links): top1 %.2f top3 %.2f top5 %.2f" % (
    len(rk), (rk == 1).mean(), (rk <= 3).mean(), (rk <= 5).mean()))


def pool(t, cross_only=True):
    # rule-safe: training records of other ID (+ optionally same-ID earlier-numbered training records)
    m = TR & (FARM != FARM[t]) if cross_only else TR & ((FARM != FARM[t]) | (DAY < DAY[t]))
    m[t] = False; return np.where(m)[0]


def scores(t, h, cand, w_in=1.0):
    if h == 23:
        o_prev = OUT[t]
    else:
        o_prev = class_curve(G[t], t)  # t's own late outdoor not yet available: use another same-date member
    dc = np.array([date_cost(o_prev, OUT[c]) for c in cand]) if o_prev is not None else np.zeros(len(cand))
    ic = np.array([np.nansum(np.abs(XIN[c, 0] - XIN[t, h]) / s_in) for c in cand])
    return dc, ic, dc + w_in * ic, o_prev is not None


print("  identification top-1 accuracy (target = truth a with CROSS-ID successor, pool = other-ID training records):")
print("    h | date-known(class has other member) | date-only top1 top3 | inside-only top1 | combined w=.5/1/2 top1 | pool median")
ID = {}
for h in HH:
    res = {k: [] for k in ("d1", "d3", "i1", "c05", "c1", "c2")}; kn = []; ps = []
    for a, b in cross:
        cand = pool(a); dc, ic, _, known = scores(a, h, cand); kn.append(known); ps.append(len(cand)); j = int(np.where(cand == b)[0][0])
        rd = (dc < dc[j]).sum() + 1; res["d1"].append(rd == 1); res["d3"].append(rd <= 3)
        res["i1"].append(ic.argmin() == j)
        for w, k in ((.5, "c05"), (1., "c1"), (2., "c2")):
            res[k].append((dc + w * ic).argmin() == j)
        if h in (22, 23): ID.setdefault(h, {})[a] = cand[(dc + 1. * ic).argmin()]
    print("   %2d | %.2f | %.2f %.2f | %.2f | %.2f/%.2f/%.2f | %d" % (h, np.mean(kn), np.mean(res["d1"]), np.mean(res["d3"]), np.mean(res["i1"]),
                                                              np.mean(res["c05"]), np.mean(res["c1"]), np.mean(res["c2"]), np.median(ps)))
# abstention: can we tell when the true successor is NOT in the cross-ID pool? targets = all truth a (same-ID successors too)
print("  abstention at h=23 (w=1): best-candidate combined score, targets with cross-ID succ vs same-ID succ")
bc, lab, hit = [], [], []
for a, b in cross + same:
    cand = pool(a); _, _, sc, _ = scores(a, 23, cand); bc.append(sc.min()); lab.append(FARM[a] != FARM[b]); hit.append(cand[sc.argmin()] == b)
bc = np.array(bc); lab = np.array(lab); hit = np.array(hit)
from scipy.stats import rankdata
rr = rankdata(-bc); n1 = lab.sum(); n0 = len(lab) - n1
print("    AUC(low score -> cross-ID succ exists) %.3f; median best score cross %.2f vs same %.2f" % (
    (rr[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), np.median(bc[lab]), np.median(bc[~lab])))
for thr in (np.percentile(bc, 10), np.percentile(bc, 20), np.percentile(bc, 30)):
    sel = bc <= thr
    print("    accept score<=%.2f: n %d, of which correct succ %d (precision %.2f), cross-ID coverage %.2f" % (
        thr, sel.sum(), hit[sel].sum(), hit[sel].mean() if sel.sum() else np.nan, hit[sel & lab].sum() / n1))

# realised gain at h=23 / 22 with IDENTIFIED successor on cross-ID targets, and with accept threshold on all targets
print("(3b) realised gain with identified successor (beta LOO from truth slopes, rows h in {22,23} only):")
for nm in ("temp", "EC"):
    Y, Q = TGT[nm]; allP = cross + same
    for h in (22, 23):
        xa, ya = xy(allP, nm, h); m = np.isfinite(xa * ya); bet = (xa[m] * ya[m]).sum() / (xa[m] ** 2).sum()
        d = 0.0; n = 0; dk = 0.0
        for a, b in cross:
            A = rid[a]; c = ID[h][a]; C = rid[c]
            if not (A in Q.index and A in Y.index and C in Y.index and rid[b] in Y.index): continue
            e0 = Y.at[A, h] - Q.at[A, h]; e1 = e0 - bet * (Y.at[C, 0] - Q.at[A, h]); ek = e0 - bet * (Y.at[rid[b], 0] - Q.at[A, h])
            d += e1 ** 2 - e0 ** 2; dk += ek ** 2 - e0 ** 2; n += 1
        tsse, tn = TOT[nm]
        print("  %-4s h%d beta %.2f: cross targets n%d, SSE change identified %+.3f vs known %+.3f; whole-validator RMSE %+.3f%% (known %+.3f%%)" % (
            nm, h, bet, n, d, dk, 100 * (np.sqrt((tsse + d) / tsse) - 1), 100 * (np.sqrt((tsse + dk) / tsse) - 1)))

# ---------------- (4) eval coverage ----------------
print("(4) evaluation records (60): does a rule-safe training successor on date+1 exist?")
EV = np.where(~TR)[0]
rows = []
for t in EV:
    cand = pool(t); dcs = np.array([date_cost(OUT[t], OUT[c]) for c in cand])  # own full-day outdoor: allowed only for the 23 h row
    # next-date class by outdoor among ALL classes that have a training member
    cg = {}
    for c, dcv in zip(cand, dcs): cg[G[c]] = dcv
    gbest = min(cg, key=cg.get) if cg else None
    # compare with best class over all classes incl. test-only dates (descriptive: is the best next date a training-covered one?)
    allc = {g: date_cost(OUT[t], OUT[np.where(G == g)[0][0]]) for g in classes if g != G[t]}
    gall = min(allc, key=allc.get)
    sameG_train = int((TR & (G == G[t]) & (FARM != FARM[t])).sum())
    rows.append(dict(rec=rid[t], best_all_cost=allc[gall], best_all_has_crossID_train=bool((TR & (G == gall) & (FARM != FARM[t])).any()),
                     n_crossID_train_in_best=int((TR & (G == gall) & (FARM != FARM[t])).sum()),
                     best_pool_cost=cg.get(gbest, np.nan), date_class_has_other_train=sameG_train > 0))
E = pd.DataFrame(rows)
# typical cost of a TRUE next date (truth links) for reference
tc = np.array([date_cost(OUT[a], OUT[b]) for a, b in zip(TL.a1, TL.b)])
print("  true next-date cost on training links: median %.2f, 90%% %.2f" % (np.median(tc), np.percentile(tc, 90)))
print("  eval recs whose best next date (all dates) has a cross-ID training record: %d / 60 (median n there %.1f)" % (
    E.best_all_has_crossID_train.sum(), E[E.best_all_has_crossID_train].n_crossID_train_in_best.median()))
print("  eval recs with best cross-ID training candidate date cost <= training 90%% true cost: %d / 60" % (E.best_pool_cost <= np.percentile(tc, 90)).sum())
print("  eval recs whose own date class has an other-ID training member (needed for h<23 date term): %d / 60" % E.date_class_has_other_train.sum())
for f in ("F13", "F47"):
    s = E[E.rec.str.startswith(f)]; print("   %s: best-next-date-has-crossID-train %d/%d -> %s" % (f, s.best_all_has_crossID_train.sum(), len(s), ",".join(s[s.best_all_has_crossID_train].rec)))
# linker v7 mirror-CV choices: training records whose chosen previous record is an eval record of the other ID
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"))
evs = set(rid[EV]); Lx = L[L.prev.isin(evs)]
Lx = Lx.assign(cross=Lx.rec.str[:3] != Lx.prev.str[:3])
print("  linker v7 (training rec -> chosen prev is eval rec): %d links, cross-ID %d, cross-ID with p>=.8 %d (distinct eval recs %d)" % (
    len(Lx), Lx.cross.sum(), (Lx.cross & (Lx.p >= .8)).sum(), Lx[Lx.cross & (Lx.p >= .8)].prev.nunique()))
os.makedirs(os.path.join(RES, "agentY_succ", "local"), exist_ok=True)
E.to_csv(os.path.join(RES, "agentY_succ", "local", "ys1_eval_coverage_v1.csv"), index=False)
Lx.to_csv(os.path.join(RES, "agentY_succ", "local", "ys1_v7_eval_successors_v1.csv"), index=False)
print("totals (DIAG10 OOF SSE, rows):", {k: (round(v[0], 2), v[1]) for k, v in TOT.items()})
