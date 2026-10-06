# -*- coding: utf-8 -*-
"""EC stage-3 WT2: TabPFN share beyond 0.4 WITH SG2 on top.  Fixed before running; 2026-10-07 집 클로드.
WT1 (6.355): PFN share .2 -> .3 -> .4 improved monotonically (B4 = .4 adopted by the rule); larger shares untested.
Configurations: core(w) = clip(shrink((1-w) R3_DP1 + w PFN)) per seed 47 / 1414 / 6464 (WT1 stored members,
PFN contexts 5-8), then SG2 (unchanged rule) on top.  w in {.4 (reference, adopted), .5, .6, .8};
w = 1.0 (PFN alone + SG2) descriptive only.
Sets: TM (111 test-matched days, DIAG10 OOF, SG2 on all rows), P2LOO and EL1 pass-2 rows.
Decision (fixed, k = 3 candidates, alpha .025/3 = .0083): a share w > .4 replaces .4 iff every seed improves over
w = .4 (+SG2) on TM, P2LOO and EL1 AND the TM seed-mean block bootstrap P(worse) < .0083; if several pass, the
largest gain on TM seed mean."""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
S = (47, 1414, 6464); SCH = ["act_thermal", "act_shade", "act_co2"]
raw, full, lab, lock, signatures, fds = sg2.p3.prepare()
R, WV, hrs, SIG = sg2.prepare_structure()
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
roles = R.set_index(["farm", "day"]).role
alld = {f: sorted(R[R.farm == f].day) for f in ("F13", "F47")}
XA = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id"] + SCH) for f in ("train_X.csv", "test_X.csv")])
XA = XA[XA.row_id.str[:3].isin(["F13", "F47"])].copy()
XA["farm"], XA["day"], XA["hour"] = XA.row_id.str[:3], XA.row_id.str[4:7].astype(int), XA.row_id.str[8:10].astype(int)
PV = {f: XA[XA.farm == f].pivot_table(index="day", columns="hour", values=SCH) for f in ("F13", "F47")}


def correct(F, pcol, vd, ref, cal, mode):
    out = F[pcol].values.copy(); cache = {}

    def twin_date(f, d, h):
        E = [e for e in alld[f] if (f, e) in ref]; cols = np.asarray(hrs <= h)
        A = WV.loc[[(f, e) for e in E]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1)); ok = dist <= .05
        return float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o])) if ok.any() else None

    def full_date(f, d):
        if (f, d) in ref: return cal[(f, d)]
        if (f, d) in cache: return cache[(f, d)]
        t = twin_date(f, d, 23)
        if t is None:
            i = alld[f].index(d); t = (full_date(f, alld[f][i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        cache[(f, d)] = t; return t
    zs = {}
    for f in ("F13", "F47"):
        refd = [e for e in alld[f] if (f, e) in ref]
        M = PV[f].loc[refd]
        zs[f] = {v: (np.nanmean(M[v].values), np.nanstd(M[v].values) or 1) for v in SCH}
    for (f, d), idx in F.groupby(["farm", "day"]).groups.items():
        E = [e for e in alld[f] if (f, e) in ref and (f, e) in ec.index and (f, e) not in lockd]
        calc = np.array([cal[(f, e)] for e in E]); i = alld[f].index(d)
        rows = F.loc[idx].sort_values("hour"); cum = rows[pcol].expanding().mean().values
        Zf = {v: (PV[f][v] - zs[f][v][0]) / zs[f][v][1] for v in SCH}
        for k, (ii, rr) in enumerate(rows.iterrows()):
            h = int(rr.hour)
            cq = twin_date(f, d, h) if h >= 5 else None
            if cq is None: cq = full_date(f, alld[f][i - 1]) + 0.1
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            if not m.any(): continue
            Em = [e for e, z in zip(E, m) if z]
            if mode == "sg3":
                q = np.concatenate([Zf[v].loc[d].values[:h + 1] for v in SCH])
                C = np.array([np.concatenate([Zf[v].loc[e].values[:h + 1] for v in SCH]) for e in Em])
                dist = np.sqrt(np.nanmean((C - q) ** 2, axis=1))
            else:
                Sh = SIG[h]; RS = Sh.loc[[(f, e) for e in alld[f] if (f, e) in ref]].astype(float)
                mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
                qq = (Sh.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(qq)
                C = np.nan_to_num(((Sh.loc[[(f, e) for e in Em]].values.astype(float) - mu) / sd)[:, use])
                dist = np.sqrt(((C - qq[use]) ** 2).mean(axis=1))
            dist = np.nan_to_num(dist, nan=9.0) + .15 * np.abs(calc[m] - cq)
            a1 = float(ec[(f, Em[int(np.argmin(dist))])]); pm = cum[k]
            if abs(a1 - pm) <= .30:
                out[F.index.get_loc(ii)] = rr[pcol] + .5 * (a1 - pm)
    return out


O = pd.read_csv(os.path.join(env.LOCAL, "ec3_WT1_all.csv"))
W = (.4, .5, .6, .8, 1.0)
for s in S:
    r3 = .6 * O["et_%d" % s] + .3 * O["lgb_%d" % s] + .1 * O["mlp_%d" % s]
    for w in W:
        O["core%d_%d" % (int(w * 10), s)] = np.clip((1 - w) * r3 + w * O.pfn, O.lo, O.hi)
TM = pd.read_csv(os.path.join(env.LOCAL, "tm1_set_v1.csv")); TMs = set(zip(TM.farm, TM.day))
O["inTM"] = [(f, d) in TMs for f, d in zip(O.farm, O.day)]
O = O[((O.validator == "DIAG10") & O.inTM) | ((O.validator != "DIAG10") & (O.day >= 179))].reset_index(drop=True)
for (v, k), G in O.groupby(["validator", "validation_fold"]):
    vd = set(zip(G.farm, G.day)); ref = {x for x in labset if x not in vd}; cal = sg2.ref_calendar(R, WV, ref)
    Gs = G.sort_values(["farm", "day", "hour"])
    for s in S:
        for w in W:
            c = "core%d_%d" % (int(w * 10), s)
            O.loc[Gs.index, "sg%d_%d" % (int(w * 10), s)] = np.clip(correct(Gs, c, vd, ref, cal, "sg2"), Gs.lo, Gs.hi)
    print("%s/%s done" % (v, k), flush=True)
O.to_csv(os.path.join(env.LOCAL, "ec3_WT2_all.csv"), index=False)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
sets = {"TM": O[O.validator == "DIAG10"], "P2LOO": O[O.validator == "P2LOO"], "EL1": O[O.validator == "EL1"]}
print("\nseed-mean RMSE with SG2 (normal / high) by PFN share:")
for nm, G in sets.items():
    G = G.assign(dm=G.groupby(["farm", "day"]).sub_ec.transform("mean"))
    line = []
    for w in W:
        p = G[["sg%d_%d" % (int(w * 10), s) for s in S]].mean(axis=1)
        line.append("w%.1f %.4f (%.4f/%.4f)" % (w, r(p - G.sub_ec), r((p - G.sub_ec)[G.dm < 1]), r((p - G.sub_ec)[G.dm >= 1])))
    print("  %-6s %s" % (nm, "  ".join(line)))
best = None
for w in (.5, .6, .8):
    ok = True; cells = []
    for nm, G in sets.items():
        for s in S:
            a, b = r(G["sg4_%d" % s] - G.sub_ec), r(G["sg%d_%d" % (int(w * 10), s)] - G.sub_ec)
            ok &= b < a; cells.append("%s s%d %+.1f%%" % (nm, s, 100 * (b / a - 1)))
    T = sets["TM"].copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
    bm = T[["sg4_%d" % s for s in S]].mean(axis=1); cm = T[["sg%d_%d" % (int(w * 10), s) for s in S]].mean(axis=1)
    dd = ((cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2).groupby(T.cl).agg(["sum", "count"]); sm, n = dd["sum"].values, dd["count"].values
    idx = np.random.default_rng(20261007).integers(0, len(sm), (20000, len(sm)))
    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    gain = r(cm - T.sub_ec) / r(bm - T.sub_ec) - 1
    print("\nw %.1f vs .4: %s | TM P(worse) %.4f -> %s" % (w, ", ".join(cells), p, "PASS" if ok and p < .0083 else "FAIL"))
    if ok and p < .0083 and (best is None or gain < best[1]):
        best = (w, gain)
print("\nWT2 decision:", "use PFN share %.1f" % best[0] if best else "keep .4")
