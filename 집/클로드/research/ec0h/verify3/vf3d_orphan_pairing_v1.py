# -*- coding: utf-8 -*-
"""VERIFY3 d (2026-10-12 집 클로드; EXPLORATORY, addendum v3).  Orphan dates, 2x3 house pairing, OOD check, whole-date indoor drop."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6
OUT = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
Xf = X.reshape(N, -1)
pe = lambda A, i: A[i, 23] + 0.5 * (A[i, 23] - A[i, 22]); pb = lambda A, j: A[j, 0] - 0.5 * (A[j, 1] - A[j, 0])
L = np.where(TRAIN)[0]
CE = np.abs(pe(EC, L)[:, None] - pb(EC, L)[None, :]) / .01 + np.abs(pe(ST, L)[:, None] - pb(ST, L)[None, :]) / .3
np.fill_diagonal(CE, np.inf)
orph = dict(zip(L, CE.min(1) > 2))
print("training records without a training next day (min cost > 2): %d / %d (%.3f)" % (sum(orph.values()), len(L), np.mean(list(orph.values()))))
# training date groups (training records only)
par = {i: i for i in L}
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a in range(len(L)):
    for b in range(a + 1, len(L)):
        if (np.abs(Xf[L[a]] - Xf[L[b]]) < 1e-6).mean() >= .9: par[fd(L[b])] = fd(L[a])
G = {}
for i in L: G.setdefault(fd(i), []).append(i)
full_orph = [g for g in G.values() if all(orph[i] for i in g)]
print("training dates: %d | dates whose every training member is orphan: %d" % (len(G), len(full_orph)))
for t_ in ["F47_221", "F13_228"]:
    t = idx[t_]
    rows = []
    for g in G.values():
        pa = X[g, 23] + 0.5 * (X[g, 23] - X[g, 22]); oj = np.sum(np.array([1, 1, .3, .5]) * np.abs(pa - X[t, 0]) / sdO, 1).min()
        rows.append((oj, all(orph[i] for i in g), " ".join(rid(i) for i in g)))
    rows.sort()
    print(f"\n{t_}: training dates by outdoor midnight cost (top 12) [oj, all-orphan, members]")
    for r in rows[:12]: print("   %.2f %s %s" % r)
    print("   all-orphan dates with oj<1.5:", [r[2] for r in rows if r[0] < 1.5 and r[1]], "| total dates oj<1.5:", sum(r[0] < 1.5 for r in rows))
# 2x3 pairing using same-ID earlier record (allowed)
for t_, s_ in (("F47_221", "F47_220"), ("F13_228", "F13_227")):
    print(f"\npairing {t_} (and same-ID earlier same-date {s_}) vs P1 members: mid / dTair / fp0 / fp23")
    for r_ in (s_, t_):
        r = idx[r_]
        for c_ in ("F47_144", "F47_145", "F13_141"):
            c = idx[c_]
            md = float(np.sum(np.minimum(np.nan_to_num(np.abs(VARS[c, 23] - VARS[r, 0]) / sdV, nan=2.0), 6)))
            print(f"   {r_} <- {c_} (EC23 {EC[c,23]:.3f}): mid {md:5.1f} dT {IN_[r,0,0]-IN_[c,23,0]:+5.1f} fp0 {F6.fp_dist(r,c,0):.2f} fp23 {F6.fp_dist(r,c,23):.2f}")
# OOD check in v2 CV
R = pd.read_csv(os.path.join(OUT, "vf3b_cv_oof_v2.csv")); D = R[R.h == 0]
print("\nv2 CV rows with P_hidden_date > .9:", len(D[D.P_hidden_date > .9]), "| top_ok", D[D.P_hidden_date > .9].top_ok.mean().round(2),
      "| v7m labelled & ok:", D[(D.P_hidden_date > .9) & D.v7m_lab_same].v7m_ok.tolist())
# whole-date indoor drop among true links
TLa = pd.read_pickle(S3 + r"\truth.pkl"); T = TLa[TLa.cert]
ev = []
for a, b in zip(T.a1.astype(int), T.b.astype(int)):
    ga = [j for j in L if (np.abs(Xf[j] - Xf[a]) < 1e-6).mean() >= .9]
    gb = [j for j in L if (np.abs(Xf[j] - Xf[b]) < 1e-6).mean() >= .9]
    dd = [IN_[k, 0, 0] - IN_[j, 23, 0] for j in ga for k in gb]
    ev.append(dict(dropmax=np.nanmax(dd) if dd else np.nan, dT=IN_[b, 0, 0] - IN_[a, 23, 0], dEC=EC[b, 0] - EC[a, 23]))
E = pd.DataFrame(ev)
w = E[E.dropmax <= -3]
print("true links where every visible cross-pair indoor change <= -3: n %d, |dEC| median %.3f max %.3f (all links median %.3f)" % (
    len(w), w.dEC.abs().median() if len(w) else np.nan, w.dEC.abs().max() if len(w) else np.nan, E.dEC.abs().median()))
print("orphan status P1:", {r: bool(orph[idx[r]]) for r in ["F47_144", "F47_145", "F13_141"]}, "| min cost F47_144 -> ", rid(L[int(np.argmin(CE[list(L).index(idx['F47_144'])]))]), round(float(CE[list(L).index(idx['F47_144'])].min()), 2))
