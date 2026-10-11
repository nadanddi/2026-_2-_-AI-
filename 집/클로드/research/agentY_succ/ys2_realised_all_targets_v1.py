# -*- coding: utf-8 -*-
"""YS2 (2026-10-11 집 클로드 agentY, descriptive): realised gain of the rule-safe cross-ID successor anchor applied to ALL
400 training targets (truth not needed: realised error uses the target's own label and the chosen candidate's 0 h label),
with an abstention threshold on the combined score; plus structure of cross-ID truth links (chunk ends) and eval chunk ends.
Reuses YS1 definitions by import.  Run: cd 집/클로드/research && PYTHONPATH="" PYTHONIOENCODING=utf-8 py -3.12 -u agentY_succ/ys2_realised_all_targets_v1.py
"""
import os, sys, importlib.util, io, contextlib
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import numpy as np, pandas as pd
P = os.path.join(r"C:\work\farmai\집\클로드\research", "agentY_succ", "ys1_cross_id_successor_feasibility_v1.py")
s = importlib.util.spec_from_file_location("ys1", P); Y1 = importlib.util.module_from_spec(s)
with contextlib.redirect_stdout(io.StringIO()):
    s.loader.exec_module(Y1)
rid, FARM, DAY, TR, TL = Y1.rid, Y1.FARM, Y1.DAY, Y1.TR, Y1.TL
# structure: cross-ID truth links — is a the last record of its same-ID consecutive-number run in its split?
ids = set(rid)
def run_end(i):
    return f"{FARM[i]}_{DAY[i]+1:03d}" not in ids or not TR[list(rid).index(f'{FARM[i]}_{DAY[i]+1:03d}')]
cr = [(a, b) for a, b in zip(TL.a1, TL.b) if FARM[a] != FARM[b]]
print("cross-ID truth links: %d; target a is last of a training run (next number missing/eval) %d; a in pass2 (>=179) %d" % (
    len(cr), sum(run_end(a) for a, b in cr), sum(DAY[a] >= 179 for a, b in cr)))
print("  same-ID links whose b == a's next number: %d / %d" % (sum(DAY[b] == DAY[a] + 1 for a, b in zip(TL.a1, TL.b) if FARM[a] == FARM[b]),
      sum(FARM[a] == FARM[b] for a, b in zip(TL.a1, TL.b))))
EV = np.where(~TR)[0]
ends = [rid[t] for t in EV if f"{FARM[t]}_{DAY[t]+1:03d}" not in set(rid[EV])]
print("eval chunk-end records (next number not an eval record): %d -> %s" % (len(ends), ",".join(ends)))
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")); Lx = L[L.prev.isin(set(rid[EV]))]
print("linker v7 training recs whose chosen prev is eval:", Lx[["rec", "prev", "p"]].to_dict("records"))
# realised gain on all training targets
for nm in ("temp", "EC"):
    Yt, Q = Y1.TGT[nm]; tsse, tn = Y1.TOT[nm]
    allP = list(zip(TL.a1, TL.b)); res = []
    for h in (20, 21, 22, 23):
        xa, ya = Y1.xy(allP, nm, h); m = np.isfinite(xa * ya); bet = (xa[m] * ya[m]).sum() / (xa[m] ** 2).sum()
        for t in np.where(TR)[0]:
            T = rid[t]
            if T not in Q.index or T not in Yt.index: continue
            cand = Y1.pool(t); dc, ic, sc, known = Y1.scores(t, h, cand)
            if not known: continue
            c = cand[sc.argmin()]; C = rid[c]
            if C not in Yt.index: continue
            e0 = Yt.at[T, h] - Q.at[T, h]; e1 = e0 - bet * (Yt.at[C, 0] - Q.at[T, h])
            if np.isfinite(e0) and np.isfinite(e1): res.append((h, sc.min(), e1 ** 2 - e0 ** 2))
    R_ = pd.DataFrame(res, columns=["h", "score", "d"])
    print("%s realised (all training targets, date-known at that h):" % nm)
    for h in (20, 21, 22, 23):
        rh = R_[R_.h == h]
        out = []
        for q in (5, 10, 20, 40, 100):
            thr = np.percentile(rh.score, q); sel = rh[rh.score <= thr]
            out.append("acc%d%%(n%d) %+.3f%%" % (q, len(sel), 100 * (np.sqrt((tsse + sel.d.sum()) / tsse) - 1)))
        print("  h%d: " % h + "  ".join(out))
    for q in (5, 10, 20):
        thr = {h: np.percentile(R_[R_.h == h].score, q) for h in (20, 21, 22, 23)}
        sel = R_[[sc <= thr[h] for h, sc in zip(R_.h, R_.score)]]
        print("  h20-23 combined, accept best %d%% per hour: rows %d, whole-validator RMSE %+.3f%%" % (q, len(sel), 100 * (np.sqrt((tsse + sel.d.sum()) / tsse) - 1)))
