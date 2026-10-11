# -*- coding: utf-8 -*-
"""SA2 (2026-10-12 집 클로드): SA1 with the per-hour weight MEASURED for this signal (SA1 diagnostic: 0 h -21%, 3 h -9%, >= 6 h no gain/slight loss because SA1 spread the correction with the carry of the WHOLE 0 h error).  SOFT ANCHOR = the model's own 23 h prediction of the
linked same-house PREVIOUS record (no label needed).
Oracle (truth links, this session): 0 h error temp .788 -> prev-day predicted 23 h (+bias) .553, blend .7 -> .480 (-39%);
EC .148 -> .121, blend .7 -> .115 (-22%).  Why: the model's 0 h prediction uses cross-record lag features that mostly come
from a DIFFERENT house (previous record number), while the previous same-house record's 23 h prediction saw that house's day.
Deployable rule (fixed before running):
  targets S = linker v7 mirror-CV records whose chosen previous record is UNLABELLED, same ID (xf False), p >= .8
             (in evaluation: same-ID earlier evaluation records -> their inputs and model predictions are allowed).
  s0      = q_prev(23) + b ;  d = w * (s0 - q_t(0)) ;  pred_h = q_h + c_h * d
  b       = mean(y_t(0) - q_prev(23)) and w = LS slope of (y_t(0) - q_t(0)) on (s0 - q_t(0)), both fitted on the OTHER chunks'
            S records (leave-one-chunk-out, chunk = (farm, (num - lo)//10)), pooled over cells/seeds;  w clipped to [0, 1]
  w_h     = per-hour LS slope of (y_t(h) - q_t(h)) on x = (q_prev(23) + b - q_t(0)), fitted on the OTHER chunks (replaces w * c_h); pred_h = q_h + w_h x
  Temperature: q = round-11 recipe DIAG10 OOF (4 cells) + submitted 1-step anchor rule (touches only labelled-anchor records,
  disjoint from S).  EC: q = RV1 CUR (+SG2) DIAG10 OOF (3 seeds).
PASS iff on S records (pooled pass 1 + 2): every cell better than q AND record bootstrap P(worse) < .025; pass-2 subset reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u sa2_soft_anchor_hourly_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s); s.loader.exec_module(A7)
A2 = A7.A2; rm = A2.rm


def carry(Y, Q, p1):
    e = Y - Q; e0 = e[:, [0]]
    return np.array([(e0[p1, 0] * e[p1, h]).sum() / (e0[p1, 0] ** 2).sum() for h in range(24)])


def run(name, recs, cellsY, cellsQ, L):
    """cellsQ: list of (n_rec, 24) prediction arrays; cellsY: (n_rec, 24) truth; recs: record ids (index)."""
    ri = {r: i for i, r in enumerate(recs)}
    S = L[(~L.prev_is_labelled.astype(bool)) & (~L.xf.astype(bool)) & (L.p >= .8) & L.rec.isin(ri) & L.prev.isin(ri)]
    t = np.array([ri[r] for r in S.rec]); pv = np.array([ri[r] for r in S.prev])
    num = np.array([int(r[4:7]) for r in recs]); farm = np.array([r[:3] for r in recs])
    lo = {f: num[farm == f].min() for f in ("F13", "F47")}
    chunk = np.array(["%s%d" % (farm[i], (num[i] - lo[farm[i]]) // 10) for i in t])
    p1 = num < 179
    outs = []
    for Q in cellsQ:
        c = carry(cellsY, Q, p1)
        new = Q.copy()
        for ch in np.unique(chunk):
            te = chunk == ch; tr = ~te
            b = np.mean(cellsY[t[tr], 0] - Q[pv[tr], 23])
            x = Q[pv[tr], 23] + b - Q[t[tr], 0]
            wh = np.array([(x * (cellsY[t[tr], h] - Q[t[tr], h])).sum() / (x * x).sum() for h in range(24)])
            xt = Q[pv[te], 23] + b - Q[t[te], 0]
            new[t[te]] = Q[t[te]] + wh[None, :] * xt[:, None]
        outs.append(new)
    rng = np.random.default_rng(20261216)
    sel = np.zeros(len(recs), bool); sel[t] = True; p2 = sel & ~p1
    rel = [rm((o - cellsY)[sel]) / rm((Q - cellsY)[sel]) - 1 for o, Q in zip(outs, cellsQ)]
    rel2 = [rm((o - cellsY)[p2]) / rm((Q - cellsY)[p2]) - 1 for o, Q in zip(outs, cellsQ)] if p2.any() else [np.nan]
    P = np.mean([((o - cellsY) ** 2 - (Q - cellsY) ** 2).sum(1) for o, Q in zip(outs, cellsQ)], axis=0)[sel]
    idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P[idx].sum(1) >= 0).mean())
    allrel = [rm(o - cellsY) / rm(Q - cellsY) - 1 for o, Q in zip(outs, cellsQ)]
    print("%s: S records %d (pass-2 %d) | S rows cells %s | pass-2 S mean %+.2f%% | improved %d/%d | P(worse) %.4f | whole-layout %s -> %s" % (
        name, len(t), int(p2.sum()), " ".join("%+.2f%%" % (100 * v) for v in rel), 100 * np.nanmean(rel2), int((P < 0).sum()), len(P), pw,
        " ".join("%+.2f%%" % (100 * v) for v in allrel), "PASS" if all(v < 0 for v in rel) and pw < .025 else "FAIL"), flush=True)


def main():
    L = pd.read_csv(A7.LK)
    La = L[L.prev_is_labelled.astype(bool) & L.anchor_st23.notna() & (L.steps_to_anchor == 1)]
    R = A2.A1.frames(); M, cells = R["DIAG10"]; key, A = A7.build(M, La)
    base = [A7.correct(M, q, key, A, .8, "const", "lin8", 1, 1.5) for q in cells]
    M = M.assign(rec=key); recs = sorted(M.rec.unique())
    piv = lambda v: pd.DataFrame({"rec": M.rec.values, "h": M.hour.values, "v": v}).pivot_table(index="rec", columns="h", values="v").reindex(recs).values
    run("TEMP", recs, piv(M.sub_temp.values), [piv(b) for b in base], L)
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy()
    O["rec"] = O.row_id.str[:7]; recsE = sorted(O.rec.unique())
    Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack().reindex(recsE).values
    QE = [O[O.seed == sd].groupby(["rec", "hour"]).sg.mean().unstack().reindex(recsE).values for sd in sorted(O.seed.unique())]
    run("EC", recsE, Y, QE, L)


if __name__ == "__main__":
    main()
