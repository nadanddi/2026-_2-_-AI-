# -*- coding: utf-8 -*-
"""SP1 (2026-10-11 집 클로드, user: "확률 섞는 방식 실험해봐"): SOFT neighbour mixing instead of SG2's hard top-1 pick.
SG2 picks the single most similar labelled training record (control signature 13 + calendar term, calendar +-3) and moves the
prediction halfway to its daily EC only if it is within .30 of the running prediction mean.  Every earlier linking attempt
(SG1/SG2/LINK/CH/GX/FC) made HARD choices; a wrong pick costs a lot.  SP1 averages ALL candidates weighted by similarity.
Fixed before running.  Same candidates, distances, calendar and causality as SG2 (hg3.correction code path; hours 0..h only).
  weights  w_c = exp(-(dist_c - min dist) / TAU),  TAU = 0.25,  normalised;  L = sum w_c * EC_c (daily mean of candidate)
  SP1a: if |L - pm_h| <= .30: pred = p + 0.5 (L - pm_h)                      (SG2 guard, soft target)
  SP1b: pred = p + 0.5 * exp(-|L - pm_h| / .30) * (L - pm_h)                  (soft guard, no hard cut)
Baseline: SG2 output on the same stored OOF (RV1 file, 9th-round-like config CUR = 0.8 R3_DP1 + 0.2 PFN, seeds 47/1414/6464).
Judgement (k = 2 -> alpha .0125), pass-2 rows only: PASS iff all 3 seeds x {DIAG10, P2LOO, EL1} improve vs SG2 AND seed-mean
farm x 5-day block bootstrap P(worse) on P2LOO < .0125.  Reported: normal (label day mean < 1) / high-EC days, per farm.
Caveat (fixed now): these OOF were used when SG2 was developed; a pass is evidence for soft > hard on the same data, a fresh
layout / new seeds check is needed before adoption.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_sp1_soft_neighbor_mix_v1.py
"""
import os, sys, json, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("hg3", os.path.join(HERE, "ec3_HG3_strict_guard_gate_v1.py")); hg3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hg3)
TAU = .25; ALPHA = .0125
CK = os.path.join(env.LOCAL, "sp1_ckpt")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


def soft(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal):
    """hg3.correction() with the hard pick replaced by soft weights.  Returns (SP1a, SP1b)."""
    A = frame[pcol].values.copy(); B = frame[pcol].values.copy()
    roles = R.set_index(["farm", "day"]).role; full_cache = {}

    def twin_date(f, d, h):
        E = [e for e in R[R.farm == f].day if (f, e) in ref]
        cols = np.asarray(hrs <= h)
        Am = WV.loc[[(f, e) for e in E]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        dist = np.sqrt(np.nanmean((Am - b) ** 2, axis=1)); ok = dist <= .05
        return float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o])) if ok.any() else None

    def full_date(f, d):
        if (f, d) in ref:
            return cal[(f, d)]
        if (f, d) in full_cache:
            return full_cache[(f, d)]
        t = twin_date(f, d, 23)
        if t is None:
            days = sorted(R[R.farm == f].day); i = days.index(d)
            t = (full_date(f, days[i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        full_cache[(f, d)] = t
        return t

    for (f, d), idx in frame.groupby(["farm", "day"]).groups.items():
        if d < 179:
            continue
        G = R[R.farm == f]
        G = G[[(f, e) in ref and (f, e) in ec.index and (f, e) not in lock for e in G.day]]
        calc = np.array([cal[(f, e)] for e in G.day])
        days = sorted(R[R.farm == f].day); i = days.index(d)
        rows = frame.loc[idx].sort_values("hour")
        cum = rows[pcol].expanding().mean().values
        for k, (ii, rr) in enumerate(rows.iterrows()):
            h = int(rr.hour)
            cq = twin_date(f, d, h) if h >= 5 else None
            if cq is None:
                cq = full_date(f, days[i - 1]) + 0.1
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            if not m.any():
                continue
            Gm = G[m]; S = SIG[h]
            RS = S.loc[[(f, e) for e in R[R.farm == f].day if (f, e) in ref]].astype(float)
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            q = (S.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
            C = np.nan_to_num(((S.loc[list(zip(Gm.farm, Gm.day))].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            w = np.exp(-(dist - dist.min()) / TAU); w /= w.sum()
            L = float(np.sum(w * np.array([ec[(f, e)] for e in Gm.day]))); pm = cum[k]
            j = frame.index.get_loc(ii)
            if abs(L - pm) <= .30:
                A[j] = rr[pcol] + 0.5 * (L - pm)
            B[j] = rr[pcol] + 0.5 * np.exp(-abs(L - pm) / .30) * (L - pm)
    return A, B


def run():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = hg3.p3.prepare()
    Rr, WV, hrs, SIG, DM = hg3.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    D10 = {i: vd for name, i, vd in fds if name == "DIAG10"}
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
    O = O[O.tag == "CUR"]
    lohi = pd.read_csv(os.path.join(env.LOCAL, "ec3_WT1_all.csv"), usecols=["row_id", "lo", "hi"]).drop_duplicates("row_id").set_index("row_id")
    for (vn, k), G0 in O.groupby(["validator", "fold"]):
        if not (G0.day >= 179).any():
            continue
        path = os.path.join(CK, "%s_%d.csv" % (vn, k))
        if os.path.exists(path):
            continue
        vd = D10[k] if vn == "DIAG10" else set(zip(G0.farm, G0.day))
        ref = {x for x in labset if x not in vd}
        cal = hg3.ref_calendar(Rr, WV, ref); part = []
        for s, G in G0.groupby("seed"):
            G = G.reset_index(drop=True)
            a, b = soft(G, "nosg", vd, lockd, ec, Rr, WV, hrs, SIG, ref, cal)
            lo, hi = lohi.lo.reindex(G.row_id).values, lohi.hi.reindex(G.row_id).values
            part.append(G.assign(sp1a=np.clip(a, lo, hi), sp1b=np.clip(b, lo, hi)))
        pd.concat(part).to_csv(path, index=False); print("  %s/%d done" % (vn, k), flush=True)


def boot(Z, col, rng):
    cl = (Z.farm + "_" + (Z.day // 5).astype(str)).values
    dd = pd.Series((Z[col] - Z.sub_ec) ** 2 - (Z.sg - Z.sub_ec) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261111)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    P = P[P.day >= 179].copy()
    P["dm"] = P.groupby(["validator", "fold", "seed", "farm", "day"]).sub_ec.transform("mean")
    for c in ("sp1a", "sp1b"):
        ok = True; print("\n==== %s (TAU %.2f)" % (c, TAU))
        for v in ("DIAG10", "P2LOO", "EL1"):
            X = P[P.validator == v]; cells = []
            for s, Z in X.groupby("seed"):
                cells.append(rm(Z[c] - Z.sub_ec) / rm(Z.sg - Z.sub_ec) - 1)
            ok &= all(x < 0 for x in cells)
            Zm = X.groupby("row_id").agg(sub_ec=("sub_ec", "first"), sg=("sg", "mean"), cc=(c, "mean"), farm=("farm", "first"), day=("day", "first"), dm=("dm", "first"))
            nm, hi = Zm.dm < 1, Zm.dm >= 1
            line = "  %-6s cells %s | SG2 %.4f -> %.4f | normal %.4f -> %.4f | high %.4f -> %.4f | F13 %+.2f%% F47 %+.2f%%" % (
                v, " ".join("%+.2f%%" % (100 * x) for x in cells), rm(Zm.sg - Zm.sub_ec), rm(Zm.cc - Zm.sub_ec),
                rm((Zm.sg - Zm.sub_ec)[nm]), rm((Zm.cc - Zm.sub_ec)[nm]), rm((Zm.sg - Zm.sub_ec)[hi]), rm((Zm.cc - Zm.sub_ec)[hi]),
                *(100 * (rm((Zm.cc - Zm.sub_ec)[Zm.farm == f]) / rm((Zm.sg - Zm.sub_ec)[Zm.farm == f]) - 1) for f in ("F13", "F47")))
            if v == "P2LOO":
                p = boot(Zm.rename(columns={"cc": c}).reset_index(), c, rng); ok &= p < ALPHA; line += " | P(worse) %.4f" % p
            print(line)
        print("VERDICT %s: %s" % (c, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("SP1_SUM") != "1":
        run()
    summarize()
