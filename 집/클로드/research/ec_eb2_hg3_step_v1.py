# -*- coding: utf-8 -*-
"""EB2 (2026-10-11 집 클로드, user: "컴퓨터 감당 가능한 선까지 EC 개선책 찾아봐"): stronger HG3 step under the evaluation
geometry.  EB1 (6.539): HG3 (high-EC correction, step 0.5 toward the mean of the two best anchors when pm >= .9, both anchors
>= 1.0 and S_low <= -1) improved every buffered cell with normal days exactly unchanged; high days stay under-predicted (the
evaluation has 16 sealed days of 60).  EB2 tests the step size only.  Fixed before running.
Baseline: CUR + HG3 (step .5, buffered references).  Candidates (k = 2, alpha .0125): HG3 step .75 / 1.0 (same gate).
PASS iff all 3 seeds better on buffered EL1 AND buffered P2LOO pass-2 rows AND normal-day RMSE unchanged (|diff| < 1e-9) in
every cell AND seed-mean block bootstrap P(worse) on buffered EL1 < .0125 (structural floor ~.038 noted in 6.341 -> a FAIL
caused only by P is reported as such).  Caveat: step chosen on the same OOF family that built HG3.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec_eb2_hg3_step_v1.py
"""
import os, sys, json, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("hg3", os.path.join(HERE, "ec3_HG3_strict_guard_gate_v1.py")); hg3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(hg3)
CK = os.path.join(env.LOCAL, "eb2_ckpt"); ALPHA = .0125
SGN = hg3.SGN
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))


def correction_a(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal, DM=None, ST=None, AH=.5):
    out = frame[pcol].values.copy(); hg = frame[pcol].values.copy()
    roles = R.set_index(["farm", "day"]).role
    full_cache = {}

    def twin_date(f, d, h):
        E = [e for e in R[R.farm == f].day if (f, e) in ref]
        cols = np.asarray(hrs <= h)
        A = WV.loc[[(f, e) for e in E]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1)); ok = dist <= .05
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
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values   # z from reference records only
            q = (S.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
            C = np.nan_to_num(((S.loc[list(zip(Gm.farm, Gm.day))].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            o = np.argsort(dist)
            a1 = ec[(f, Gm.iloc[o[0]].day)]; pm = cum[k]
            a2 = ec[(f, Gm.iloc[o[1]].day)] if len(o) > 1 else np.nan
            j = frame.index.get_loc(ii)
            if abs(a1 - pm) <= .30:
                out[j] = rr[pcol] + 0.5 * (a1 - pm)
            hg[j] = out[j]
            mu_, sd_ = ST[(f, h)]
            slow = float(np.nansum(SGN * (DM.loc[(f, d, h)].values - mu_) / sd_))
            if pm >= .9 and a1 >= 1.0 and np.isfinite(a2) and a2 >= 1.0 and slow <= -1.0:
                hg[j] = rr[pcol] + AH * ((a1 + a2) / 2 - pm)
    return out, hg



def run():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = hg3.p3.prepare()
    Rr, WV, hrs, SIG, DM = hg3.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
    O = O[(O.tag == "CUR") & O.validator.isin(["EL1", "P2LOO"])]
    lohi = pd.read_csv(os.path.join(env.LOCAL, "ec3_WT1_all.csv"), usecols=["row_id", "lo", "hi"]).drop_duplicates("row_id").set_index("row_id")
    for (vn, k), G0 in O.groupby(["validator", "fold"]):
        path = os.path.join(CK, "%s_%d.csv" % (vn, k))
        if os.path.exists(path):
            continue
        vd = set(zip(G0.farm, G0.day)); buf = set()
        for f in {x[0] for x in vd}:
            labd = sorted(d for (ff, d) in labset if ff == f); hid = sorted(d for (ff, d) in vd if ff == f)
            b = [d for d in labd if d < hid[0] and (f, d) not in vd]; a = [d for d in labd if d > hid[-1] and (f, d) not in vd]
            if b: buf.add((f, b[-1]))
            if a: buf.add((f, a[0]))
        ref = {x for x in labset if x not in vd and x not in buf}
        cal = hg3.ref_calendar(Rr, WV, ref); ST = hg3.s_low_table(DM, ref); part = []
        for (tag, s), G in G0.groupby(["tag", "seed"]):
            G = G.reset_index(drop=True); lo, hi = lohi.lo.reindex(G.row_id).values, lohi.hi.reindex(G.row_id).values
            res = {}
            for ah in (.5, .75, 1.0):
                _, hg = correction_a(G, "nosg", vd, lockd, ec, Rr, WV, hrs, SIG, ref, cal, DM, ST, AH=ah); res["h%03d" % int(ah * 100)] = np.clip(hg, lo, hi)
            part.append(G[["validator", "fold", "row_id", "tag", "seed", "farm", "day", "hour", "sub_ec"]].assign(**res))
        pd.concat(part).to_csv(path, index=False); print("  %s/%d done" % (vn, k), flush=True)


def summarize():
    rng = np.random.default_rng(20261118)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    P = P[P.day >= 179]
    W = P.copy()
    W["dm"] = W.groupby(["validator", "seed", "farm", "day"]).sub_ec.transform("mean")
    base = "h050"; cands = {"HG3 step .75": "h075", "HG3 step 1.0": "h100"}
    for name, c in cands.items():
        ok = True; print("\n==== %s (buffered)" % name)
        for v in ("EL1", "P2LOO"):
            X = W[W.validator == v]; cells = [rm(Z[c] - Z.sub_ec) / rm(Z[base] - Z.sub_ec) - 1 for _, Z in X.groupby("seed")]
            ok &= all(x < 0 for x in cells)
            for _, Z in X.groupby("seed"):
                zn = Z[Z.dm < 1]; ok &= abs(rm(zn[c] - zn.sub_ec) - rm(zn[base] - zn.sub_ec)) < 1e-9
            M = X.groupby("row_id").agg(y=("sub_ec", "first"), b=(base, "mean"), c=(c, "mean"), farm=("farm", "first"), day=("day", "first"), dm=("dm", "first"))
            nm = M.dm < 1
            line = "  %-5s cells %s | base %.4f -> %.4f | normal %.4f -> %.4f | high %.4f -> %.4f | F13 %+.2f%% F47 %+.2f%%" % (
                v, " ".join("%+.2f%%" % (100 * x) for x in cells), rm(M.b - M.y), rm(M.c - M.y), rm((M.b - M.y)[nm]), rm((M.c - M.y)[nm]),
                rm((M.b - M.y)[~nm]), rm((M.c - M.y)[~nm]), *(100 * (rm((M.c - M.y)[M.farm == f]) / rm((M.b - M.y)[M.farm == f]) - 1) for f in ("F13", "F47")))
            if v == "EL1":
                cl = (M.farm + "_" + (M.day // 5).astype(str)).values
                dd = pd.Series((M.c - M.y) ** 2 - (M.b - M.y) ** 2).groupby(cl).agg(["sum", "count"])
                idx = rng.integers(0, len(dd), (20000, len(dd))); p = float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())
                ok &= p < ALPHA; line += " | P(worse) %.4f" % p
            print(line)
        print("VERDICT %s: %s" % (name, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("EB2_SUM") != "1":
        run()
    summarize()
