# -*- coding: utf-8 -*-
"""SG4: SG2 neighbour selection with a 'dong' fingerprint from the thermal-curtain schedule (ADDED, not replacing
the SG2 signature - SG3, 6.360, which REPLACED it, failed).  (2026-10-09 집 클로드, user: "동 지문(커튼 일정 포함)으로
SG2 이웃 고르기 개선 해봐"; expert answer 10-09: each dong is managed differently).  Fixed before running.
Base predictions: CT2 checkpoints (submission pipeline 0.8 R3 + 0.2 PFN, shrink, clip; seeds 8383/1919/7171;
PFN contexts 9-12): column pre_REF_<seed> = before SG2, fin_REF_<seed> = with SG2 (the comparator).  No refit.
Variants (applied to pre_REF exactly like SG2: same reference set, calendar, guard .30, step .5, clip):
  A  ADD    SG2 signature (13 values, hours 0..h) + 3 curtain values on hours 0..h: thf = hours with
            act_thermal >= 99.9, thn = 1 if every hour 0..min(h,8) had act_thermal <= .1, th9 = act_thermal at 9 h
            (NaN before 9 h; NaN entries are skipped by SG2's distance as before)
  B  GATE   SG2 unchanged, but candidates restricted to records whose curtain class at hour h equals the query's
            (class = 'schedule-consistent so far': hours 0..min(h,8) <= .1, hour 9 in 50-95 if h >= 9,
            hours 10..min(h,15) >= 99.9, 16 in 50-95 if h >= 16, 17..h <= .1); if no candidate remains -> plain SG2
Sets: pass-2 rows of DIAG10 (6 folds), EL1, P2LOO (CT2 folds).
RULE (k = 2, alpha .0125), X vs SG2 (fin_REF):
  (a) every seed x {DIAG10, EL1, P2LOO} better (9/9);
  (b) P2LOO seed-mean day bootstrap (farm, day) share(X not better) < .0125;
  (c) P2LOO seed mean still better after removing the 3 largest-gain days, and separately after removing the
      curtain-discovery days F13 214, F13 217, F47 229, F47 231.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u sg4_dong_fingerprint_sg2_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "run"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
SEEDS = (8383, 1919, 7171)
CK = os.path.join(env.LOCAL, "ct2_ckpt")
OUT = os.path.join(env.LOCAL, "sg4_ckpt")
DISC = {("F13", 214), ("F13", 217), ("F47", 229), ("F47", 231)}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def curtain_tables():
    X = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id", "act_thermal"]) for f in ("train_X.csv", "test_X.csv")])
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    P = X.pivot_table(index=["farm", "day"], columns="hour", values="act_thermal")
    EXTRA, CLS = {}, {}
    for h in range(24):
        sub = P[list(range(h + 1))]
        thf = (sub >= 99.9).sum(1).astype(float)
        thn = (P[list(range(min(h, 8) + 1))] <= .1).all(1).astype(float)
        th9 = P[9] if h >= 9 else pd.Series(np.nan, index=P.index)
        EXTRA[h] = pd.DataFrame({"thf": thf, "thn": thn, "th9": th9})
        ok = (P[list(range(min(h, 8) + 1))] <= .1).all(1)
        if h >= 9: ok &= P[9].between(50, 95)
        if h >= 10: ok &= (P[list(range(10, min(h, 15) + 1))] >= 99.9).all(1)
        if h >= 16: ok &= P[16].between(50, 95)
        if h >= 17: ok &= (P[list(range(17, h + 1))] <= .1).all(1)
        CLS[h] = ok
    return EXTRA, CLS


def correction_gate(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal, CLS):
    """copy of sg2.correction with one change: candidates filtered to the query's curtain class (fallback: none)."""
    out = frame[pcol].values.copy()
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
            qc = bool(CLS[h].get((f, d), False))
            mg = m & np.array([bool(CLS[h].get((f, e), False)) == qc for e in G.day])
            if mg.any():
                m = mg
            if not m.any():
                continue
            Gm = G[m]; S = SIG[h]
            RS = S.loc[[(f, e) for e in R[R.farm == f].day if (f, e) in ref]].astype(float)
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            q = (S.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
            C = np.nan_to_num(((S.loc[list(zip(Gm.farm, Gm.day))].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            b = Gm.iloc[int(np.argmin(dist))]
            a1 = ec[(f, b.day)]; pm = cum[k]
            if abs(a1 - pm) <= .30:
                out[frame.index.get_loc(ii)] = rr[pcol] + 0.5 * (a1 - pm)
    return out


def day_share(g, a, b, n=20000, seed=0):
    keys, inv = np.unique((g.farm + "_" + g.day.astype(str)).values, return_inverse=True)
    sa = np.bincount(inv, weights=a, minlength=len(keys)); sb = np.bincount(inv, weights=b, minlength=len(keys))
    rng = np.random.default_rng(seed); worse = 0
    for _ in range(n // 1000):
        ix = rng.integers(0, len(keys), size=(1000, len(keys)))
        worse += int((sb[ix].sum(1) >= sa[ix].sum(1)).sum())
    return worse / n


def main():
    os.makedirs(OUT, exist_ok=True)
    raw, full, lab, lock, signatures, fds = sg2.p3.prepare()
    R, WV, hrs, SIG = sg2.prepare_structure()
    EXTRA, CLS = curtain_tables()
    SIGA = {h: SIG[h].join(EXTRA[h], how="left") for h in range(24)}
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    CLSd = {h: CLS[h].to_dict() for h in range(24)}
    for fn in sorted(os.listdir(CK)):
        path = os.path.join(OUT, fn)
        if os.path.exists(path):
            continue
        G = pd.read_csv(os.path.join(CK, fn))
        G = G[["row_id", "farm", "day", "hour", "sub_ec", "validator", "validation_fold"] +
              ["pre_REF_%d" % s for s in SEEDS] + ["fin_REF_%d" % s for s in SEEDS]].reset_index(drop=True)
        vd = set(zip(G.farm, G.day)); ref = {x for x in labset if x not in vd}; cal = sg2.ref_calendar(R, WV, ref)
        keep = [(f, d) not in vd for f, d in zip(lab.farm, lab.day)]; lo, hi = lab[keep].sub_ec.min(), lab[keep].sub_ec.max()
        for s in SEEDS:
            pc = "pre_REF_%d" % s
            # clip: labels outside the validation days (close to CT2's training-range clip; corrections are <= .15)
            G["A_%d" % s] = np.clip(sg2.correction(G, pc, vd, lockd, ec, R, WV, hrs, SIGA, ref, cal), lo, hi)
            G["B_%d" % s] = np.clip(correction_gate(G, pc, vd, lockd, ec, R, WV, hrs, SIG, ref, cal, CLSd), lo, hi)
        G.to_csv(path, index=False)
        print(fn, "done", flush=True)
    summarize()


def summarize():
    G = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))], ignore_index=True)
    G = G[G.day >= 179].reset_index(drop=True)
    print("folds:", G.groupby("validator").validation_fold.nunique().to_dict())
    mean = lambda g, c: np.mean([g["%s_%d" % (c, s)] for s in SEEDS], axis=0)
    for c in ("A", "B"):
        print("\n==== %s vs SG2 (pass-2 rows)" % c)
        cells = []
        for v in ("DIAG10", "EL1", "P2LOO"):
            g = G[G.validator == v]; y = g.sub_ec.to_numpy(float)
            sr = [(r(g["fin_REF_%d" % s] - y), r(g["%s_%d" % (c, s)] - y)) for s in SEEDS]; cells += [b < a for a, b in sr]
            mS, mC = mean(g, "fin_REF"), mean(g, c)
            print("  %-6s SG2 %.4f  %s %.4f  %+.2f%%  seeds %s  (before any correction %.4f)" % (
                v, r(mS - y), c, r(mC - y), 100 * (r(mC - y) / r(mS - y) - 1), "".join("+" if b < a else "-" for a, b in sr), r(mean(g, "pre_REF") - y)))
        g = G[G.validator == "P2LOO"].reset_index(drop=True); y = g.sub_ec.to_numpy(float)
        mS, mC = mean(g, "fin_REF"), mean(g, c); keys = list(zip(g.farm, g.day))
        share = day_share(g, (mS - y) ** 2, (mC - y) ** 2)
        gain = pd.Series((mS - y) ** 2 - (mC - y) ** 2).groupby(pd.MultiIndex.from_tuples(keys)).sum().sort_values(ascending=False)
        aux = {}
        for nm, drop in (("without top-3 gain days %s" % sorted(gain.index[:3]), set(gain.index[:3])), ("without discovery days", DISC)):
            m = np.array([k not in drop for k in keys]); aux[nm] = r(mC[m] - y[m]) < r(mS[m] - y[m])
            print("  aux %-60s SG2 %.4f %s %.4f -> %s" % (nm, r(mS[m] - y[m]), c, r(mC[m] - y[m]), "better" if aux[nm] else "NOT better"))
        print("  days better %d / %d" % ((gain > 0).sum(), len(gain)))
        ok = len(cells) == 9 and all(cells) and share < .0125 and all(aux.values())
        print("  VERDICT %s: seed x set better %d/9, P2LOO day-share %.4f, aux %s -> %s" % (c, sum(cells), share, all(aux.values()), "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if MODE == "sum":
        summarize()
    else:
        main()
