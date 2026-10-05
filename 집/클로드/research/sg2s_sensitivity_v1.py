# -*- coding: utf-8 -*-
"""SG2S (robustness / sensitivity, fixed before running; 2026-10-05 집 클로드).
Is SG2's gain (6.310) a lucky choice of its settings?  Recompute the SG2 correction on stored
out-of-fold predictions while varying every hand-set knob; NOT a selection step (a better
variant would still need a new pre-registered formal run).
Knobs: calendar window W in {2, 3, 4, 5}; signature set in {all 13, operation 7 (th, he, co, sh,
ve, fo, cf), climate 6 (n_t, n_h, n_c, d_t, d_c, mx_t)}; anchors k in {1, mean of best 2,
mean of best 3}; guard G in {.2, .3, .4}; blend .5 fixed.  Current SG2 = (3, all, 1, .3).
Data: pass-2 rows of DIAG10 (hk0_rows_v1.csv: R3S seed mean, folds) and EL1 (ec3_AF0_all.csv
base_7/101/2024 mean, folds); reference = labelled records outside the fold (as SG2).
Report: change vs R3S for every variant on both sets; share of variants better on BOTH;
rank of the current setting.  Robust if >= 80 % of the 108 variants improve on both sets."""
import env  # noqa: F401
import importlib.util, json, os, sys, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
SIGC = ["n_t", "n_h", "n_c", "d_t", "d_c", "mx_t", "th", "he", "co", "sh", "ve", "fo", "cf"]
SETS = {"all13": SIGC, "ops7": ["th", "he", "co", "sh", "ve", "fo", "cf"], "clim6": ["n_t", "n_h", "n_c", "d_t", "d_c", "mx_t"]}
WINS = (2, 3, 4, 5); KS = (1, 2, 3); GS = (.2, .3, .4)
raw, full, lab, lock, signatures, fds = sg2.p3.prepare()
R, WV, hrs, SIG = sg2.prepare_structure()
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
roles = R.set_index(["farm", "day"]).role
days_all = {f: sorted(R[R.farm == f].day) for f in ("F13", "F47")}


def anchor_lists(F, vd):
    """per row: dict (W, set) -> list of anchor labels sorted by score (best first)."""
    ref = {x for x in labset if x not in vd}; cal = sg2.ref_calendar(R, WV, ref); cache = {}
    def twin_date(f, d, h):
        E = [e for e in days_all[f] if (f, e) in ref]; cols = np.asarray(hrs <= h)
        A = WV.loc[[(f, e) for e in E]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1)); ok = dist <= .05
        return float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o])) if ok.any() else None
    def full_date(f, d):
        if (f, d) in ref: return cal[(f, d)]
        if (f, d) in cache: return cache[(f, d)]
        t = twin_date(f, d, 23)
        if t is None:
            i = days_all[f].index(d); t = (full_date(f, days_all[f][i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        cache[(f, d)] = t; return t
    out = {}
    for (f, d), idx in F.groupby(["farm", "day"]).groups.items():
        E = [e for e in days_all[f] if (f, e) in ref and (f, e) in ec.index and (f, e) not in lockd]
        calc = np.array([cal[(f, e)] for e in E]); ev = np.array([ec[(f, e)] for e in E])
        i = days_all[f].index(d)
        for ii in idx:
            h = int(F.loc[ii, "hour"])
            cq = twin_date(f, d, h) if h >= 5 else None
            if cq is None: cq = full_date(f, days_all[f][i - 1]) + 0.1
            S = SIG[h]; RS = S.loc[[(f, e) for e in days_all[f] if (f, e) in ref]].astype(float)
            mu, sd = RS.mean(), RS.std().replace(0, np.nan)
            qa = (S.loc[(f, d)].astype(float) - mu) / sd
            res = {}
            for Wn in WINS:
                m = (np.abs(calc - cq) <= Wn) & (calc != cq)
                if not m.any():
                    continue
                Cm = ((S.loc[[(f, e) for e, z in zip(E, m) if z]].astype(float) - mu) / sd)
                for sn, cols in SETS.items():
                    q = qa[cols].values; use = ~np.isnan(q)
                    C = np.nan_to_num(Cm[cols].values[:, use])
                    sc = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
                    res[(Wn, sn)] = ev[m][np.argsort(sc)][:3]
            out[ii] = res
    return out


frames = []
H = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv")); H = H[H.day >= 179].copy(); H["src"] = "DIAG10"
A = pd.read_csv(os.path.join(env.LOCAL, "ec3_AF0_all.csv")); A = A[(A.validator == "EL1") & (A.day >= 179)].copy()
A["p"] = A[["base_7", "base_101", "base_2024"]].mean(axis=1); A["src"] = "EL1"
for D in (H, A):
    D = D.sort_values(["validation_fold", "farm", "day", "hour"]).reset_index(drop=True)
    D["pm"] = D.groupby(["farm", "day"]).p.transform(lambda z: z.expanding().mean())
    AL = {}
    for k, F in D.groupby("validation_fold"):
        vd = set(zip(F.farm, F.day))  # validation set of this fold (pass-2 days of the fold)
        if D.src.iloc[0] == "DIAG10":
            O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"), usecols=["farm", "day", "validator", "validation_fold"])
            O = O[(O.validator == "DIAG10") & (O.validation_fold == k)]; vd = set(zip(O.farm, O.day))
        AL.update(anchor_lists(F, vd))
        print(D.src.iloc[0], "fold", k, "done", flush=True)
    D["al"] = [AL.get(i, {}) for i in D.index]
    frames.append(D)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
rows = []
for Wn, sn, k, G in itertools.product(WINS, SETS, KS, GS):
    cell = dict(W=Wn, sig=sn, k=k, G=G)
    for D in frames:
        a = np.array([np.mean(x[(Wn, sn)][:k]) if (Wn, sn) in x and len(x[(Wn, sn)]) >= 1 else np.nan for x in D.al])
        d = a - D.pm.values
        new = np.where(~np.isnan(a) & (np.abs(d) <= G), D.p.values + .5 * d, D.p.values)
        dm = D.groupby(["farm", "day"]).sub_ec.transform("mean")
        base = r(D.p - D.sub_ec)
        cell[D.src.iloc[0]] = 100 * (r(new - D.sub_ec) / base - 1)
        cell[D.src.iloc[0] + "_normal"] = 100 * (r((new - D.sub_ec)[dm < 1]) / r((D.p - D.sub_ec)[dm < 1]) - 1)
    rows.append(cell)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(env.LOCAL, "sg2s_grid_v1.csv"), index=False)
both = (T.DIAG10 < 0) & (T.EL1 < 0)
T["avg"] = (T.DIAG10 + T.EL1) / 2
cur = T[(T.W == 3) & (T.sig == "all13") & (T.k == 1) & (T.G == .3)].iloc[0]
print("\nvariants %d; better than R3S on BOTH sets: %d (%.0f%%)" % (len(T), both.sum(), 100 * both.mean()))
print("current SG2 (3, all13, 1, .3): DIAG10 %+.2f%%, EL1 %+.2f%%, rank by average %d of %d" % (
    cur.DIAG10, cur.EL1, int((T.avg < cur.avg).sum()) + 1, len(T)))
print("\nby knob (mean change %): ")
for kn in ("W", "sig", "k", "G"):
    print(T.groupby(kn)[["DIAG10", "EL1", "DIAG10_normal", "EL1_normal"]].mean().round(2).to_string(), "\n")
print("best 8 by average:"); print(T.sort_values("avg").head(8).round(2).to_string(index=False))
print("\nSG2S robust (>= 80% of variants better on both):", both.mean() >= .8)
