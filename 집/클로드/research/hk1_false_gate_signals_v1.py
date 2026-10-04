# -*- coding: utf-8 -*-
"""HK1 (diagnostic, fixed before running; 2026-10-05 집 클로드).  After HG1 (6.313: high
days -15..-30 % but normal days +5..+30 %, P .040): what separates FALSE gated days
(model says high, anchors high, label < 1; e.g. F13 231 / 233) from TRUE high days?
Broad gate to get more false days: rows with pm_h >= 0.8 and a1 >= 1.0.
Per gated row (hour-causal, as SG2): d1 / d2 signature distance of the best two
anchors, d2 - d1, |cal(a1) - cal(a2)|, |a1 - a2|, pm_h, a1 - pm_h, share of EC >= 1
among the top-5 anchors (hi5), mean of top-5 anchors (m5), signature distance gap
between the best high anchor and the best low (< 1) anchor (dgap = d_low - d_high;
positive = high anchors are clearly closer), hour.  Day summary = mean over its gated
rows (+ first gated hour).
Data: DIAG10 R3S seed-mean OOF, all days, reference = labelled records outside fold.
Report per feature: medians TRUE vs FALSE days and AUC (FALSE vs TRUE).
Clue (fixed): best |AUC - .5| feature with family-wise permutation p < .05 (labels
permuted within farm, 10000, max over features) -> a candidate filter for HG2."""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)


def anchors(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal):
    """Per row: a1, a2 (best two anchors), pm_h.  SG2 logic, all days."""
    out = pd.DataFrame(index=frame.index, columns=["a1", "a2", "pm", "d1", "d2", "dcal", "hi5", "m5", "dgap"], dtype=float)
    roles = R.set_index(["farm", "day"]).role
    cache = {}

    def twin_date(f, d, h):
        E = [e for e in R[R.farm == f].day if (f, e) in ref]
        cols = np.asarray(hrs <= h)
        A = WV.loc[[(f, e) for e in E]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1)); ok = dist <= .05
        return float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o])) if ok.any() else None

    def full_date(f, d):
        if (f, d) in ref:
            return cal[(f, d)]
        if (f, d) in cache:
            return cache[(f, d)]
        t = twin_date(f, d, 23)
        if t is None:
            days = sorted(R[R.farm == f].day); i = days.index(d)
            t = (full_date(f, days[i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        cache[(f, d)] = t
        return t

    for (f, d), idx in frame.groupby(["farm", "day"]).groups.items():
        G = R[R.farm == f]
        G = G[[(f, e) in ref and (f, e) in ec.index and (f, e) not in lock for e in G.day]]
        calc = np.array([cal[(f, e)] for e in G.day])
        days = sorted(R[R.farm == f].day); i = days.index(d)
        rows = frame.loc[idx].sort_values("hour")
        cum = rows[pcol].expanding().mean().values
        for k, (ii, rr) in enumerate(rows.iterrows()):
            h = int(rr.hour)
            out.loc[ii, "pm"] = cum[k]
            cq = twin_date(f, d, h) if h >= 5 else None
            if cq is None:
                cq = (full_date(f, days[i - 1]) + 0.1) if i else 0.0
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            if not m.any():
                continue
            Gm = G[m]; S = SIG[h]
            RS = S.loc[[(f, e) for e in R[R.farm == f].day if (f, e) in ref]].astype(float)
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            q = (S.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
            C = np.nan_to_num(((S.loc[list(zip(Gm.farm, Gm.day))].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            o = np.argsort(dist)
            ev = np.array([ec[(f, e)] for e in Gm.day]); cc = calc[m]
            out.loc[ii, "a1"] = ev[o[0]]; out.loc[ii, "d1"] = dist[o[0]]
            if len(o) > 1:
                out.loc[ii, "a2"] = ev[o[1]]; out.loc[ii, "d2"] = dist[o[1]]
                out.loc[ii, "dcal"] = abs(cc[o[0]] - cc[o[1]])
            t5 = o[:5]
            out.loc[ii, "hi5"] = float((ev[t5] >= 1).mean()); out.loc[ii, "m5"] = float(ev[t5].mean())
            hi_m, lo_m = ev >= 1, ev < 1
            if hi_m.any() and lo_m.any():
                out.loc[ii, "dgap"] = dist[lo_m].min() - dist[hi_m].min()
    return out


def main():
    raw, full, lab, lock, signatures, fds = sg2.p3.prepare()
    R, WV, hrs, SIG = sg2.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].reset_index(drop=True)
    O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
    parts = []
    for k, G in O.groupby("validation_fold"):
        vd = set(zip(G.farm, G.day)); ref = {x for x in labset if x not in vd}
        cal = sg2.ref_calendar(R, WV, ref)
        parts.append(anchors(G, "p", vd, lockd, ec, R, WV, hrs, SIG, ref, cal))
        print("fold %s done" % k, flush=True)
    A = pd.concat(parts); O = O.join(A)
    O.to_csv(os.path.join(env.LOCAL, "hk1_rows_v1.csv"), index=False)
    O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
    g = O.a1.notna() & (O.pm >= .8) & (O.a1 >= 1.0)
    G = O[g].copy()
    G["d21"] = G.d2 - G.d1; G["a12"] = (G.a1 - G.a2).abs(); G["apm"] = G.a1 - G.pm
    F = ["d1", "d2", "d21", "dcal", "a12", "pm", "apm", "hi5", "m5", "dgap", "hour"]
    D = G.groupby(["farm", "day"]).agg({**{c: "mean" for c in F}, "dm": "first"}).reset_index()
    D["first_h"] = G.groupby(["farm", "day"]).hour.min().values
    D["false"] = (D.dm < 1).astype(int); F2 = F + ["first_h"]
    print("gated days %d: true high %d, false %d" % (len(D), (D.false == 0).sum(), D.false.sum()))
    print(D[D.false == 1][["farm", "day", "dm"] + F2].round(2).to_string(index=False))
    from sklearn.metrics import roc_auc_score
    def aucs(lbl):
        return {c: roc_auc_score(lbl, D[c].fillna(D[c].median())) for c in F2}
    A0 = aucs(D.false.values)
    print("\nfeature   median TRUE  median FALSE  AUC(false)")
    for c in F2:
        print("%-8s %10.3f %12.3f %10.2f" % (c, D[D.false == 0][c].median(), D[D.false == 1][c].median(), A0[c]))
    obs = max(abs(v - .5) for v in A0.values())
    rng = np.random.default_rng(20261005); cnt = 0
    for _ in range(10000):
        lab_p = D.false.values.copy()
        for f in ("F13", "F47"):
            mm = (D.farm == f).values; lab_p[mm] = rng.permutation(lab_p[mm])
        if lab_p.sum() == 0:
            continue
        cnt += max(abs(v - .5) for v in aucs(lab_p).values()) >= obs
    best = max(A0, key=lambda c: abs(A0[c] - .5))
    p = (cnt + 1) / 10001
    print("\nbest %s AUC %.2f, family-wise p %.4f" % (best, A0[best], p))
    print("HK1 clue:", p < .05)


if __name__ == "__main__":
    main()
