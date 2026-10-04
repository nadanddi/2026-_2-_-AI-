# -*- coding: utf-8 -*-
"""EC stage-3 AF0 (diagnostic, fixed before running; 2026-10-05 집 클로드).
Model-wide review after the organizer's 10-05 clarification (6.309): instead of hand-made
post-processing rules on reference anchors (SG2 guard .30 PASSED 6.310; HG1 gate FAILED
6.313; HK1 found anchor distance informative but not significant 6.316), give the R3
members the reference-anchor information as FEATURES and let them learn how much to
trust it ("a model that stores and references the training data", allowed).
Anchor features per row (hour-causal; the SG2 search): calendar from training reference
records only; query date at hour h = outdoor twin over hours 0..h (h >= 5) among the
reference, else previous record's date + 0.1; candidates = labelled reference records of
the same farm, |cal - cal_q| <= 3, cal != cal_q, not locked; 13-value signature on hours
0..h, z from reference records; score = distance + .15 |dcal|.
  af_a1, af_a2  label day means of the best two anchors
  af_d1, af_d2  their scores
  af_hi5        share of EC >= 1 among the best five
  af_m5         mean label of the best five
  af_n          number of candidates
TRAINING rows: the record's own +-3 record days are removed from the twin search and the
candidates (evaluation days sit in 5-10 day blocks with no labelled neighbours, so
training rows must not see unrealistically close anchors).  Validation rows: reference =
labelled records outside the validation set (as at test time).
Candidate AF: R3 with FS/BS + 7 af_ features (seeds 7 / 101 / 2024).  Comparison:
R3S (same seeds, refitted here) and R3S + SG2 correction.
Sets: DIAG10 folds with pass-2 days, EL1; judged on pass-2 rows.
Clue (fixed): AF beats R3S + SG2 for all 3 seeds on DIAG10 pass-2 rows AND on EL1, with
pass-2 high days better and pass-2 normal days not worse than R3S + SG2 by > 2 %.
Run (checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_AF0_anchor_features_v1.py
"""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
dc5, dc4, p3, core = sg2.dc5, sg2.dc4, sg2.p3, sg2.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "af0_ckpt")
AF = ["af_a1", "af_a2", "af_d1", "af_d2", "af_hi5", "af_m5", "af_n"]
EXCL = 3


def anchor_features(queries, train_q, R, WV, hrs, SIG, ec, ref, cal, lock):
    """queries: list of (farm, day); train_q: set of those that are training rows (exclusion).
    Returns DataFrame farm, day, hour + AF."""
    roles = R.set_index(["farm", "day"]).role
    alld = {f: sorted(R[R.farm == f].day) for f in ("F13", "F47")}
    cache = {}
    out = []
    pre = {}
    for f in ("F13", "F47"):
        E = [e for e in alld[f] if (f, e) in ref]
        pre[f] = dict(E=np.array(E), W=WV.loc[[(f, e) for e in E]].values, cal=np.array([cal[(f, e)] for e in E]),
                      lab=np.array([(f, e) in ec.index and (f, e) not in lock for e in E]),
                      ec=np.array([ec.get((f, e), np.nan) for e in E]))
    hmask = {h: np.asarray(hrs <= h) for h in range(24)}
    zs = {}
    for h in range(24):
        S = SIG[h].astype(float)
        for f in ("F13", "F47"):
            RS = S.loc[[(f, e) for e in pre[f]["E"]]]
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            zs[(f, h)] = (mu, sd, (RS.values - mu) / sd)

    def full_date(f, d):
        if (f, d) in cal:
            return cal[(f, d)]
        if (f, d) in cache:
            return cache[(f, d)]
        P = pre[f]; b = WV.loc[(f, d)].values
        dist = np.sqrt(np.nanmean((P["W"] - b) ** 2, axis=1)); ok = dist <= .05
        if ok.any():
            t = float(P["cal"][ok].mean())
        else:
            i = alld[f].index(d)
            t = (full_date(f, alld[f][i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        cache[(f, d)] = t
        return t

    for f, d in queries:
        P = pre[f]; keep = np.ones(len(P["E"]), bool)
        if (f, d) in train_q:
            keep &= np.abs(P["E"] - d) > EXCL
        i = alld[f].index(d)
        b = WV.loc[(f, d)].values
        for h in range(24):
            row = dict(farm=f, day=d, hour=h)
            cq = None
            if h >= 5:
                cm = hmask[h]
                dist = np.sqrt(np.nanmean((P["W"][:, cm] - b[cm]) ** 2, axis=1)); ok = (dist <= .05) & keep
                if ok.any():
                    cq = float(P["cal"][ok].mean())
            if cq is None:
                cq = (full_date(f, alld[f][i - 1]) + 0.1) if i else 0.0
            m = keep & P["lab"] & (np.abs(P["cal"] - cq) <= 3) & (P["cal"] != cq)
            row["af_n"] = float(m.sum())
            if m.any():
                mu, sd, Zr = zs[(f, h)]
                q = (SIG[h].loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
                C = np.nan_to_num(Zr[m][:, use])
                sc = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(P["cal"][m] - cq)
                o = np.argsort(sc); ev = P["ec"][m]
                row["af_a1"], row["af_d1"] = ev[o[0]], sc[o[0]]
                if len(o) > 1:
                    row["af_a2"], row["af_d2"] = ev[o[1]], sc[o[1]]
                row["af_hi5"] = float((ev[o[:5]] >= 1).mean()); row["af_m5"] = float(ev[o[:5]].mean())
            out.append(row)
    return pd.DataFrame(out)


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    FS0 = [c for c in core.FULL if c != "day"] + ["season"]; BS0 = [c for c in core.BASE if c != "day"] + ["season"]
    R, WV, hrs, SIG = sg2.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    folds = [x for x in fds if x[0] == "DIAG10"]
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    for name, i, vd in folds:
        if not any(d >= 179 for f, d in vd):
            continue
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        ref = {(f, int(d)) for f, d in labset if (f, int(d)) not in vd}
        cal = sg2.ref_calendar(R, WV, ref)
        tq = [(f, int(d)) for f, d in tdays.itertuples(index=False)]
        vq_ = [(f, int(d)) for f, d in vdays[["farm", "day"]].itertuples(index=False)]
        A = anchor_features(tq + vq_, set(tq), R, WV, hrs, SIG, ec, ref, cal, lockd)
        tr = tr.merge(A, on=["farm", "day", "hour"], how="left").set_index(tr.index)
        va = va.merge(A, on=["farm", "day", "hour"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"] + AF].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["base_%d" % s] = dc5.r3(tr, va, s, FS0, BS0)
            frame["af_%d" % s] = dc5.r3(tr, va, s, FS0 + AF, BS0 + AF)
            frame["sg_%d" % s] = sg2.correction(frame, "base_%d" % s, vd, lockd, ec, R, WV, hrs, SIG, ref, cal)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O = O[O.day >= 179].copy()
    O.to_csv(os.path.join(env.LOCAL, "ec3_AF0_all.csv"), index=False)
    O["dm"] = O.groupby(["validator", "farm", "day"]).sub_ec.transform("mean")
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    clue = True
    print("\nPASS-2 rows (seeds 7/101/2024): R3S | R3S+SG2 | AF")
    for v in ("DIAG10", "EL1"):
        for nm, m in (("all", None), ("normal", "n"), ("high", "h")):
            G = O[O.validator == v]
            if m == "n":
                G = G[G.dm < 1]
            elif m == "h":
                G = G[G.dm >= 1]
            cells = []
            for s in SEEDS:
                a, b, c = r(G["base_%d" % s] - G.sub_ec), r(G["sg_%d" % s] - G.sub_ec), r(G["af_%d" % s] - G.sub_ec)
                cells.append("s%d %.4f | %.4f | %.4f (%+.1f%% vs SG2)" % (s, a, b, c, 100 * (c / b - 1)))
                if m is None:
                    clue &= c < b
                elif m == "h":
                    clue &= c < b
                else:
                    clue &= c <= 1.02 * b
            print("  %-7s %-6s %s" % (v, nm, "  ".join(cells)))
    print("\nAF0 clue:", clue)


if __name__ == "__main__":
    main()
