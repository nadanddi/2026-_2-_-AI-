# -*- coding: utf-8 -*-
"""EC stage-3 AF0b (diagnostic, fixed before running; 2026-10-05 집 클로드).
AF0 (6.317): anchor features cut high-day error 14-25 % but normal days got 35-54 % worse,
because anchors from the OTHER 동 cannot be told apart.  AF0b adds 동-match trust features.
동 probability pB (prob. that a record is the pair 'second' = mostly 동 B), hour-causal:
  labels: consecutive record pairs (d, d+1) BOTH in the training reference whose full-day
          outdoor weather is an exact twin (z-RMSE <= .05): d = first (0), d+1 = second (1);
  features at hour h: expanding means over hours 0..h of in_temp, in_hum, in_co2 and the
          7 actuators (ST8 list), z from training reference rows; one logistic regression
          per hour (C = 1), fitted on reference pair records only (no test / validation
          inputs);
  query row at hour h: pB = 1 if its outdoor hours 0..h equal the previous record's
          (causal pair second), else the hour-h model; anchor record (reference): its own
          full-day value (hour 23 model; 1 / 0 if it is a known reference pair second / first).
New features (on top of AF0's 7): af_pq (query pB), af_dd1 / af_dd2 = |pB_q - pB_anchor|
for the best two anchors, af_a1s / af_d1s = best anchor among those with |dpB| < .5.
Same folds, seeds (7 / 101 / 2024), exclusion (+-3 record days for training rows) as AF0;
R3S and R3S + SG2 columns re-used from AF0's checkpoints (same folds / seeds).
Clue (fixed, as AF0): beats R3S + SG2 for all 3 seeds on DIAG10 and EL1 pass-2 rows, high
days better, normal days not worse by > 2 %.  Also reports pB accuracy on reference pairs
(leave-fold-out) and the share of best anchors with |dpB| < .5.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_AF0b_anchor_dong_trust_v1.py
"""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("af0", os.path.join(HERE, "ec3_AF0_anchor_features_v1.py"))
af0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(af0)
sg2, dc5, dc4, p3, core = af0.sg2, af0.dc5, af0.dc4, af0.p3, af0.core
SEEDS = af0.SEEDS
CK = os.path.join(env.LOCAL, "af0b_ckpt"); CK0 = af0.CK
CC = ["in_temp", "in_hum", "in_co2", "act_heating", "act_vent", "act_thermal", "act_shade", "act_circfan", "act_fog", "act_co2"]
NEW = ["af_pq", "af_dd1", "af_dd2", "af_a1s", "af_d1s"]


def load_X():
    X = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id"] + CC) for f in ("train_X.csv", "test_X.csv")])
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    X = X.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
    g = X.groupby(["farm", "day"])
    for c in CC:
        X["em_" + c] = g[c].transform(lambda s: s.expanding().mean())
    return X


def dong_model(X, WV, hrs, ref):
    """Returns pq(f, d, h) and pa(f, d) functions and pair-label accuracy info."""
    EM = ["em_" + c for c in CC]
    lab = {}
    for f in ("F13", "F47"):
        E = sorted(d for ff, d in ref if ff == f)
        for a, b in zip(E[:-1], E[1:]):
            if b == a + 1 and np.sqrt(np.nanmean((WV.loc[(f, a)].values - WV.loc[(f, b)].values) ** 2)) <= .05:
                lab[(f, a)] = 0; lab[(f, b)] = 1
    key = list(zip(X.farm, X.day))
    inref = np.array([k in ref for k in key])
    mu, sd = X.loc[inref, EM].mean(), X.loc[inref, EM].std().replace(0, 1)
    Z = ((X[EM] - mu) / sd).fillna(0).values
    y = np.array([lab.get(k, -1) for k in key])
    P = np.full(len(X), np.nan)
    for h in range(24):
        hm = (X.hour == h).values
        tm = hm & (y >= 0)
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(Z[tm], y[tm])
        P[hm] = clf.predict_proba(Z[hm])[:, 1]
    T = pd.Series(P, index=pd.MultiIndex.from_arrays([X.farm, X.day, X.hour]))
    hcols = hrs
    ALLD = {f: set(X[X.farm == f].day) for f in ("F13", "F47")}

    def pq(f, d, h):
        if (d - 1) in ALLD[f]:
            cm = np.asarray(hcols <= h)
            a, b = WV.loc[(f, d)].values[cm], WV.loc[(f, d - 1)].values[cm]
            if np.sqrt(np.nanmean((a - b) ** 2)) <= .05 and h >= 2:
                return 1.0
        return float(T[(f, d, h)])

    def pa(f, d):
        return float(lab[(f, d)]) if (f, d) in lab else float(T[(f, d, 23)])
    return pq, pa, len(lab)


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    FS0 = [c for c in core.FULL if c != "day"] + ["season"]; BS0 = [c for c in core.BASE if c != "day"] + ["season"]
    R, WV, hrs, SIG = sg2.prepare_structure()
    X = load_X()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    dg = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).set_index(["farm", "day"]).dong
    folds = [x for x in fds if x[0] == "DIAG10"]
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    acc = []
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
        pq, pa, npair = dong_model(X, WV, hrs, ref)
        tq = [(f, int(d)) for f, d in tdays.itertuples(index=False)]
        vq_ = [(f, int(d)) for f, d in vdays[["farm", "day"]].itertuples(index=False)]
        # pB accuracy vs st_dong_assign on validation records (descriptive)
        for f, d in vq_:
            acc.append(((pq(f, d, 23) >= .5) == (dg.get((f, d)) == "B")))
        A = af0.anchor_features(tq + vq_, set(tq), R, WV, hrs, SIG, ec, ref, cal, lockd)
        # recompute best anchors with dong trust (same search, extra outputs)
        A = add_trust(A, tq, R, WV, hrs, SIG, ec, ref, cal, lockd, pq, pa)
        tr = tr.merge(A, on=["farm", "day", "hour"], how="left").set_index(tr.index)
        va = va.merge(A, on=["farm", "day", "hour"], how="left").set_index(va.index)
        frame = pd.read_csv(os.path.join(CK0, "%s_%d.csv" % (name, i)))
        assert (frame.row_id.values == va.row_id.values).all()
        for c in NEW:
            frame[c] = va[c].values
        for s in SEEDS:
            frame["afb_%d" % s] = dc5.r3(tr, va, s, FS0 + af0.AF + NEW, BS0 + af0.AF + NEW)
        frame.to_csv(path, index=False)
        print("%s/%d done (reference pairs %d)" % (name, i, npair), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O = O[O.day >= 179].copy()
    O.to_csv(os.path.join(env.LOCAL, "ec3_AF0b_all.csv"), index=False)
    if acc:
        print("pB(23h) >= .5 agrees with st_dong_assign 동B on validation records: %.2f (n %d)" % (np.mean(acc), len(acc)))
    print("best-anchor |dpB| < .5 share on pass-2 rows: %.2f" % (O.af_dd1 < .5).mean())
    O["dm"] = O.groupby(["validator", "farm", "day"]).sub_ec.transform("mean")
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    clue = True
    print("\nPASS-2 rows: R3S+SG2 | AF0 | AF0b")
    for v in ("DIAG10", "EL1"):
        for nm in ("all", "normal", "high"):
            G = O[O.validator == v]
            G = G[G.dm < 1] if nm == "normal" else (G[G.dm >= 1] if nm == "high" else G)
            cells = []
            for s in SEEDS:
                b, a0, c = r(G["sg_%d" % s] - G.sub_ec), r(G["af_%d" % s] - G.sub_ec), r(G["afb_%d" % s] - G.sub_ec)
                cells.append("s%d %.4f | %.4f | %.4f (%+.1f%%)" % (s, b, a0, c, 100 * (c / b - 1)))
                clue &= (c <= 1.02 * b) if nm == "normal" else (c < b)
            print("  %-7s %-6s %s" % (v, nm, "  ".join(cells)))
    print("\nAF0b clue:", clue)


def add_trust(A, tq, R, WV, hrs, SIG, ec, ref, cal, lock, pq, pa):
    """Second pass of the AF0 search returning 동-trust features (identical candidate rule)."""
    trq = set(tq)
    alld = {f: sorted(R[R.farm == f].day) for f in ("F13", "F47")}
    roles = R.set_index(["farm", "day"]).role
    pre = {}
    for f in ("F13", "F47"):
        E = [e for e in alld[f] if (f, e) in ref]
        pre[f] = dict(E=np.array(E), W=WV.loc[[(f, e) for e in E]].values, cal=np.array([cal[(f, e)] for e in E]),
                      lab=np.array([(f, e) in ec.index and (f, e) not in lock for e in E]),
                      ec=np.array([ec.get((f, e), np.nan) for e in E]), pb=np.array([pa(f, e) for e in E]))
    zs = {}
    for h in range(24):
        S = SIG[h].astype(float)
        for f in ("F13", "F47"):
            RS = S.loc[[(f, e) for e in pre[f]["E"]]]
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            zs[(f, h)] = (mu, sd, (RS.values - mu) / sd)
    cache = {}

    def full_date(f, d):
        if (f, d) in cal:
            return cal[(f, d)]
        if (f, d) in cache:
            return cache[(f, d)]
        P = pre[f]; dist = np.sqrt(np.nanmean((P["W"] - WV.loc[(f, d)].values) ** 2, axis=1)); ok = dist <= .05
        if ok.any():
            t = float(P["cal"][ok].mean())
        else:
            i = alld[f].index(d)
            t = (full_date(f, alld[f][i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        cache[(f, d)] = t
        return t
    rows = []
    for (f, d), G in A.groupby(["farm", "day"]):
        P = pre[f]; keep = np.ones(len(P["E"]), bool)
        if (f, d) in trq:
            keep &= np.abs(P["E"] - d) > af0.EXCL
        i = alld[f].index(d); b = WV.loc[(f, d)].values
        for h in range(24):
            row = dict(farm=f, day=d, hour=h); q_pb = pq(f, d, h); row["af_pq"] = q_pb
            cq = None
            if h >= 5:
                cm = np.asarray(hrs <= h)
                dist = np.sqrt(np.nanmean((P["W"][:, cm] - b[cm]) ** 2, axis=1)); ok = (dist <= .05) & keep
                if ok.any():
                    cq = float(P["cal"][ok].mean())
            if cq is None:
                cq = (full_date(f, alld[f][i - 1]) + 0.1) if i else 0.0
            m = keep & P["lab"] & (np.abs(P["cal"] - cq) <= 3) & (P["cal"] != cq)
            if m.any():
                mu, sd, Zr = zs[(f, h)]
                q = (SIG[h].loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
                sc = np.sqrt(((np.nan_to_num(Zr[m][:, use]) - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(P["cal"][m] - cq)
                o = np.argsort(sc); pb = P["pb"][m]; ev = P["ec"][m]
                dd = np.abs(pb - q_pb)
                row["af_dd1"] = dd[o[0]]
                if len(o) > 1:
                    row["af_dd2"] = dd[o[1]]
                ms = [j for j in o if dd[j] < .5]
                if ms:
                    row["af_a1s"], row["af_d1s"] = ev[ms[0]], sc[ms[0]]
            rows.append(row)
    return A.merge(pd.DataFrame(rows), on=["farm", "day", "hour"], how="left")


if __name__ == "__main__":
    main()
