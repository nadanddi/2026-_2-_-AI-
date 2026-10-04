# -*- coding: utf-8 -*-
"""EC stage-3 HG1: high-EC gate on top of SG2 (6.310 PASS), judged ONCE by the pass-2-only
protocol (6.279).  Fixed before running; 2026-10-05 집 클로드.
Evidence (HK0, 6.312): on high days the hour-causal reference anchor knows the SIZE
(Spearman(a1 - pm, y - pm) = .60, anchor mean 1.53 vs label 1.57 vs model 1.23) but SG2's
.30 guard blocks it.  HK0's pre-set clue FAILED (HGB: all rows -6.2 %, pass-2 -13.3 %,
high -12.3 %, but normal days +5.4 % and P .095), mostly from two pass-2 normal days whose
two anchors were both high (F13 231 / 233).  This run asks the protocol directly.
Candidate (HGB, unchanged from HK0): SG2 everywhere, except rows where the model's
today-so-far mean pm_h >= 0.9 AND the best two anchors are both >= 1.0:
pred = p + 0.5 (mean(a1, a2) - pm_h)  (no .30 guard).  Pass-2 rows only.
Baseline: R3S + SG2 (same new seeds 29 / 909 / 1111).  Reference / anchors / calendar /
signatures exactly as SG2 (training records only, either side, hour-causal query).
Sets (pass-2 rows): DIAG10, DIAG10z fresh layout (8-record chunks: chunk = day // 8,
fold = chunk % 10, +-1 purge, lock excluded; never used before), EL1.
PASS iff every seed x all three sets improve over R3S + SG2 AND seed-mean block
bootstrap (farm x 5-day, 20000) P(worse) on DIAG10z pass-2 rows < .025.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_HG1_high_gate_anchor_v1.py
"""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp1)
dc5, dc4, p3, core = dp1.dc5, dp1.dc4, dp1.p3, dp1.core
SEEDS = (29, 909, 1111)
CK = os.path.join(env.LOCAL, "hg1_ckpt")
W = ["out_temp", "out_hum", "out_rad", "out_wspd"]
SIGC = ["n_t", "n_h", "n_c", "d_t", "d_c", "mx_t", "th", "he", "co", "sh", "ve", "fo", "cf"]


def prepare_structure():
    R = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).sort_values(["farm", "day"]).reset_index(drop=True)
    R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
    X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W)
    for f in ("F13", "F47"):
        m = WV.index.get_level_values(0) == f; p1 = m & (WV.index.get_level_values(1) < 179)
        for v in W:
            mu, sd = np.nanmean(WV.loc[p1, v].values), np.nanstd(WV.loc[p1, v].values)
            WV.loc[m, v] = (WV.loc[m, v].values - mu) / sd
    hrs = WV.columns.get_level_values(1)
    # signatures at every hour h (hours 0..h), z per farm per h
    SIG = {}
    for h in range(24):
        Xh = X[X.hour <= h]
        def sig(g):
            n = g[g.hour <= 5]; d = g[(g.hour >= 10) & (g.hour <= 15)]
            v = g.act_vent.fillna(0).values
            return pd.Series(dict(n_t=n.in_temp.mean(), n_h=n.in_hum.mean(), n_c=n.in_co2.mean(),
                                  d_t=d.in_temp.mean() if len(d) else np.nan, d_c=d.in_co2.mean() if len(d) else np.nan,
                                  mx_t=g.in_temp.max(), th=(g.act_thermal > 0).sum(), he=(g.act_heating > 0).sum(),
                                  co=(g.act_co2 > 0).sum(), sh=(g.act_shade > 0).sum(), ve=(v > 0).sum(),
                                  fo=(g.hour.values[v > 0].min() if (v > 0).any() else 24), cf=n.act_circfan.mean()))
        S = Xh.groupby(["farm", "day"]).apply(sig)
        # raw signatures; z-scaling uses reference (training) records only, inside correction()
        SIG[h] = S
    return R, WV, hrs, SIG


def ref_calendar(R, WV, ref):
    """ref: set of (farm, day) training reference records.  Returns dict -> date."""
    cal = {}
    for f in ("F13", "F47"):
        G = R[(R.farm == f)].sort_values("day")
        P1 = [d for d in G.day if d < 179 and (f, d) in ref]

        def tw(a, b):
            return np.sqrt(np.nanmean((WV.loc[(f, a)].values - WV.loc[(f, b)].values) ** 2)) <= .05
        g1 = [(a, b) for a, b in zip(P1[:-1], P1[1:]) if b - a == 1]
        rho = np.mean([not tw(a, b) for a, b in g1]) if g1 else .65
        c = 0.0
        for k, d in enumerate(P1):
            if k:
                c = c if tw(P1[k - 1], d) else c + max(1, round((d - P1[k - 1]) * rho))
            cal[(f, d)] = c
        A1 = WV.loc[[(f, d) for d in P1]].values
        prev = None
        for d in [d for d in G.day if d >= 179 and (f, d) in ref]:
            dist = np.sqrt(np.nanmean((A1 - WV.loc[(f, d)].values) ** 2, axis=1)); ok = dist <= .05
            if ok.any():
                cal[(f, d)] = float(np.mean([cal[(f, e)] for e, o in zip(P1, ok) if o]))
            else:
                cal[(f, d)] = (cal[(f, prev)] + 0.1) if prev is not None else 0.0
            prev = d
    return cal


def correction(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal):
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
            if pm >= .9 and a1 >= 1.0 and np.isfinite(a2) and a2 >= 1.0:
                hg[j] = rr[pcol] + 0.5 * ((a1 + a2) / 2 - pm)
    return out, hg


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    FS0 = [c for c in core.FULL if c != "day"] + ["season"]; BS0 = [c for c in core.BASE if c != "day"] + ["season"]
    R, WV, hrs, SIG = prepare_structure()
    labset = set(zip(lab.farm, lab.day))
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean()
    days = lab[["farm", "day"]].drop_duplicates()
    folds = [x for x in fds if x[0] == "DIAG10"]
    for k in range(10):
        folds.append(("DIAG10z", k, {(f, int(d)) for f, d in zip(days.farm, days.day) if (d // 8) % 10 == k and (f, int(d)) not in lock}))
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
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        ref = {(f, int(d)) for f, d in labset if (f, int(d)) not in vd}
        cal = ref_calendar(R, WV, ref)
        for s in SEEDS:
            frame["base_%d" % s] = dc5.r3(tr, va, s, FS0, BS0)
            frame["sgb_%d" % s], frame["sg_%d" % s] = correction(frame, "base_%d" % s, vd, lockd, ec, R, WV, hrs, SIG, ref, cal)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O = O[O.day >= 179].copy()
    O.to_csv(os.path.join(env.LOCAL, "ec3_HG1_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    ok = True
    print("\nPASS-2 rows: R3S+SG2 -> HG1 (new seeds)")
    for v in ("DIAG10", "DIAG10z", "EL1"):
        G = O[O.validator == v]; cells = []
        for s in SEEDS:
            a, b = r(G["sgb_%d" % s] - G.sub_ec), r(G["sg_%d" % s] - G.sub_ec)
            ok &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-8s days %2d  %s" % (v, G.groupby(["farm", "day"]).ngroups, "  ".join(cells)))
    T = O[O.validator == "DIAG10z"].copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
    bm, cm = T[["sgb_%d" % s for s in SEEDS]].mean(axis=1), T[["sg_%d" % s for s in SEEDS]].mean(axis=1)
    dd = (cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2
    cl = dd.groupby(T.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
    idx = np.random.default_rng(20261004).integers(0, len(sm), (20000, len(sm)))
    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    print("  DIAG10z pass-2 seed-mean RMSE %.4f -> %.4f  P(worse) %.4f" % (r(bm - T.sub_ec), r(cm - T.sub_ec), p))
    for v in ("DIAG10", "DIAG10z", "EL1"):
        A = O[O.validator == v].copy(); A["dm"] = A.groupby(["farm", "day"]).sub_ec.transform("mean")
        for nm, m in (("normal", A.dm < 1), ("high", A.dm >= 1)):
            G = A[m]
            print("  %-8s %s days: %s" % (v, nm, "  ".join("s%d %.4f->%.4f" % (s, r(G["sgb_%d" % s] - G.sub_ec), r(G["sg_%d" % s] - G.sub_ec)) for s in SEEDS)))
    print("\nHG1 decision:", "PASS" if ok and p < 0.025 else "FAIL", "(all seeds x sets better %s, P %.4f)" % (ok, p))


if __name__ == "__main__":
    main()
