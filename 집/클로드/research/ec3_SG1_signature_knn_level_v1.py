# -*- coding: utf-8 -*-
"""EC stage-3 SG1: guarded control-signature kNN day-level correction (from LB4-LB9,
catalog 6.303-6.307; LB9 GU1 passed its diagnostic clue), judged ONCE by the pass-2-only
protocol (catalog 6.279).  Fixed before running; 2026-10-04 집 클로드.
Hour-causal version (row at hour h of a pass-2 record uses only that record's hours
0..h and EARLIER records of the same farm; labels: public train_y of earlier records):
  calendar  every record's full-day calendar CALF = LB8 rule G (pass-1: record order,
            pairs share a date; pass-2: exact 24 h outdoor twin among earlier records ->
            twin's date, else previous record + 0.1, pair 'second' + 0); for the query
            row at hour h: twin over outdoor hours 0..h (h >= 5, z-RMSE <= .05) among
            earlier records -> their CALF mean; else previous record's CALF + 0.1.
  signature 13 values of LB4 computed on hours 0..h for the query AND for candidates
            (same hours), z per farm per h; distance over dims available for the query.
  candidates labelled records with day < query day, not in the validation set, not
            locked, |CALF - cal_q| <= 3, CALF != cal_q; score = distance + .15 |dcal|.
  level     a1 = label day mean of the best candidate; pm_h = mean of the model's
            predictions for hours 0..h of the query day; if |a1 - pm_h| <= .30:
            pred = p + 0.5 (a1 - pm_h), else p.  Pass-2 rows only.
Baseline: R3S (FS/BS, DC4 season) with new seeds 17 / 606 / 7070; candidate = the same
predictions + correction (no refit).
Sets (pass-2 rows): DIAG10 original layout, DIAG10x fresh layout (7-record chunks:
chunk = day // 7, fold = chunk % 10, +-1 purge, lock excluded; never used before), EL1.
PASS iff every seed x all three sets improve AND seed-mean block bootstrap (farm x
5-day, 20000) P(worse) on DIAG10x pass-2 rows < .025.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_SG1_signature_knn_level_v1.py
"""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp1)
dc5, dc4, p3, core = dp1.dc5, dp1.dc4, dp1.p3, dp1.core
SEEDS = (17, 606, 7070)
CK = os.path.join(env.LOCAL, "sg1_ckpt")
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
    CALF, PREV = {}, {}
    for f in ("F13", "F47"):
        G = R[R.farm == f].sort_values("day"); prev = None
        for r in G.itertuples():
            PREV[(f, r.day)] = prev
            if r.day < 179:
                CALF[(f, r.day)] = float(r.date)
            else:
                E = [d for d in G.day if d < r.day]
                dist = np.sqrt(np.nanmean((WV.loc[[(f, d) for d in E]].values - WV.loc[(f, r.day)].values) ** 2, axis=1))
                ok = dist <= .05
                CALF[(f, r.day)] = float(np.mean([CALF[(f, d)] for d, o in zip(E, ok) if o])) if ok.any() else \
                    CALF[prev] + (0.0 if r.role == "second" else 0.1)
            prev = (f, r.day)

    def cal_q(f, d, h):
        if h >= 5:
            E = [e for e in R[R.farm == f].day if e < d]
            cols = np.asarray(hrs <= h)
            A = WV.loc[[(f, e) for e in E]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
            dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1)); ok = dist <= .05
            if ok.any():
                return float(np.mean([CALF[(f, e)] for e, o in zip(E, ok) if o]))
        return CALF[PREV[(f, d)]] + 0.1

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
        for f in ("F13", "F47"):
            m = S.index.get_level_values(0) == f
            sd = S.loc[m].std().replace(0, np.nan)
            S.loc[m] = ((S.loc[m] - S.loc[m].mean()) / sd).values
        SIG[h] = S
    return R, CALF, cal_q, SIG


def correction(frame, pcol, vd, lock, ec, R, CALF, cal_q, SIG):
    """frame: validation rows (farm, day, hour, pcol); returns corrected predictions (pass-2 rows only)."""
    out = frame[pcol].values.copy()
    for (f, d), idx in frame.groupby(["farm", "day"]).groups.items():
        if d < 179:
            continue
        G = R[(R.farm == f) & (R.day < d)]
        G = G[[(f, e) in ec.index and (f, e) not in vd and (f, e) not in lock for e in G.day]]
        rows = frame.loc[idx].sort_values("hour")
        cum = rows[pcol].expanding().mean().values
        for k, (ii, rr) in enumerate(rows.iterrows()):
            h = int(rr.hour); cq = cal_q(f, d, h)
            calc = np.array([CALF[(f, e)] for e in G.day])
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            if not m.any():
                continue
            Gm = G[m]; S = SIG[h]
            q = S.loc[(f, d)].values.astype(float); use = ~np.isnan(q)
            C = np.nan_to_num(S.loc[list(zip(Gm.farm, Gm.day))].values.astype(float)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            b = Gm.iloc[int(np.argmin(dist))]
            a1 = ec[(f, b.day)]; pm = cum[k]
            if abs(a1 - pm) <= .30:
                out[frame.index.get_loc(ii)] = rr[pcol] + 0.5 * (a1 - pm)
    return out


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    FS0 = [c for c in core.FULL if c != "day"] + ["season"]; BS0 = [c for c in core.BASE if c != "day"] + ["season"]
    R, CALF, cal_q, SIG = prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean()
    days = lab[["farm", "day"]].drop_duplicates()
    folds = [x for x in fds if x[0] == "DIAG10"]
    for k in range(10):
        folds.append(("DIAG10x", k, {(f, int(d)) for f, d in zip(days.farm, days.day) if (d // 7) % 10 == k and (f, int(d)) not in lock}))
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
        for s in SEEDS:
            frame["base_%d" % s] = dc5.r3(tr, va, s, FS0, BS0)
            frame["sg_%d" % s] = correction(frame, "base_%d" % s, vd, lockd, ec, R, CALF, cal_q, SIG)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O = O[O.day >= 179].copy()
    O.to_csv(os.path.join(env.LOCAL, "ec3_SG1_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    ok = True
    print("\nPASS-2 rows: refitted R3S -> SG1 (new seeds)")
    for v in ("DIAG10", "DIAG10x", "EL1"):
        G = O[O.validator == v]; cells = []
        for s in SEEDS:
            a, b = r(G["base_%d" % s] - G.sub_ec), r(G["sg_%d" % s] - G.sub_ec)
            ok &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-8s days %2d  %s" % (v, G.groupby(["farm", "day"]).ngroups, "  ".join(cells)))
    T = O[O.validator == "DIAG10x"].copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
    bm, cm = T[["base_%d" % s for s in SEEDS]].mean(axis=1), T[["sg_%d" % s for s in SEEDS]].mean(axis=1)
    dd = (cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2
    cl = dd.groupby(T.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
    idx = np.random.default_rng(20261004).integers(0, len(sm), (20000, len(sm)))
    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    print("  DIAG10x pass-2 seed-mean RMSE %.4f -> %.4f  P(worse) %.4f" % (r(bm - T.sub_ec), r(cm - T.sub_ec), p))
    for v in ("DIAG10", "DIAG10x", "EL1"):
        A = O[O.validator == v].copy(); A["dm"] = A.groupby(["farm", "day"]).sub_ec.transform("mean")
        for nm, m in (("normal", A.dm < 1), ("high", A.dm >= 1)):
            G = A[m]
            print("  %-8s %s days: %s" % (v, nm, "  ".join("s%d %.4f->%.4f" % (s, r(G["base_%d" % s] - G.sub_ec), r(G["sg_%d" % s] - G.sub_ec)) for s in SEEDS)))
    print("\nSG1 decision:", "PASS" if ok and p < 0.025 else "FAIL", "(all seeds x sets better %s, P %.4f)" % (ok, p))


if __name__ == "__main__":
    main()
