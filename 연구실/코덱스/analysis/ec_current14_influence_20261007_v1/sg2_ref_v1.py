"""SG2 post-processing for submission_14 (EC), 2026-10-05 집 클로드.
Guarded control-signature reference correction (catalog 6.310 PASS under the pass-2-only
protocol; 6.311 same direction on the submission_13 configuration).

For an evaluation row at hour h of record (farm, day):
  * only that record's hours 0..h and EARLIER records of the same farm are used as inputs;
  * the stored training data (train_X / train_y of the same farm, any date) is referenced
    (organizer clarification 2026-10-05: storing and referencing training data is allowed);
  * nothing is fitted on evaluation inputs: weather standardization uses pass-1 training
    records, signature standardization uses training reference records.
Steps (identical to research/ec3_SG2_reference_knn_level_v1.py):
  1. reference calendar from training records only (pass-1 record order with exact
     outdoor-weather twins sharing a date; pass-2 training records dated by their exact
     twin among pass-1 training records, else previous pass-2 training record + 0.1);
  2. query date at hour h: exact twin over outdoor hours 0..h (h >= 5, z-RMSE <= .05)
     among training records -> their date mean; otherwise the previous record's date
     + 0.1 (previous record: training -> its date, else its own full-day twin date or
     recursively previous + 0.1, +0 if it is the second of an exact-twin pair);
  3. candidates: labelled training records of the same farm with |date - date_q| <= 3
     and date != date_q; 13-value control signature over hours 0..h (z from training
     records at the same h); score = RMS distance + .15 |date difference|;
  4. a1 = daily mean label of the best candidate; pm_h = mean of the model's
     predictions for hours 0..h of the query record; if |a1 - pm_h| <= .30:
     pred = p + 0.5 (a1 - pm_h); else unchanged.  Pass-2 records only (day >= 179).
"""
import numpy as np
import pandas as pd

W = ["out_temp", "out_hum", "out_rad", "out_wspd"]


def _ident(df):
    d = df.copy()
    d["farm"], d["day"], d["hour"] = d.row_id.str[:3], d.row_id.str[4:7].astype(int), d.row_id.str[8:10].astype(int)
    return d


def prepare(x_all, ref):
    """x_all: train_X + test_X rows of F13/F47 (row_id, W, in_*, act_*)."""
    X = _ident(x_all).sort_values(["farm", "day", "hour"]).reset_index(drop=True)
    WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W)
    for f in ("F13", "F47"):
        m = WV.index.get_level_values(0) == f; p1 = m & (WV.index.get_level_values(1) < 179) & np.array([(str(ff), int(dd)) in ref for ff, dd in WV.index])
        for v in W:
            mu, sd = np.nanmean(WV.loc[p1, v].values), np.nanstd(WV.loc[p1, v].values)
            WV.loc[m, v] = (WV.loc[m, v].values - mu) / sd
    hrs = WV.columns.get_level_values(1)
    days = {f: sorted(X[X.farm == f].day.unique().tolist()) for f in ("F13", "F47")}
    # pair 'second' = same rule as research/st4 (greedy left to right: record d is second if
    # d-1 is not itself a second and their raw 24 h outdoor weather is identical); uses d and
    # earlier records only
    raw = X.pivot_table(index=["farm", "day"], columns="hour", values=W)
    second = {}
    for f in ("F13", "F47"):
        for d in days[f]:
            s = False
            if (d - 1) in days[f] and not second[(f, d - 1)]:
                a, b = raw.loc[(f, d - 1)].values, raw.loc[(f, d)].values; ok = ~np.isnan(a) & ~np.isnan(b)
                s = bool(ok.sum() >= 80 and np.max(np.abs(a[ok] - b[ok])) < 1e-9)
            second[(f, d)] = s
    SIG = {}
    for h in range(24):
        Xh = X[X.hour <= h]

        def sig(g):
            n = g[g.hour <= 5]; dd = g[(g.hour >= 10) & (g.hour <= 15)]
            v = g.act_vent.fillna(0).values
            return pd.Series(dict(n_t=n.in_temp.mean(), n_h=n.in_hum.mean(), n_c=n.in_co2.mean(),
                                  d_t=dd.in_temp.mean() if len(dd) else np.nan, d_c=dd.in_co2.mean() if len(dd) else np.nan,
                                  mx_t=g.in_temp.max(), th=(g.act_thermal > 0).sum(), he=(g.act_heating > 0).sum(),
                                  co=(g.act_co2 > 0).sum(), sh=(g.act_shade > 0).sum(), ve=(v > 0).sum(),
                                  fo=(g.hour.values[v > 0].min() if (v > 0).any() else 24), cf=n.act_circfan.mean()))
        SIG[h] = Xh.groupby(["farm", "day"]).apply(sig)
    return dict(WV=WV, hrs=hrs, days=days, second=second, SIG=SIG)


def ref_calendar(S, ref):
    WV, cal = S["WV"], {}
    for f in ("F13", "F47"):
        P1 = [d for d in S["days"][f] if d < 179 and (f, d) in ref]

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
        for d in [d for d in S["days"][f] if d >= 179 and (f, d) in ref]:
            dist = np.sqrt(np.nanmean((A1 - WV.loc[(f, d)].values) ** 2, axis=1)); ok = dist <= .05
            cal[(f, d)] = float(np.mean([cal[(f, e)] for e, o in zip(P1, ok) if o])) if ok.any() else ((cal[(f, prev)] + 0.1) if prev is not None else 0.0)
            prev = d
    return cal


def correct(frame, pred, S, ec, ref, cal):
    """frame: query rows (row_id ...), pred: model predictions aligned with frame.
    ec: Series (farm, day) -> labelled daily mean EC.  Returns corrected predictions."""
    F = _ident(frame[["row_id"]]).reset_index(drop=True); F["p"] = np.asarray(pred, float)
    out = F.p.values.copy()
    WV, hrs, SIG, days = S["WV"], S["hrs"], S["SIG"], S["days"]
    cache = {}

    def twin_date(f, d, h):
        E = [e for e in days[f] if (f, e) in ref]
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
            i = days[f].index(d)
            t = (full_date(f, days[f][i - 1]) + (0.0 if S["second"][(f, d)] else 0.1)) if i else 0.0
        cache[(f, d)] = t
        return t

    for (f, d), idx in F.groupby(["farm", "day"]).groups.items():
        if d < 179:
            continue
        E = [e for e in days[f] if (f, e) in ref and (f, e) in ec.index]
        calc = np.array([cal[(f, e)] for e in E])
        i = days[f].index(d)
        rows = F.loc[idx].sort_values("hour")
        cum = rows.p.expanding().mean().values
        for k, (ii, rr) in enumerate(rows.iterrows()):
            h = int(rr.hour)
            cq = twin_date(f, d, h) if h >= 5 else None
            if cq is None:
                cq = full_date(f, days[f][i - 1]) + 0.1
            m = (np.abs(calc - cq) <= 3) & (calc != cq)
            if not m.any():
                continue
            Em = [e for e, z in zip(E, m) if z]; Sh = SIG[h]
            RS = Sh.loc[[(f, e) for e in days[f] if (f, e) in ref]].astype(float)
            mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            q = (Sh.loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
            C = np.nan_to_num(((Sh.loc[[(f, e) for e in Em]].values.astype(float) - mu) / sd)[:, use])
            dist = np.sqrt(((C - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(calc[m] - cq)
            a1 = float(ec[(f, Em[int(np.argmin(dist))])]); pm = cum[k]
            if abs(a1 - pm) <= .30:
                out[ii] = rr.p + 0.5 * (a1 - pm)
    return out

