# -*- coding: utf-8 -*-
"""submission_18 EC post-processing (C18, 2026-10-11 집 클로드).  Input: the stored submission_14 EC members
(output_14/prediction_details.npz from model.py, unchanged 9th-round pipeline) + public data.  Steps:
  1. B4  : 0.6 R3 + 0.4 TabPFN (instead of 0.8 / 0.2), same shrink + clip as model.py.
  2. HG3 : SG2 + strict high-EC correction (functions copied verbatim from the research file ec3_HG3_strict_guard_gate_v1.py;
           with all 400 labelled days as references it reproduces the submitted SG2 exactly - checked below).
  3. W2  : pure pass-2 window rule 'same ID record number - 2 = same house, previous date' (catalog 6.549 / 6.550).
Causality: every evaluation row uses hours 0..h of its own record, earlier records of the same farm and training labels.
st_dong_assign_v1.csv = research input-only record role table (first / second / single, A / B) used by SG2's calendar.
Run:  python c18post.py   (writes output_18/ec_c18.csv)
"""
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
W = ["out_temp", "out_hum", "out_rad", "out_wspd"]


def prepare_structure():
    R = pd.read_csv(os.path.join(HERE, "st_dong_assign_v1.csv")).sort_values(["farm", "day"]).reset_index(drop=True)
    R["date"] = R.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
    X = pd.concat([pd.read_csv(os.path.join(DATA, f)) for f in ("train_X.csv", "test_X.csv")])
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
    DV = ["in_hum", "act_shade", "act_thermal", "act_fog", "in_temp", "in_co2"]
    XX = X.sort_values(["farm", "day", "hour"]).copy(); gg = XX.groupby(["farm", "day"])
    for v in DV:
        XX["cm_" + v] = gg[v].transform(lambda z: z.expanding().mean())
    DM = XX.set_index(["farm", "day", "hour"])[["cm_" + v for v in DV]]
    return R, WV, hrs, SIG, DM


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


SGN = np.array([1, 1, 1, 1, -1, -1], float)


def s_low_table(DM, ref):
    """dict (farm, h) -> (mu, sd) from reference records' 0..h means."""
    T = {}
    for f in ("F13", "F47"):
        for h in range(24):
            M = DM.xs(h, level="hour").loc[f]
            M = M[[(f, d) in ref for d in M.index]]
            T[(f, h)] = (M.mean().values, M.std().replace(0, 1).values)
    return T


def correction(frame, pcol, vd, lock, ec, R, WV, hrs, SIG, ref, cal, DM=None, ST=None):
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
                hg[j] = rr[pcol] + 0.5 * ((a1 + a2) / 2 - pm)
    return out, hg



def shrink(p, F):
    d = F[["farm", "day", "hour"]].reset_index(drop=True).copy(); d["p"] = p; d = d.sort_values(["farm", "day", "hour"])
    avg = d.groupby(["farm", "day"]).p.transform(lambda s: s.expanding().mean()); o = np.empty(len(p)); o[d.index.values] = (.5 * d.p + .5 * avg).values
    return o


def w2(F, pcol, WV, hrs, ec23, labset, exist):
    out = F[pcol].values.copy(); fired = []

    def twin(f, a, b, h):
        cols = np.asarray(hrs <= h)
        return np.sqrt(np.nanmean((WV.loc[(f, a)].values[cols] - WV.loc[(f, b)].values[cols]) ** 2)) <= .05

    def has_p1_twin(f, d, h):
        P1 = [e for e in WV.loc[f].index if e < 179]
        cols = np.asarray(hrs <= h)
        A = WV.loc[[(f, e) for e in P1]].values[:, cols]; b = WV.loc[(f, d)].values[cols]
        return bool((np.sqrt(np.nanmean((A - b) ** 2, axis=1)) <= .05).any())

    for (f, d), idx in F.groupby(["farm", "day"]).groups.items():
        if d < 179 or (f, d - 2) not in labset or has_p1_twin(f, d - 2, 23):
            continue
        if (f, d - 1) in exist:
            ok_c = twin(f, d - 2, d - 1, 23)
        else:
            ok_c = not ((f, d - 3) in exist and twin(f, d - 2, d - 3, 23))
        if not ok_c:
            continue
        rows = F.loc[idx].sort_values("hour"); cum = rows[pcol].expanding().mean().values; a = ec23[(f, d - 2)]
        for k, (ii, rr) in enumerate(rows.iterrows()):
            if has_p1_twin(f, d, int(rr.hour)):
                continue
            j = F.index.get_loc(ii); out[j] = rr[pcol] + (a - cum[k]); fired.append((f, d, int(rr.hour)))
    return out, fired



def main():
    out_dir = os.path.join(HERE, "output_18"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ec_c18.csv"); assert not os.path.exists(out), "refusing to overwrite " + out
    z = np.load(os.path.join(HERE, "output_14", "prediction_details.npz")); rid = z["row_id"]
    s2 = np.load(os.path.join(HERE, "output_14", "sg2_details.npz")); assert (s2["row_id"] == rid).all()
    F = pd.DataFrame({"row_id": rid, "farm": [r[:3] for r in rid], "day": [int(r[4:7]) for r in rid], "hour": [int(r[8:10]) for r in rid]})
    ty = pd.read_csv(os.path.join(DATA, "train_y.csv")); ty = ty[ty.row_id.str[:3].isin(["F13", "F47"])].copy()
    ty["farm"], ty["day"], ty["hour"] = ty.row_id.str[:3], ty.row_id.str[4:7].astype(int), ty.row_id.str[8:10].astype(int)
    ec = ty.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index); ec23 = ty[ty.hour == 23].set_index(["farm", "day"]).sub_ec
    lo, hi = float(ty.sub_ec.min()), float(ty.sub_ec.max())
    r3, pfn = z["raw_r3_by_seed"].mean(0), z["raw_pfn_by_context"].mean(0)
    F["p14pre"] = np.clip(shrink(.8 * r3 + .2 * pfn, F), lo, hi)
    rep13 = bool(np.abs(F.p14pre.values - s2["pred_submission13_config"]).max() < 1e-12)
    F["pb4"] = np.clip(shrink(.6 * r3 + .4 * pfn, F), lo, hi)
    Rr, WV, hrs, SIG, DM = prepare_structure()
    vd = set(zip(F.farm, F.day)); cal = ref_calendar(Rr, WV, labset); ST = s_low_table(DM, labset)
    sg14, _ = correction(F, "p14pre", vd, set(), ec, Rr, WV, hrs, SIG, labset, cal, DM, ST)
    rep14 = bool(np.abs(np.clip(sg14, lo, hi) - s2["pred_final"]).max() < 1e-12)   # True on the original public data
    _, hgb = correction(F, "pb4", vd, set(), ec, Rr, WV, hrs, SIG, labset, cal, DM, ST)
    F["b4hg3"] = np.clip(hgb, lo, hi)
    ids = pd.concat([pd.read_csv(os.path.join(DATA, f), usecols=["row_id"]) for f in ("train_X.csv", "test_X.csv")]).row_id
    ids = ids[ids.str[:3].isin(["F13", "F47"])]
    exist = {(a, int(b)) for a, b in zip(ids.str[:3], ids.str[4:7])}
    w, fired = w2(F, "b4hg3", WV, hrs, ec23, labset, exist)
    F["sub_ec"] = np.clip(w, lo, hi)
    F[["row_id", "sub_ec"]].to_csv(out, index=False, float_format="%.6f")
    d = F.sub_ec - s2["pred_final"]
    print("reproduces submission_14 pre-SG2 %s, SG2 %s (both must be True on the original data); W2 fired on" % (rep13, rep14), sorted({(f, d_) for f, d_, h in fired}))
    print("wrote %s | change vs submission_14 EC: RMS %.4f mean %+.4f min %+.3f max %+.3f" % (out, np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max()))


if __name__ == "__main__":
    main()
