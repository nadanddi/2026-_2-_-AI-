# -*- coding: utf-8 -*-
"""agentY y1: systematic causal conditional-residual scan (temperature DIAG10 4 cells, EC DIAG10 3 seeds).
Rule fixed before running:
  lead iff (a) in each of 4 farm x pass cells mean residual same sign and |mean| > 2 record-clustered SE,
           (b) every prediction cell (temp 4 / EC 3 seeds) has same sign in all 4 farm x pass cells,
           (c) Stouffer z over 4 cells exceeds Bonferroni two-sided z for alpha .05 / n_bins scanned.
  Bins: deciles (or fixed bins for counters/states) on training rows; kept if >= 100 rows.
  Every conditioning variable uses only hours 0..h of the same record.
Gain: pred + (pass-1 pooled bin mean) on pass-2 rows; RMSE change per farm on all pass-2 rows.
"""
import sys, os, re
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__)); LOC = os.path.join(HERE, "local")
STATES = {"heat_on": lambda D: D.act_heating.values > 0, "thermal_on": lambda D: D.act_thermal.values > 50,
          "vent_on": lambda D: D.act_vent.values > 0, "shade_on": lambda D: D.act_shade.values > 50,
          "cold_air<10": lambda D: D.in_temp.values < 10, "fog_on": lambda D: D.act_fog.values > 0}
BLOCKS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]


def since_change(on):
    """hours since state last changed within the record (causal); 99 = no change yet today"""
    out = np.full(len(on), 99.0); last = None
    for i in range(1, len(on)):
        if on[i] != on[i - 1]:
            last = i
        if last is not None:
            out[i] = i - last
    return out


def feats(X):
    X = X.copy(); X["farm"] = X.row_id.str[:3]; X["day"] = X.row_id.str[4:7].astype(int); X["hour"] = X.row_id.str[8:10].astype(int)
    X = X.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
    g = X.groupby(["farm", "day"])
    for k in (1, 3, 6):
        X["d_in%d" % k] = X.in_temp - g.in_temp.shift(k)
    X["cumrad"] = g.out_rad.cumsum()
    X["in_min_sofar"] = g.in_temp.cummin()
    X["in_minus_out"] = X.in_temp - X.out_temp
    X["vent_mean_sofar"] = g.act_vent.cumsum() / (X.hour + 1)
    X["heat_mean_sofar"] = g.act_heating.cumsum() / (X.hour + 1)
    X["in_minus_in0"] = X.in_temp - g.in_temp.transform("first")
    for c, thr in (("act_heating", 0), ("act_thermal", 50), ("act_vent", 0), ("act_shade", 50)):
        s = np.zeros(len(X))
        for _, idx in g.groups.items():
            s[idx] = since_change(X.loc[idx, c].values > thr)
        X["since_%s" % c[4:]] = s
    return X


def add_pred_feats(D, p):
    D = D.copy(); D["pred"] = p
    g = D.groupby(["farm", "day"])
    D["pred_minus_p0"] = D.pred - g.pred.transform("first")
    D["air_minus_pred"] = D.in_temp - D.pred
    D["dpred1"] = D.pred - g.pred.shift(1)
    return D


CONT = ["in_temp", "out_temp", "in_hum", "in_co2", "out_rad", "out_hum", "out_wspd", "d_in1", "d_in3", "d_in6", "cumrad", "in_min_sofar",
        "in_minus_out", "vent_mean_sofar", "heat_mean_sofar", "in_minus_in0", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan",
        "pred", "pred_minus_p0", "air_minus_pred", "dpred1"]
FIXED = {"since_heating": [0, 1, 2, 4, 8, 99, 100], "since_thermal": [0, 1, 2, 4, 8, 99, 100], "since_vent": [0, 1, 2, 4, 8, 99, 100],
         "since_shade": [0, 1, 2, 4, 8, 99, 100], "act_fog": [0, 0.5, 200], "act_co2": [0, 0.5, 200]}


def binning(D):
    B = []
    for c in CONT:
        v = D[c].values.astype(float); ok = ~np.isnan(v)
        e = np.unique(np.nanquantile(v, np.linspace(0, 1, 11)))
        if len(e) < 3:
            continue
        b = np.digitize(v, e[1:-1])
        for i in range(len(e) - 1):
            B.append(("%s[%.4g,%.4g]" % (c, e[i], e[i + 1]), ok & (b == i), (c, e[i], e[i + 1], i == 0, i == len(e) - 2)))
    for c, edges in FIXED.items():
        v = D[c].values
        for lo, hi in zip(edges[:-1], edges[1:]):
            B.append(("%s[%g,%g)" % (c, lo, hi), (v >= lo) & (v < hi), (c, lo, hi, None, None)))
    for h in range(24):
        B.append(("hour=%d" % h, D.hour.values == h, ("hour", h, h, None, None)))
    for nm, f in STATES.items():
        m = f(D)
        for lo, hi in BLOCKS:
            hb = (D.hour.values >= lo) & (D.hour.values < hi)
            B.append(("h%d-%d&%s" % (lo, hi - 1, nm), hb & m, ("st", lo, hi, nm, False)))
            B.append(("h%d-%d&not_%s" % (lo, hi - 1, nm), hb & ~m, ("st", lo, hi, nm, True)))
    return [b for b in B if b[1].sum() >= 100]


def mask_from_spec(TX, spec):
    c, lo, hi, a, b = spec
    if c == "hour":
        return TX.hour.values == lo
    if c == "st":
        hb = (TX.hour.values >= lo) & (TX.hour.values < hi); m = STATES[a](TX)
        return hb & (~m if b else m)
    if c not in TX.columns:
        return None
    v = TX[c].values.astype(float)
    if a is None:
        return (v >= lo) & (v < hi)
    # decile bin: lower edge inclusive only for first bin (digitize semantics: e[i] <= v < e[i+1]); extend outer bins
    lo_ok = (v >= lo) if not a else np.ones(len(v), bool)
    hi_ok = (v < hi) if not b else np.ones(len(v), bool)
    return lo_ok & hi_ok & ~np.isnan(v)


def cell_stats(r, m, rec):
    rr = r[m]; gg = rec[m]; n = len(rr)
    if n < 20:
        return np.nan, np.nan, n
    mu = rr.mean(); s = pd.Series(rr - mu).groupby(gg).sum().values; G = len(s)
    if G < 3:
        return mu, np.nan, n
    return mu, np.sqrt((s ** 2).sum() * G / (G - 1)) / n, n


def scan(D, preds, ytrue, label, TX):
    p = np.mean(preds, axis=0)
    D = add_pred_feats(D, p)
    r = ytrue - p; rk = [ytrue - q for q in preds]
    rec = (D.farm + "_" + D.day.astype(str)).values
    B = binning(D); nb = len(B); zb = norm.ppf(1 - .025 / nb)
    p2 = D.day.values >= 179; fm = D.farm.values
    cellm = {"F13p1": (fm == "F13") & ~p2, "F47p1": (fm == "F47") & ~p2, "F13p2": (fm == "F13") & p2, "F47p2": (fm == "F47") & p2}
    rows = []
    for name, m, spec in B:
        st = {k: cell_stats(r, m & cm, rec) for k, cm in cellm.items()}
        mus = np.array([v[0] for v in st.values()]); ses = np.array([v[1] for v in st.values()]); ns = [v[2] for v in st.values()]
        z = mus / ses
        same = (not np.isnan(mus).any()) and np.all(np.sign(mus) == np.sign(mus[0]))
        all2 = bool(same and np.all(np.abs(z) > 2))
        zst = float(np.nansum(z) / 2) if same else 0.0
        pc_same = bool(same) and all(np.all(np.sign([q[m & cm].mean() for cm in cellm.values()]) == np.sign(mus[0])) for q in rk)
        rows.append(dict(bin=name, n=int(m.sum()), **{k + "_n": ns[i] for i, k in enumerate(cellm)}, **{k + "_mu": mus[i] for i, k in enumerate(cellm)},
                         **{k + "_z": z[i] for i, k in enumerate(cellm)}, stouffer=zst, same=bool(same), all2=all2, cells_same=pc_same,
                         bonf=abs(zst) > zb, lead=bool(all2 and pc_same and abs(zst) > zb)))
    T = pd.DataFrame(rows)
    print("\n==== %s: %d bins scanned, Bonferroni |Stouffer z| > %.2f (alpha .05/%d two-sided) ====" % (label, nb, zb, nb))
    print("bins same sign in 4 cells: %d | + all 4 |z|>2: %d | + all pred cells same sign: %d | + Bonferroni: %d" % (
        T.same.sum(), T.all2.sum(), (T.all2 & T.cells_same).sum(), T.lead.sum()))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    cols = ["bin", "n", "F13p1_mu", "F47p1_mu", "F13p2_mu", "F47p2_mu", "F13p1_z", "F47p1_z", "F13p2_z", "F47p2_z", "stouffer", "cells_same", "bonf", "lead"]
    show = T[T.all2].sort_values("stouffer", key=abs, ascending=False)
    print("-- bins with all 4 cells |z|>2 same sign --"); print(show[cols].round(3).to_string(index=False) if len(show) else "(none)")
    near = T[T.same & ~T.all2].assign(minz=T[["F13p1_z", "F47p1_z", "F13p2_z", "F47p2_z"]].abs().min(axis=1)).sort_values("minz", ascending=False).head(15)
    print("-- near misses (same sign 4 cells, sorted by weakest cell |z|) --"); print(near[cols + ["minz"]].round(3).to_string(index=False))
    # most striking pass-2-only bins (to show pass1/pass2 instability)
    T["p2z"] = (T.F13p2_z.fillna(0) + T.F47p2_z.fillna(0)) / np.sqrt(2)
    top2 = T.sort_values("p2z", key=abs, ascending=False).head(12)
    print("-- strongest pass-2 bins (both farms combined) whatever pass 1 says --"); print(top2[cols + ["p2z"]].round(3).to_string(index=False))
    sel = list(show.bin) + list(near.bin.head(5)) + list(top2.bin.head(5))
    out = []
    for name, m, spec in B:
        if name not in sel:
            continue
        for mode in ("pooled", "farm"):
            g = []
            adj = p.copy()
            for f in ("F13", "F47"):
                delta = r[m & ~p2].mean() if mode == "pooled" else r[m & ~p2 & (fm == f)].mean()
                sel_rows = m & p2 & (fm == f); adj[sel_rows] += delta
            for f in ("F13", "F47"):
                mm = p2 & (fm == f)
                g.append(100 * (np.sqrt(np.mean((ytrue - adj)[mm] ** 2)) / np.sqrt(np.mean((ytrue - p)[mm] ** 2)) - 1))
            tm = mask_from_spec(TX, spec)
            tn = (np.nan, np.nan) if tm is None else (int(tm.sum()), int((tm & (TX.in_temp.values < 8)).sum()))
            out.append((name, mode, r[m & ~p2].mean(), g[0], g[1], int((m & p2).sum()), tn[0], tn[1]))
    G = pd.DataFrame(out, columns=["bin", "delta_from", "delta_p1_pooled", "F13_p2_rmse%", "F47_p2_rmse%", "p2_rows_hit", "test_rows_hit", "test_cold<8_hit"])
    print("-- gain of adding pass-1 bin mean residual, judged on all pass-2 rows per farm --"); print(G.round(3).to_string(index=False))
    T.to_csv(os.path.join(LOC, "y1_scan_%s.csv" % label), index=False)
    return T, D, r


def worst_best(D, r, ytrue, p, label, k=20):
    p2 = D.day.values >= 179
    E = D[p2].assign(res=r[p2], y=ytrue[p2], p=p[p2]); E["rec"] = E.farm + "_" + E.day.astype(str).str.zfill(3)
    S = E.groupby("rec").agg(rmse=("res", lambda v: np.sqrt(np.mean(v ** 2))), bias=("res", "mean"), y_mean=("y", "mean"), p_mean=("p", "mean"),
                             y_rng=("y", lambda v: v.max() - v.min()), p_rng=("p", lambda v: v.max() - v.min()),
                             in_mean=("in_temp", "mean"), in_min=("in_temp", "min"), out_mean=("out_temp", "mean"), rad_sum=("out_rad", "sum"),
                             hum=("in_hum", "mean"), co2=("in_co2", "mean"), heat=("act_heating", "mean"), therm=("act_thermal", "mean"),
                             vent=("act_vent", "mean"), shade=("act_shade", "mean"), fan=("act_circfan", "mean"), fog=("act_fog", "mean"),
                             co2act=("act_co2", "mean"), in_na=("in_temp", lambda v: int(np.isnan(v).sum())))
    S["frac_bias"] = S.bias ** 2 / S.rmse ** 2
    S = S.sort_values("rmse", ascending=False); W = S.head(k); Bt = S.tail(k)
    pd.set_option("display.width", 300)
    print("\n==== %s: 20 worst pass-2 records (of %d) ====" % (label, len(S))); print(W.round(2).to_string())
    print("---- 20 best ----"); print(Bt.round(2).to_string())
    print("---- mean worst vs best vs all ----"); print(pd.DataFrame({"worst": W.mean(), "best": Bt.mean(), "all": S.mean()}).round(3).to_string())
    print("share of pass-2 SSE in worst 20: %.1f%%; worst 5: %.1f%%" % (100 * (W.rmse ** 2).sum() / (S.rmse ** 2).sum(), 100 * (W.rmse.head(5) ** 2).sum() / (S.rmse ** 2).sum()))
    print("---- hourly side-by-side, worst 20 (h0..h23): res=y-p, y, p, air temp, heating/thermal/vent ----")
    for rc in W.index:
        e = E[E.rec == rc].sort_values("hour")
        print("%-8s res %s" % (rc, " ".join("%+5.2f" % v for v in e.res)))
        print("%-8s y   %s" % ("", " ".join("%5.2f" % v for v in e.y)))
        print("%-8s p   %s" % ("", " ".join("%5.2f" % v for v in e.p)))
        print("%-8s air %s" % ("", " ".join("%5.1f" % v for v in e.in_temp)))
        print("%-8s act %s" % ("", " ".join("%d/%d/%d" % (a, b, c) for a, b, c in zip(e.act_heating.fillna(-1), e.act_thermal.fillna(-1), e.act_vent.fillna(-1)))))
    print("---- hourly residual, best 20 ----")
    for rc in Bt.index:
        e = E[E.rec == rc].sort_values("hour"); print("%-8s res %s" % (rc, " ".join("%+5.2f" % v for v in e.res)))
    # residual-shape summary: mean residual by hour block, worst vs best
    E["blk"] = E.hour // 4
    hb = E.groupby(["rec", "blk"]).res.mean().unstack()
    print("---- mean residual by 4h block: worst20 / best20 / all ----")
    print(pd.DataFrame({"worst": hb.loc[W.index].mean(), "worst_abs": hb.loc[W.index].abs().mean(), "best": hb.loc[Bt.index].mean(), "all": hb.mean()}).round(3).to_string())
    S.to_csv(os.path.join(LOC, "y1_records_%s.csv" % label))
    return S


def main():
    X0 = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X0 = X0[X0.row_id.str[:3].isin(["F13", "F47"])]
    X = feats(X0); TX = feats(pd.read_csv(os.path.join(env.DATA, "test_X.csv")))
    M = pd.read_csv(os.path.join(LOC, "temp_DIAG10.csv"))
    D = M[["row_id", "sub_temp", "p0", "p1", "p2", "p3"]].merge(X, on="row_id", how="left")
    preds = [D["p%d" % i].values for i in range(4)]; y = D.sub_temp.values
    T, Dt, r = scan(D, preds, y, "TEMP_DIAG10", TX)
    worst_best(Dt, r, y, np.mean(preds, axis=0), "TEMP_DIAG10")
    E = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv")
    E = E[(E.tag == "CUR") & (E.validator == "DIAG10")]
    W = E.pivot_table(index="row_id", columns="seed", values="sg"); yy = E.groupby("row_id").sub_ec.first()
    De = pd.DataFrame({"row_id": W.index, "sub_ec": yy.loc[W.index].values})
    De = pd.concat([De, pd.DataFrame(W.values, columns=["s%d" % c for c in W.columns])], axis=1).merge(X, on="row_id", how="left")
    pe = [De["s%d" % c].values for c in W.columns]
    Te, De2, re_ = scan(De, pe, De.sub_ec.values, "EC_DIAG10", TX)
    worst_best(De2, re_, De.sub_ec.values, np.mean(pe, axis=0), "EC_DIAG10")


if __name__ == "__main__":
    main()
