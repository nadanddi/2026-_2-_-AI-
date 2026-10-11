# -*- coding: utf-8 -*-
"""MX3: drill the MX1 top hits (act_shade_stepmin = shade 'schedule mode', in_hum_run, in_temp_even, in_hum_nuniq):
partial Spearman with day-mean OOF residual after controlling 14 day-mean input summaries (+day),
per farm x pass sign check, causal 0..h version vs rest-of-day residual (h>h0), test_X prevalence."""
import os, sys
R0 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R0)
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
OUT = os.path.join(R0, "local", "agentX_meta")
CTL = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]


def load(fn):
    X = pd.read_csv(os.path.join(env.DATA, fn)); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"] = X.row_id.str[:3]; X["day"] = X.row_id.str[4:7].astype(int); X["hour"] = X.row_id.str[8:10].astype(int)
    return X.sort_values(["farm", "day", "hour"])


def shade_mode(v):
    """1 if shade follows a 0/100 schedule: all values 0 or 100 except <=2 transition hours"""
    v = v[~np.isnan(v)]
    return float(np.sum((v > 0) & (v < 100)) <= 2) if len(v) else np.nan


def stepmin(v):
    v = v[~np.isnan(v)]; d = np.abs(np.diff(v)); d = d[d > 0]
    return d.min() if len(d) else 0.


def runlen(v):
    best = cur = 1
    for i in range(1, len(v)):
        cur = cur + 1 if v[i] == v[i - 1] and not np.isnan(v[i]) else 1; best = max(best, cur)
    return best


def partial(x, y, C):
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(C).any(1)); x, y, C = rankdata(x[ok]), rankdata(y[ok]), C[ok]
    C1 = np.c_[np.ones(len(C)), (C - C.mean(0)) / (C.std(0) + 1e-9)]
    rx = x - C1 @ np.linalg.lstsq(C1, x, rcond=None)[0]; ry = y - C1 @ np.linalg.lstsq(C1, y, rcond=None)[0]
    return spearmanr(rx, ry)


def main():
    X = load("train_X.csv"); Tx = load("test_X.csv"); H = pd.read_pickle(os.path.join(OUT, "resid.pkl"))
    def dayfeat(X, hmax=23):
        rows = []
        for (f, d), g in X[X.hour <= hmax].groupby(["farm", "day"]):
            r = {"farm": f, "day": d}
            r.update({c: g[c].mean() for c in CTL}); r["vent0"] = np.mean(g.act_vent == 0)
            sv = g.act_shade.values.astype(float); r["shade_mode"] = shade_mode(sv); r["shade_stepmin"] = stepmin(sv)
            r["hum_run"] = runlen(g.in_hum.values.astype(float)); r["hum_nuniq"] = g.in_hum.nunique()
            t = g.in_temp.dropna().values; r["temp_even"] = np.mean(np.round(t * 10) % 2 == 0) if len(t) else np.nan
            rows.append(r)
        return pd.DataFrame(rows)
    F = dayfeat(X); FT = dayfeat(Tx)
    D = H.groupby(["farm", "day"]).agg(rt=("rt", "mean"), re=("re", "mean"), rle=("rle", "mean"), ec=("sub_ec", "mean")).reset_index()
    Z = D.merge(F, on=["farm", "day"])
    C = Z[CTL + ["vent0", "day"]].values.astype(float)
    print("shade schedule-mode share: train %.3f (F13 %.3f F47 %.3f), test %.3f (F13 %.3f F47 %.3f)" % (
        F.shade_mode.mean(), F[F.farm == "F13"].shade_mode.mean(), F[F.farm == "F47"].shade_mode.mean(),
        FT.shade_mode.mean(), FT[FT.farm == "F13"].shade_mode.mean(), FT[FT.farm == "F47"].shade_mode.mean()))
    print("train schedule-mode by pass: p1 %.3f p2 %.3f; day range of schedule-mode train days: %s" % (
        F[F.day < 179].shade_mode.mean(), F[F.day >= 179].shade_mode.mean(), sorted(F[F.shade_mode == 1].day.tolist())[:5] + ["..."] + sorted(F[F.shade_mode == 1].day.tolist())[-5:]))
    for x in ("shade_mode", "shade_stepmin", "hum_run", "hum_nuniq", "temp_even"):
        for tg in ("rt", "re", "rle"):
            line = []
            for nm, m in (("all", Z.day >= 0), ("p1", Z.day < 179), ("p2", Z.day >= 179)):
                for fm in ("F13", "F47"):
                    mm = (m & (Z.farm == fm)).values
                    if mm.sum() < 15: continue
                    r, p = spearmanr(Z[x][mm], Z[tg][mm], nan_policy="omit"); line.append("%s/%s %+.2f" % (nm, fm, r))
            rp, pp = partial(Z[x].values.astype(float), Z[tg].values.astype(float), C)
            r0, p0 = spearmanr(Z[x], Z[tg], nan_policy="omit")
            print("%-14s %-3s raw %+.3f (p %.2g) partial|15 controls %+.3f (p %.2g) | %s" % (x, tg, r0, p0, rp, pp, "  ".join(line)))
    # shade mode: mean residual by mode x pass, plus high-EC share
    print("\nEC residual / level by shade schedule-mode:")
    print(Z.groupby([Z.day >= 179, "shade_mode"]).agg(n=("re", "size"), re=("re", "mean"), rle=("rle", "mean"), ec=("ec", "mean"), hiEC=("ec", lambda v: np.mean(v >= 1)), rt=("rt", "mean"), out_temp=("out_temp", "mean"), vent0=("vent0", "mean")).round(3))
    # within closed / high-EC: does schedule mode still matter?
    for nm, m in (("vent0>=.8", Z.vent0 >= .8), ("vent0<.8", Z.vent0 < .8)):
        S = Z[m]; print("  %s: n %d, re by mode %s" % (nm, len(S), S.groupby("shade_mode").re.agg(["count", "mean"]).round(3).to_dict()))
    # causal: shade mode known by hour h0 (0..h0) vs residual mean of hours > h0
    print("\ncausal check: mode from hours 0..h0, residual of hours h0+1..23")
    for h0 in (9, 12, 16):
        Fh = dayfeat(X, h0)[["farm", "day", "shade_mode", "shade_stepmin"]]
        Rh = H[H.hour > h0].groupby(["farm", "day"]).agg(re=("re", "mean"), rt=("rt", "mean")).reset_index().merge(Fh, on=["farm", "day"]).merge(Z[["farm", "day"] + CTL + ["vent0"]], on=["farm", "day"])
        Ch = Rh[CTL + ["vent0", "day"]].values.astype(float)
        for tg in ("re", "rt"):
            a = spearmanr(Rh.shade_stepmin, Rh[tg], nan_policy="omit"); b = partial(Rh.shade_stepmin.values.astype(float), Rh[tg].values.astype(float), Ch)
            print("  h0 %2d %s: stepmin raw %+.3f (p %.2g) partial %+.3f (p %.2g); mode share %.2f" % (h0, tg, a[0], a[1], b[0], b[1], Rh.shade_mode.mean()))
    Z.to_pickle(os.path.join(OUT, "mx3_Z.pkl"))


if __name__ == "__main__":
    main()
