# -*- coding: utf-8 -*-
"""LF1: interpolation / fill traces in the LABELS (train_y), F13 and F47.  Exploratory forensics (2026-10-09 집 클로드,
user: "정답 보간 흔적 더 찾아봐").  Reference for 'untouched sensor series' = test_X inputs (the brief states no
artificial restoration/alteration was applied to test inputs); train_X inputs (stated to contain restored values)
shown for contrast.
Traces looked for, per series (farm, contiguous hours):
  T1 single-hour mean fill: v[t] == (v[t-1] + v[t+1]) / 2 at the series resolution, v[t-1] != v[t+1]
  T2 multi-hour linear fill: runs of >= 3 consecutive equal NON-ZERO first differences (2nd difference == 0)
  T3 flat fill: runs of >= 3 identical consecutive values (first difference == 0)
  T4 joint missing hour: T1 holds for sub_temp AND sub_ec in the same row (vs the product of the marginal rates)
  T5 whole-day copies: a day's 24 labels identical to another day's, or identical up to a constant offset
  T6 resolution drops: rows whose label has fewer decimals than the series' usual resolution
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u lf1_label_interp_forensics_v1.py
"""
import env  # noqa: F401
import common
import numpy as np, pandas as pd

FARMS = ("F13", "F47")


def ndec(v):
    s = ("%.6f" % v).rstrip("0").split(".")
    return len(s[1]) if len(s) > 1 else 0


def split(df):
    df = df.copy()
    df[["farm", "day", "hour"]] = df.row_id.str.split("_", expand=True)
    df.day = df.day.astype(int); df.hour = df.hour.astype(int)
    df["t"] = df.day * 24 + df.hour
    return df[df.farm.isin(FARMS)].sort_values(["farm", "t"]).reset_index(drop=True)


def traces(df, col, res):
    """rates of T1/T2/T3 on contiguous hourly series of one column; returns dict and per-row T1 flag."""
    v = df[col].to_numpy(float); t = df.t.to_numpy(); f = df.farm.to_numpy()
    cont_prev = np.r_[False, (t[1:] - t[:-1] == 1) & (f[1:] == f[:-1])]
    cont_next = np.r_[cont_prev[1:], False]
    ok = cont_prev & cont_next & ~np.isnan(v) & ~np.isnan(np.r_[np.nan, v[:-1]]) & ~np.isnan(np.r_[v[1:], np.nan])
    p, n = np.r_[np.nan, v[:-1]], np.r_[v[1:], np.nan]
    t1 = ok & (np.abs(v - (p + n) / 2) < res / 2 + 1e-12) & (np.abs(p - n) > res / 2)
    d = np.r_[np.nan, np.diff(v)]; d[~cont_prev] = np.nan
    eqd = np.r_[False, np.abs(d[1:] - d[:-1]) < res / 2 + 1e-12] & ~np.isnan(d) & ~np.isnan(np.r_[np.nan, d[:-1]])
    nz = np.abs(d) > res / 2
    # run lengths of consecutive equal non-zero diffs (>= 2 equal diffs = >= 3 points on a line beyond start)
    def runs(mask):
        out, cur = [], 0
        for m in mask:
            if m: cur += 1
            elif cur: out.append(cur); cur = 0
        if cur: out.append(cur)
        return np.array(out)
    lin = runs(eqd & nz)
    flat = runs((np.abs(d) <= res / 2) & ~np.isnan(d))
    n_ok = int(ok.sum())
    return dict(rows=int((~np.isnan(v)).sum()), T1=t1[ok].mean() if n_ok else np.nan,
                T2_runs_ge2=int((lin >= 2).sum()), T2_rows=int(lin[lin >= 2].sum()) if len(lin) else 0,
                T3_runs_ge2=int((flat >= 2).sum()), T3_rows=int(flat[flat >= 2].sum()) if len(flat) else 0), t1, ok


def main():
    tX, ty, sX = common.load_raw()
    Y, XT, XS = split(ty), split(tX), split(sX)
    print("== T1-T3 per series (resolution = most common decimals)")
    rows = []
    for name, df, cols in (("train_y", Y, ["sub_temp", "sub_ec"]), ("test_X (untouched)", XS, ["in_temp", "in_hum", "in_co2"]),
                           ("train_X", XT, ["in_temp", "in_hum", "in_co2"])):
        for c in cols:
            for f in FARMS:
                g = df[df.farm == f]
                dec = g[c].dropna().map(ndec).mode().iloc[0]; res = 10.0 ** -dec
                r, _, _ = traces(g, c, res)
                n = r["rows"]
                rows.append((name, c, f, dec, n, 100 * r["T1"], r["T2_runs_ge2"], 1000 * r["T2_rows"] / n, r["T3_runs_ge2"], 1000 * r["T3_rows"] / n))
    print("%-20s %-8s %-4s dec %6s  T1%%   T2runs T2/1000  T3runs T3/1000" % ("set", "col", "farm", "rows"))
    for x in rows:
        print("%-20s %-8s %-4s %3d %6d  %5.1f  %6d  %6.1f   %6d  %6.1f" % x)

    print("\n== T4 joint single-hour fill (sub_temp and sub_ec both = neighbour mean)")
    for f in FARMS:
        g = Y[Y.farm == f]
        _, a, oka = traces(g, "sub_temp", .01); _, b, okb = traces(g, "sub_ec", .001)
        ok = oka & okb
        pa, pb, pab = a[ok].mean(), b[ok].mean(), (a & b)[ok].mean()
        print("%s rows %d  P(temp)=%.3f P(ec)=%.3f  joint %.4f vs independent %.4f  ratio %.2f  (n joint %d)"
              % (f, ok.sum(), pa, pb, pab, pa * pb, pab / (pa * pb) if pa * pb else np.nan, int((a & b)[ok].sum())))

    print("\n== T5 whole-day copies in labels")
    for c in ("sub_temp", "sub_ec"):
        P = Y.pivot_table(index=["farm", "day"], columns="hour", values=c).dropna()
        M = P.to_numpy(); k = P.index.tolist()
        exact, offset = [], []
        for i in range(len(M)):
            dd = M[i + 1:] - M[i]
            same = np.where(np.all(np.abs(dd) < 1e-9, axis=1))[0]
            exact += [(k[i], k[i + 1 + j]) for j in same]
            const = np.where((np.ptp(dd, axis=1) < 1e-9) & (np.abs(dd[:, 0]) > 1e-9))[0]
            offset += [(k[i], k[i + 1 + j], round(float(dd[j, 0]), 4)) for j in const]
        print("%s: days %d  exact copies %d %s  constant-offset copies %d %s" % (c, len(M), len(exact), exact[:5], len(offset), offset[:5]))

    print("\n== T6 resolution drops in labels (fewer decimals than usual)")
    hi = Y.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1.2
    for c, usual in (("sub_temp", 2), ("sub_ec", 3)):
        dd = Y[c].map(lambda v: ndec(v) if pd.notna(v) else -1)
        low = (dd >= 0) & (dd < usual)
        # chance of fewer decimals when the true value ends in 0: about 10% for one dropped digit
        print("%s: rows with < %d decimals %.1f%% (a trailing 0 by chance ~10%%); high-EC days %.1f%%, other days %.1f%%"
              % (c, usual, 100 * low[dd >= 0].mean(), 100 * low[(dd >= 0) & hi].mean(), 100 * low[(dd >= 0) & ~hi].mean()))
        lowdays = Y[low].groupby(["farm", "day"]).size()
        print("   days with >= 6 such rows: %d of %d; top %s" % (int((lowdays >= 6).sum()), Y.groupby(["farm", "day"]).ngroups,
                                                          lowdays.sort_values(ascending=False).head(5).to_dict()))


if __name__ == "__main__":
    main()
