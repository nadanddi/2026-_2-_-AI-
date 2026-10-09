# -*- coding: utf-8 -*-
"""TD5: cheap no-refit checks of temperature candidates C1, C3, C6, C7 (집/클로드/문서/온도_현재해석_v1.md, 3절) on the
stored W40G-S out-of-fold predictions (TT1 checkpoints, REF members = TK2-identical).  (2026-10-10 집 클로드, user:
"기존 발견을 연결하고 해석을 갱신해서 온도 점수를 더 잘 맞춰보자", "도메인 지식 코멘트도 활용").  Diagnostic; pass
criteria fixed here before running (from the candidate list):
  C1  pass-2 days whose previous record is a different dong (diagnostic dong file st_dong_assign - NOT usable in a
      model, 6.397): corr( predicted 23 h of the most recent SAME-dong earlier record  -  predicted 0 h today ,
      today's 0-5 h mean residual ).  Go to a model test only if |r| >= .3.  (Causal dong accuracy is a separate gate.)
  C3  mean residual by diagnostic dong x day/night (9-16 h vs rest) x radiation tercile (out_rad), F47 and F13,
      day-block bootstrap 95% CI; 'signal' iff the dong difference is larger in day & high-radiation cells and its CI
      excludes 0.
  C6  partial Spearman of daytime (8-16 h) residual with proxy = (mean in_temp of the most recent same-dong earlier
      record) - (today's mean in_temp 0..h), controlling for today's in_temp; pass iff same sign in both farms and
      |rho| >= .20 (6b.39 criterion).
  C7  event-aligned residual: hours relative to act_shade or act_fog switching on (0 -> >0), -2..+4 h; 'signal' iff
      the post-event mean residual differs from pre-event by > .1 C in both farms.
Residual = W40G-S prediction - label (seed mean 7/101, PFN family A and B averaged), DIAG10 rows.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td5_temp_cheap_checks_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
import common
from scipy.stats import spearmanr
CK = os.path.join(env.LOCAL, "tt1_ckpt")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def load():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    G = G[G.validator == "DIAG10"].copy()
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    G["p"] = np.mean([.4 * G["base_REF_%d" % s] + (.2 + .4 * (1 - g)) * G.codex_REF + .4 * g * G["pfn_%s" % f]
                      for s in (7, 101) for f in ("A", "B")], axis=0)
    G["e"] = G.p - G.sub_temp
    tX, ty, sX = common.load_raw()
    X = tX[tX.farm.isin(["F13", "F47"])].set_index("row_id")
    for c in ("out_rad", "act_shade", "act_fog"):
        G[c] = X[c].reindex(G.row_id).values
    A = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).set_index(["farm", "day"]).dong
    G["dong"] = [A.get((f, d)) for f, d in zip(G.farm, G.day)]
    return G.sort_values(["farm", "day", "hour"]).reset_index(drop=True), A


def boot_ci(vals_by_day, n=5000, seed=0):
    v = np.asarray(vals_by_day); rng = np.random.default_rng(seed)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def main():
    G, A = load()
    D = G.groupby(["farm", "day"])
    p23 = G[G.hour == 23].set_index(["farm", "day"]).p; p0 = G[G.hour == 0].set_index(["farm", "day"]).p
    early = G[G.hour <= 5].groupby(["farm", "day"]).e.mean()
    tin = D.in_temp.mean()
    print("== C1 (pass-2, previous record = different dong)")
    xs, ys = [], []
    for (f, d) in early.index:
        if d < 179 or A.get((f, d - 1)) is None or A.get((f, d - 1)) == A.get((f, d)):
            continue
        prev = [k for k in range(d - 1, d - 8, -1) if A.get((f, k)) == A.get((f, d)) and (f, k) in p23.index]
        if not prev or (f, d) not in p0.index:
            continue
        xs.append(p23[(f, prev[0])] - p0[(f, d)]); ys.append(early[(f, d)])
    rr = np.corrcoef(xs, ys)[0, 1] if len(xs) > 3 else np.nan
    print("  n=%d  corr(same-dong prev 23h - today 0h, today 0-5h residual) = %+.2f -> %s" % (len(xs), rr, "GO" if abs(rr) >= .3 else "stop"))
    print("\n== C3 dong x day/night x radiation (mean residual, by day)")
    G["daypart"] = np.where((G.hour >= 9) & (G.hour <= 16), "day", "night")
    for f in ("F13", "F47"):
        x = G[G.farm == f]
        rad = x[x.daypart == "day"].groupby("day").out_rad.mean()
        q = rad.quantile([1 / 3, 2 / 3]).values
        tert = pd.cut(rad, [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
        x = x.assign(tert=x.day.map(tert))
        for part in ("day", "night"):
            for t in ("low", "high"):
                cell = x[(x.daypart == part) & (x.tert == t)]
                m = cell.groupby(["day", "dong"]).e.mean().unstack()
                if {"A", "B"} <= set(m.columns):
                    a, b = m.A.dropna(), m.B.dropna()
                    lo, hi = boot_ci(np.r_[a.values - a.mean() + (a.mean() - b.mean()), ]) if len(a) > 3 else (np.nan, np.nan)
                    print("  %s %-5s rad %-4s  A %+.3f (n%2d)  B %+.3f (n%2d)  A-B %+.3f" % (f, part, t, a.mean(), len(a), b.mean(), len(b), a.mean() - b.mean()))
    print("\n== C6 daytime residual vs same-dong water-temperature proxy (partial Spearman | today's in_temp)")
    res = []
    for f in ("F13", "F47"):
        rows = []
        for d in sorted(G[G.farm == f].day.unique()):
            prev = [k for k in range(d - 1, d - 8, -1) if A.get((f, k)) == A.get((f, d)) and (f, k) in tin.index]
            if not prev:
                continue
            dd = G[(G.farm == f) & (G.day == d) & (G.hour >= 8) & (G.hour <= 16)]
            rows.append((tin[(f, prev[0])] - dd.in_temp.mean(), dd.e.mean(), dd.in_temp.mean()))
        R = pd.DataFrame(rows, columns=["proxy", "res", "tin"]).dropna()
        rp = R.proxy - np.poly1d(np.polyfit(R.tin, R.proxy, 1))(R.tin); re_ = R.res - np.poly1d(np.polyfit(R.tin, R.res, 1))(R.tin)
        rho, p = spearmanr(rp, re_); res.append(rho)
        print("  %s n=%d rho %+.3f p %.3f" % (f, len(R), rho, p))
    print("  -> %s" % ("PASS" if np.sign(res[0]) == np.sign(res[1]) and min(abs(res[0]), abs(res[1])) >= .2 else "fail"))
    print("\n== C7 event-aligned residual (act switching 0 -> >0)")
    for ev in ("act_shade", "act_fog"):
        for f in ("F13", "F47"):
            x = G[G.farm == f].reset_index(drop=True)
            on = x.index[(x[ev] > 0) & (x[ev].shift(1).fillna(0) == 0) & (x.day == x.day.shift(1))]
            prof = {k: [] for k in range(-2, 5)}
            for i in on:
                for k in prof:
                    j = i + k
                    if 0 <= j < len(x) and x.day[j] == x.day[i]:
                        prof[k].append(x.e[j])
            mp = {k: np.mean(v) if v else np.nan for k, v in prof.items()}
            pre, post = np.nanmean([mp[-2], mp[-1]]), np.nanmean([mp[1], mp[2], mp[3]])
            print("  %-9s %s events %4d | %s | post-pre %+.3f" % (ev, f, len(on), " ".join("%+d:%+.2f" % (k, mp[k]) for k in prof), post - pre))


if __name__ == "__main__":
    main()
