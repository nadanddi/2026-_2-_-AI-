# -*- coding: utf-8 -*-
"""SF1: sealed-day forensics, high-EC (16) vs normal (37) sealed days.
Hypotheses, traces and pass rules are fixed in
EC_밀폐날_역추적_사전고정_2026-10-02.md (commit a7fac75) before this run.
2026-10-02 집 클로드.

Scope: DIAG10 OOF days of EC v2 (Codex phase 3, 360 days, final-lock 40 days
excluded).  Labels of locked days are never read (lock kept sealed).

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u sf1_sealed_forensics_v1.py
"""
import env  # noqa: F401
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans

ROOT = env.ROOT
LOCK = os.path.join(ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
OOF = os.path.join(ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1", "oof_predictions.csv")
ACT = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]


def split(df):
    p = df.row_id.str.split("_", expand=True)
    df["farm"], df["day"], df["hour"] = p[0], p[1].astype(int), p[2].astype(int)
    return df


def main():
    tX = split(pd.read_csv(os.path.join(env.DATA, "train_X.csv")))
    sX = split(pd.read_csv(os.path.join(env.DATA, "test_X.csv")))
    ty = split(pd.read_csv(os.path.join(env.DATA, "train_y.csv")))
    lock = {(s["farm"], int(s["day"])) for s in json.load(open(LOCK, encoding="utf-8"))["selected"]}
    tX["set"], sX["set"] = "train", "test"
    X = pd.concat([tX, sX], ignore_index=True)
    X = X[X.farm.isin(["F13", "F47"])]
    y = ty[ty.farm.isin(["F13", "F47"]) & ty.sub_ec.notna()]
    y = y[[(f, d) not in lock for f, d in zip(y.farm, y.day)]]
    X = X.merge(y[["row_id", "sub_ec", "sub_temp"]], on="row_id", how="left")

    k = ["farm", "day"]
    g = X.groupby(k)
    D = pd.DataFrame({
        "set": g.set.first(),
        "fan": g.act_circfan.mean(), "vent0": g.act_vent.apply(lambda s: (s == 0).mean()),
        "heat": g.act_heating.mean(), "co2sup": g.act_co2.mean(), "shade": g.act_shade.mean(),
        "thermal": g.act_thermal.mean(), "fog": g.act_fog.mean(),
        "tin": g.in_temp.mean(), "tin_min": g.in_temp.min(), "tin_max": g.in_temp.max(),
        "hum": g.in_hum.mean(), "co2": g.in_co2.mean(), "rad": g.out_rad.sum(), "tout": g.out_temp.mean(),
        "ec": g.sub_ec.mean(), "ec0": g.apply(lambda d: d.loc[d.hour == 0, "sub_ec"].mean(), include_groups=False),
        "ec23": g.apply(lambda d: d.loc[d.hour == 23, "sub_ec"].mean(), include_groups=False),
        "ecmin": g.sub_ec.min(), "ecmax": g.sub_ec.max(), "tsub": g.sub_temp.mean(),
    }).reset_index()
    D["sealed"] = (D.fan < 10) & (D.vent0 > 0.85)
    D["rise"] = D.ec23 - D.ec0
    D["ratio0"] = D.ec0 / D.ec
    lab = {(f, d): v for f, d, v in D[["farm", "day", "ec"]].itertuples(index=False) if not np.isnan(v)}
    for lag in (1, 2, 4):
        D["ec_m%d" % lag] = [lab.get((f, d - lag), np.nan) for f, d in zip(D.farm, D.day)]
    # 0-h actuator fingerprint clusters (inputs only, all days incl. test)
    h0 = X[X.hour == 0].set_index(k)[ACT].reindex(pd.MultiIndex.from_frame(D[k])).fillna(0)
    D["clu"] = KMeans(4, n_init=20, random_state=0).fit_predict((h0 - h0.mean()) / (h0.std() + 1e-9))

    o = pd.read_csv(OOF, encoding="utf-8-sig")
    o = o[o.validator == "DIAG10"]
    o["e"] = o.v2 - o.sub_ec
    pm = o.groupby(k).v2.mean().rename("pm")
    D = D.merge(pm.reset_index(), on=k, how="left")
    oofdays = set(map(tuple, o[k].drop_duplicates().values))
    D["oof"] = [(f, d) in oofdays for f, d in zip(D.farm, D.day)]
    S = D[D.oof & D.sealed].copy()
    S["high"] = S.ec >= 1.2
    H, N = S[S.high], S[~S.high]
    print("sealed OOF days %d: high %d, normal %d" % (len(S), len(H), len(N)))
    print(S.sort_values(["high", "farm", "day"])[["farm", "day", "high", "ec", "ec0", "ec23", "rise", "ec_m2", "ec_m1",
                                                    "pm", "tsub", "tin", "heat", "rad", "clu"]].round(3).to_string(index=False))

    def mw(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            return np.nan, len(a), len(b)
        return mannwhitneyu(a, b, alternative="two-sided").pvalue, len(a), len(b)

    print("\n== S1 accumulated state ==")
    n_r = int((H.ratio0 >= 0.85).sum())
    p, na, nb = mw(H.ec_m2, N.ec_m2)
    print("  high days with ec0/daymean >= 0.85: %d of %d (rule >= 12)" % (n_r, len(H)))
    print("  d-2 label EC: high median %.3f (n=%d) vs normal %.3f (n=%d), MW p %.4f (rule < 0.01, high > normal)"
          % (H.ec_m2.median(), na, N.ec_m2.median(), nb, p))
    s1 = n_r >= 12 and p < 0.01 and H.ec_m2.median() > N.ec_m2.median()
    p1, na1, nb1 = mw(H.ec_m1, N.ec_m1)
    print("  (descr) d-1 label EC: high %.3f (n=%d) vs normal %.3f (n=%d) p %.4f | d-4 high %.3f normal %.3f"
          % (H.ec_m1.median(), na1, N.ec_m1.median(), nb1, p1, H.ec_m4.median(), N.ec_m4.median()))
    print("  -> S1 %s" % ("PASS" if s1 else "FAIL"))

    print("\n== S2 within-day concentration ==")
    p, _, _ = mw(H.rise, N.rise)
    print("  rise (23h-0h): high median %+.3f vs normal %+.3f, MW p %.4f" % (H.rise.median(), N.rise.median(), p))
    s2 = (H.rise.median() >= 2 * max(N.rise.median(), 1e-9)) and p < 0.01 and H.rise.median() > 0
    for c in ("rad", "tin_max", "heat"):
        r, pp = spearmanr(S.rise, S[c])
        print("  spearman(rise, %-7s) %+.2f p %.3f" % (c, r, pp))
    print("  -> S2 %s" % ("PASS" if s2 else "FAIL"))

    print("\n== S3 step changes (descriptive) ==")
    S["jump2"] = S.ec - S.ec_m2
    print("  change vs d-2: high median %+.3f (n=%d), normal %+.3f (n=%d)"
          % (S.jump2[S.high].median(), S.jump2[S.high].notna().sum(), S.jump2[~S.high].median(), S.jump2[~S.high].notna().sum()))

    print("\n== S4 temperature ==")
    p, _, _ = mw(H.tsub, N.tsub)
    print("  sub_temp mean: high %.2f vs normal %.2f p %.3f" % (H.tsub.median(), N.tsub.median(), p))

    print("\n== S5 source / place ==")
    print("  high by cluster:", H.clu.value_counts().to_dict(), " normal by cluster:", N.clu.value_counts().to_dict())
    print("  high by farm x late:", H.groupby(["farm", H.day >= 179]).size().to_dict(),
          " normal:", N.groupby(["farm", N.day >= 179]).size().to_dict())
    top = max(H.clu.value_counts().max(), H.groupby(["farm", H.day >= 179]).size().max())
    print("  -> S5 %s (max concentration %d of %d, rule >= 12)" % ("PASS" if top >= 12 else "FAIL", top, len(H)))

    print("\n== other inputs (high vs normal medians, MW p) ==")
    for c in ("heat", "co2sup", "shade", "thermal", "fog", "tin", "tin_min", "tin_max", "hum", "co2", "rad", "tout", "fan",
              "vent0", "day"):
        p, _, _ = mw(H[c], N[c])
        print("  %-8s %8.2f %8.2f  p %.3f" % (c, H[c].median(), N[c].median(), p))

    print("\n== v2 residual by hour block on sealed days ==")
    oo = o.merge(S[k + ["high"]], on=k)
    oo["blk"] = oo.hour // 6
    print(oo.groupby(["high", "blk"]).e.mean().round(3).unstack().to_string())

    # test sealed days: where do they sit?
    T = D[(D.set == "test")]
    TS = T[T.sealed]
    print("\n== test inputs: sealed days %d of %d ==" % (len(TS), len(T)))
    feats = ["heat", "shade", "thermal", "fog", "tin", "tin_min", "tin_max", "hum", "co2", "rad", "tout", "co2sup"]
    mu, sd = S[feats].mean(), S[feats].std() + 1e-9
    ch = ((H[feats] - mu) / sd).mean()
    cn = ((N[feats] - mu) / sd).mean()
    z = (TS[feats] - mu) / sd
    dh = np.sqrt(((z - ch) ** 2).sum(1))
    dn = np.sqrt(((z - cn) ** 2).sum(1))
    print("  nearest centroid: high-like %d, normal-like %d" % ((dh < dn).sum(), (dh >= dn).sum()))
    TS = TS.assign(near=np.where(dh < dn, "high", "normal"))
    print("  test sealed days:", TS[["farm", "day", "near", "tin", "heat", "clu", "ec_m2"]].round(2).to_string(index=False))
    D.to_csv(os.path.join(env.LOCAL, "sf1_daytable.csv"), index=False)


if __name__ == "__main__":
    main()
