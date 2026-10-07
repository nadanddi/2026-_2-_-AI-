# -*- coding: utf-8 -*-
"""RT1 (diagnostic, 2026-10-07 집 클로드; user: "use each model's strength only where it is strong — conditional").
Upper bound of conditional (regime-routed) combination of already-saved EC experts.  No new fitting.

Pools (rows aligned by validator / validation_fold / row_id; base columns must agree):
  S pool  seeds 7/101/2024, base = r3s (season R3).  Experts = base + every saved stage-3 corrector with
          per-seed columns: DB1 DC6 DC7 DI1 DP1 DP2 HM1 HM2 LG1 LR1 MW1 SB1 SE3 ST5 ST8 TF1 TS1 TS2 PAR1.
          Validators DIAG10 A B EXT10 EXT12 EL1.
  W pool  seeds 47/1414/6464, base = CUR = clip(.8 R3_DP1 + .2 PFN).  Experts = CUR, PFN share .4/.6/.8/1.0,
          R3 only, LH1 (LGB/MLP full features), no-DP1, PF1 B/C/D and PF2 E (0.6 R3 + 0.4 variant), ET / LGB / MLP
          alone.  Validators DIAG10 P2LOO EL1 (+ TM subset of DIAG10).
Cells:
  truth regime  = phase (day >= 179 -> pass 2) x high (label day mean >= 1)                 [oracle, not usable]
  input gate    = phase x (causal cumulative mean of the base prediction over hours 0..h >= .9)   [usable]
Strategies per seed and validator:
  S0  base
  S1  oracle-regime routing, CROSS-FIT: each fold uses, per truth cell, the expert with the lowest SSE on the
      other folds of the same validator
  S2  same, in-sample (optimistic)
  S3  input-gate routing, cross-fit (as S1 with gate cells)                      <- the usable version
  S4  per-day best expert (ceiling; in-sample by construction)
Reading rule (fixed before running; diagnostic only, NOT an adoption test):
  S1 gain < 3 % on DIAG10 and EL1 (seed mean)          -> regime routing has little to give; close the direction
  S1 >= 3 % and S3 captures < 1/3 of S1's gain        -> the gate signal is the bottleneck
  S3 captures >= 1/3, same sign for every seed on DIAG10 and EL1 -> worth a separately pre-registered adoption test
Run:  PYTHONPATH="" py -3.12 -u rt1_regime_routing_upper_bound_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd

L = env.LOCAL
KEY = ["validator", "validation_fold", "row_id"]
r = lambda e: float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


def s_pool():
    spec = {"DB1": "db", "DC6": "dc6", "DC7": "dc7", "DI1": "di", "DP1": "dp", "DP2": "dp2", "HM1": "hm", "HM2": "hm",
            "LG1": "lg", "LR1": "lr", "MW1": "mw", "SB1": "sb", "SE3": "se", "ST5": "st5", "ST8": "st", "TF1": "tf",
            "TS1": "ts", "TS2": "ts", "PAR1": "par"}
    seeds = (7, 101, 2024); D = None
    for name, p in spec.items():
        f = os.path.join(L, ("ec2_%s_all.csv" if name == "PAR1" else "ec3_%s_all.csv") % name)
        d = pd.read_csv(f)
        cols = {"%s_%d" % (p, s): "%s|%d" % (name, s) for s in seeds}
        if D is None:
            D = d[KEY + ["farm", "day", "hour", "sub_ec"] + ["r3s_%d" % s for s in seeds]].rename(
                columns={"r3s_%d" % s: "base|%d" % s for s in seeds})
        else:
            chk = D[KEY + ["base|%d" % s for s in seeds]].merge(d[KEY + ["r3s_%d" % s for s in seeds]], on=KEY)
            dif = max(float(np.abs(chk["base|%d" % s] - chk["r3s_%d" % s]).max()) for s in seeds)
            if dif > 1e-6:
                print("  skip %s: base differs (max %.3g)" % (name, dif)); continue
        D = D.merge(d[KEY + list(cols)].rename(columns=cols), on=KEY)
    names = ["base"] + [n for n in spec if "%s|7" % n in D.columns]
    return D, names, seeds


def w_pool():
    seeds = (47, 1414, 6464)
    P = pd.read_csv(os.path.join(L, "ec3_PF2_all.csv"))
    H = pd.read_csv(os.path.join(L, "ec3_LH1_all.csv"))
    hc = ["pfn"] + ["nodp_%d" % s for s in seeds] + ["%s_%d" % (m, s) for m in ("lgbF", "mlpF") for s in seeds]
    D = P.merge(H[KEY + hc], on=KEY)
    out = D[KEY + ["farm", "day", "hour", "sub_ec"]].copy()
    cl = lambda x: np.clip(x, D.lo, D.hi)
    for s in seeds:
        r3 = .6 * D["et_%d" % s] + .3 * D["lgb_%d" % s] + .1 * D["mlp_%d" % s]
        lh = .6 * D["et_%d" % s] + .3 * D["lgbF_%d" % s] + .1 * D["mlpF_%d" % s]
        out["base|%d" % s] = cl(.8 * r3 + .2 * D.pfn)
        for w in (4, 6, 8, 10):
            out["pfn%02d|%d" % (w, s)] = cl((1 - w / 10) * r3 + w / 10 * D.pfn)
        out["r3only|%d" % s] = cl(r3)
        out["LH1|%d" % s] = cl(.8 * lh + .2 * D.pfn)
        out["noDP1|%d" % s] = cl(.8 * D["nodp_%d" % s] + .2 * D.pfn)
        for t in "BCDE":
            out["PF_%s|%d" % (t, s)] = cl(.6 * r3 + .4 * D["pfn" + t])
        for m in ("et", "lgb", "mlp"):
            out["%s|%d" % (m, s)] = cl(D["%s_%d" % (m, s)])
    names = ["base", "pfn04", "pfn06", "pfn08", "pfn10", "r3only", "LH1", "noDP1", "PF_B", "PF_C", "PF_D", "PF_E",
             "et", "lgb", "mlp"]
    TM = pd.read_csv(os.path.join(L, "tm1_set_v1.csv")); tms = set(zip(TM.farm, TM.day))
    tm = out[(out.validator == "DIAG10") & np.array([(f, d) in tms for f, d in zip(out.farm, out.day)])].copy()
    tm["validator"] = "TM"
    return pd.concat([out, tm], ignore_index=True), names, seeds


def route(G, names, s, cell, crossfit):
    """Per fold, per cell, pick the expert with the lowest SSE (other folds if crossfit)."""
    E = np.column_stack([G["%s|%d" % (n, s)].to_numpy() for n in names])
    se = (E - G.sub_ec.to_numpy()[:, None]) ** 2
    T = pd.DataFrame(se, columns=names); T["fold"] = G.validation_fold.values; T["cell"] = cell
    tot = T.groupby(["fold", "cell"])[names].sum()
    pred = E[:, 0].copy(); picks = {}
    for f in np.unique(T.fold):
        for c in np.unique(cell):
            ref = tot.xs(c, level="cell")
            ref = ref.drop(index=f, errors="ignore") if crossfit else ref
            if not len(ref):
                continue
            k = int(np.argmin(ref.sum().values)); m = (T.fold.values == f) & (cell == c)
            pred[m] = E[m, k]; picks.setdefault(c, []).append(names[k])
    return pred, picks


def run(tag, D, names, seeds):
    print("\n" + "=" * 100 + "\n%s pool: %d experts %s\nrows %d" % (tag, len(names), names, len(D)))
    D = D.sort_values(KEY[:2] + ["farm", "day", "hour"]).reset_index(drop=True)
    day = D.groupby(["validator", "validation_fold", "farm", "day"])
    hi = (day.sub_ec.transform("mean") >= 1).to_numpy(); p2 = (D.day >= 179).to_numpy()
    truth = np.where(p2, 2, 0) + hi.astype(int)
    for v in sorted(D.validator.unique()):
        G = D[D.validator == v]; ix = G.index.values
        print("\n[%s] rows %d, days %d (pass-2 %d, high %d)" % (v, len(G), G.groupby(["farm", "day", "validation_fold"]).ngroups,
              G[p2[ix]].groupby(["farm", "day", "validation_fold"]).ngroups,
              G[hi[ix]].groupby(["farm", "day", "validation_fold"]).ngroups))
        res = {k: [] for k in ("S0", "S1", "S2", "S3", "S4")}; seg = {k: [] for k in res}
        for s in seeds:
            b = G["base|%d" % s].to_numpy()
            gk = G.groupby(["validation_fold", "farm", "day"]).cumcount()
            cm = pd.Series(b, index=G.index).groupby([G.validation_fold, G.farm, G.day]).cumsum() / (gk + 1)
            gate = np.where(p2[ix], 2, 0) + (cm.to_numpy() >= .9).astype(int)
            s1, pk1 = route(G, names, s, truth[ix], True)
            s2, _ = route(G, names, s, truth[ix], False)
            s3, pk3 = route(G, names, s, gate, True)
            E = np.column_stack([G["%s|%d" % (n, s)].to_numpy() for n in names]); y = G.sub_ec.to_numpy()
            dkey = G.validation_fold.astype(str) + G.farm + G.day.astype(str)
            dse = pd.DataFrame((E - y[:, None]) ** 2).groupby(dkey.values).transform("sum").to_numpy()
            s4 = E[np.arange(len(E)), dse.argmin(1)]
            for k, p in (("S0", b), ("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4)):
                res[k].append(r(p - y))
                seg[k].append([r((p - y)[m]) for m in (p2[ix] & ~hi[ix], p2[ix] & hi[ix], ~p2[ix] & ~hi[ix], ~p2[ix] & hi[ix])])
            if s == seeds[0]:
                cn = {0: "p1-normal", 1: "p1-high", 2: "p2-normal", 3: "p2-high"}
                print("   seed %d S1 picks (truth cell: most frequent): %s" % (s, {cn[c]: max(set(v_), key=v_.count) for c, v_ in pk1.items()}))
                print("   seed %d S3 picks (gate cell : most frequent): %s" % (s, {cn[c].replace("high", "gate>=.9").replace("normal", "gate<.9"): max(set(v_), key=v_.count) for c, v_ in pk3.items()}))
        b0 = np.mean(res["S0"])
        for k in ("S0", "S1", "S2", "S3", "S4"):
            sg = np.nanmean(np.array(seg[k], float), 0)
            print("   %s  RMSE %s  mean %.4f (%+.1f%%) | p2-normal %.4f p2-high %.4f p1-normal %.4f p1-high %.4f" % (
                k, " ".join("%.4f" % x for x in res[k]), np.mean(res[k]), 100 * (np.mean(res[k]) / b0 - 1), *sg))
        g1 = b0 - np.mean(res["S1"]); g3 = b0 - np.mean(res["S3"])
        print("   S3 captures %.0f%% of S1 gain; S3 better than S0 in %d/%d seeds" % (100 * g3 / g1 if g1 > 0 else float("nan"),
              sum(a < c for a, c in zip(res["S3"], res["S0"])), len(seeds)))


if __name__ == "__main__":
    run("S", *s_pool())
    run("W", *w_pool())
