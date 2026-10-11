# -*- coding: utf-8 -*-
"""MX1: per-record input META features (precision, NaN, uniques, runs, steps, actuator grids, roughness, parity)
(a) same-house adjacency consistency (truth chain b<-a1 vs same-farm controls with day within +-4 of a1)
(b) association with day-mean OOF residual (temp rt, EC re / log re), pass1 (day<179) and pass2;
    ExtraTrees grouped by DIAG10 fold, R2 vs 30 label-permutation null.
Read-only on data; test_X only for distribution glance (no fitting)."""
import os, sys, pickle
R0 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R0)
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
OUT = os.path.join(R0, "local", "agentX_meta")
COLS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
INS = ["in_temp", "in_hum", "in_co2"]; ACTS = COLS[7:]


def longest_run(v):
    v = v[~np.isnan(v)]
    if len(v) < 2: return np.nan
    best = cur = 1
    for i in range(1, len(v)):
        cur = cur + 1 if v[i] == v[i - 1] else 1; best = max(best, cur)
    return best


def feats(X):
    X = X.copy(); X["farm"] = X.row_id.str[:3]; X["day"] = X.row_id.str[4:7].astype(int); X["hour"] = X.row_id.str[8:10].astype(int)
    rows = []
    for (f, d), g in X.groupby(["farm", "day"]):
        g = g.sort_values("hour"); r = {"farm": f, "day": d, "nh": len(g)}
        for c in COLS:
            v = g[c].values.astype(float); vn = v[~np.isnan(v)]
            r[c + "_nan"] = np.isnan(v).sum()
            if len(vn) == 0: continue
            frac10 = np.round(vn * 10) % 10; r[c + "_dec1"] = np.mean(np.abs(vn - np.round(vn)) > 1e-9)
            r[c + "_nuniq"] = len(np.unique(vn)); r[c + "_run"] = longest_run(v)
            dv = np.abs(np.diff(vn)); dz = dv[dv > 1e-9]
            r[c + "_stepmin"] = dz.min() if len(dz) else 0; r[c + "_zdiff"] = np.mean(dv < 1e-9) if len(dv) else np.nan
            if c in INS:
                r[c + "_even"] = np.mean((np.round(vn * (10 if c == "in_temp" else 1)) % 2) == 0)
                if len(dv) > 3:
                    dd = np.diff(vn); r[c + "_ac1"] = np.corrcoef(dd[:-1], dd[1:])[0, 1] if dd.std() > 0 else np.nan
                    r[c + "_d2"] = np.mean(np.abs(np.diff(vn, 2)))
            if c in ACTS:
                mid = vn[(vn > 0) & (vn < 100)]
                r[c + "_mid"] = len(mid); r[c + "_grid60"] = np.mean(np.abs(mid * 60 / 100 - np.round(mid * 60 / 100)) < 1e-6) if len(mid) else np.nan
                r[c + "_int"] = np.mean(np.abs(mid - np.round(mid)) < 1e-9) if len(mid) else np.nan
                r[c + "_z"] = np.mean(vn == 0); r[c + "_full"] = np.mean(vn == 100)
        r["in_co2_min"] = np.nanmin(g.in_co2.values) if g.in_co2.notna().any() else np.nan
        r["in_hum_max"] = np.nanmax(g.in_hum.values) if g.in_hum.notna().any() else np.nan
        r["in_hum_sat"] = np.mean(g.in_hum.values >= 99) if g.in_hum.notna().any() else np.nan
        nanh = g.hour.values[g[INS].isna().any(axis=1).values]; r["nan_first_h"] = nanh.min() if len(nanh) else -1
        r["nan_in_rows"] = len(nanh)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])]
    Tx = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); Tx = Tx[Tx.row_id.str[:3].isin(["F13", "F47"])]
    F = feats(X); FT = feats(Tx); F.to_pickle(os.path.join(OUT, "meta_train.pkl")); FT.to_pickle(os.path.join(OUT, "meta_test.pkl"))
    fc = [c for c in F.columns if c not in ("farm", "day", "nh")]
    fc = [c for c in fc if F[c].nunique(dropna=True) > 1]
    print("records train %d test %d, varying meta features %d" % (len(F), len(FT), len(fc)))
    # --- train vs test distribution glance
    print("\n== train vs test (mean) for features with |std diff|>0.5 ==")
    for c in fc:
        if c in FT:
            a, b = F[c].mean(), FT[c].mean(); s = F[c].std()
            if s > 0 and abs(a - b) / s > .5: print("  %-22s train %.3f test %.3f" % (c, a, b))
    # --- (a) chain consistency
    R = pickle.load(open(os.path.join(R0, "local", "linker_v3", "base.pkl"), "rb")); rec = R["recs"].reset_index(drop=True)
    T = pd.read_pickle(os.path.join(R0, "local", "linker_v3", "truth.pkl"))
    key = {i: (rec.farm[i], int(rec.day[i])) for i in range(len(rec))}
    FI = F.set_index(["farm", "day"])
    res = {c: [] for c in fc}; npair = 0
    for _, t in T.iterrows():
        b, a = key[int(t.b)], key[int(t.a1)]
        if b not in FI.index or a not in FI.index: continue
        ctr = [k for k in FI.index if k[0] == a[0] and abs(k[1] - a[1]) <= 4 and k != a and k != b]
        if not ctr: continue
        npair += 1; fb = FI.loc[b]; fa = FI.loc[a]; fcx = FI.loc[ctr]
        for c in fc:
            if pd.isna(fb[c]) or pd.isna(fa[c]): continue
            dt = abs(fb[c] - fa[c]); dc = (fcx[c] - fb[c]).abs().dropna().values
            if len(dc): res[c].append(np.mean(dc > dt) + .5 * np.mean(dc == dt))
    print("\n== (a) same-house adjacency AUC (truth a1 vs same-farm +-4d controls), %d pairs ==" % npair)
    A = sorted([(np.mean(v), len(v), c) for c, v in res.items() if len(v) > 50], reverse=True)
    for auc, n, c in A[:15]: print("  %-22s AUC %.3f (n %d)" % (c, auc, n))
    print("  ... lowest:", ", ".join("%s %.3f" % (c, a) for a, n, c in A[-3:]))
    pd.DataFrame(A, columns=["auc", "n", "feat"]).to_csv(os.path.join(OUT, "mx1_chain_auc.csv"), index=False)
    # --- (b) residual association
    H = pd.read_pickle(os.path.join(OUT, "resid.pkl"))
    D = H.groupby(["farm", "day"]).agg(rt=("rt", "mean"), re=("re", "mean"), rle=("rle", "mean"), fold=("fold", "first"), fold_e=("fold_e", "first")).reset_index()
    Z = D.merge(F, on=["farm", "day"], how="inner")
    rng = np.random.default_rng(7)
    for tgt, fold in (("rt", "fold"), ("re", "fold_e"), ("rle", "fold_e")):
        for pname, msk in (("pass1", Z.day < 179), ("pass2", Z.day >= 179), ("all", Z.day >= 0)):
            S = Z[msk & Z[tgt].notna()]
            cs = []
            for c in fc:
                v = S[c]; ok = v.notna()
                if v[ok].nunique() < 3: continue
                rho, p = spearmanr(v[ok], S[tgt][ok]); cs.append((p, rho, c, ok.sum()))
            cs.sort(); m = len(cs)
            print("\n== (b) %s %s: n %d records; top Spearman (Bonferroni m=%d) ==" % (tgt, pname, len(S), m))
            for p, rho, c, n in cs[:5]: print("  %-22s rho %+.3f p %.2g  bonf %.2g" % (c, rho, p, min(1, p * m)))
            Xm = S[fc].fillna(-9).values; y = S[tgt].values; g = S[fold].values

            def cvr2(yy):
                pr = np.zeros_like(yy)
                for k in np.unique(g):
                    tr = g != k; et = ExtraTreesRegressor(200, min_samples_leaf=5, max_features=.3, random_state=0, n_jobs=-1).fit(Xm[tr], yy[tr]); pr[~tr] = et.predict(Xm[~tr])
                return 1 - np.sum((yy - pr) ** 2) / np.sum((yy - yy.mean()) ** 2)
            r2 = cvr2(y); null = [cvr2(rng.permutation(y)) for _ in range(30 if pname != "all" else 0)]
            print("  ExtraTrees fold-CV R2 %.3f%s" % (r2, ("  null mean %.3f max %.3f p %.3f" % (np.mean(null), np.max(null), (1 + np.sum(np.array(null) >= r2)) / 31)) if null else ""))
    Z.to_pickle(os.path.join(OUT, "mx1_Z.pkl"))


if __name__ == "__main__":
    main()
