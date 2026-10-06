# -*- coding: utf-8 -*-
"""EC stage-3 PF2: TabPFN with reference-anchor features PLUS 'difference to the anchor' features (user choice
"1": let the model learn how to adjust the anchor's EC by today's condition difference).  Fixed before running;
2026-10-07 집 클로드.  GPU for speed only.
Variant E = PF1 variant D features (FULL + season + 7 AF0 anchor features) + 7 difference features:
  for the best anchor (same SG2-style hour-causal search as AF0; TRAINING rows exclude their own +-3 record days),
  dX = mean of X over hours 0..h of the query record - mean of X over hours 0..h of the anchor record, for
  X = in_temp, in_hum, in_co2, sealed (act_vent == 0), co2 dosing (act_co2 > 0), heating (> 0), thermal curtain (> 0).
Same TabPFN settings and contexts as PF1 (2000 rows, contexts 5..8, n_estimators 4); A and D re-used from PF1
checkpoints (identical folds / seeds), E computed here.
Blend for judging: clip(shrink(0.6 R3_DP1 + 0.4 PFN)), R3 seeds 47 / 1414 / 6464.
Decision (fixed, single candidate): E replaces A iff every seed improves over A on TM, P2LOO and EL1 AND TM seed-mean
farm x 5-day block bootstrap P(worse) < .025.  E vs D reported (does the difference information add to D).
Run (after PF1; checkpointed):  PYTHONPATH="" py -3.12 -u ec3_PF2_anchor_difference_features_v1.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("pf1", os.path.join(HERE, "ec3_PF1_tabpfn_upgrade_v1.py"))
pf1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf1)
af0, sg2, dc4, p3, core, dp1 = pf1.af0, pf1.sg2, pf1.dc4, pf1.p3, pf1.core, pf1.dp1
SEEDS = pf1.SEEDS; CK1 = pf1.CK; CK = os.path.join(env.LOCAL, "pf2_ckpt")
DV = ["in_temp", "in_hum", "in_co2", "seal", "co2on", "heaton", "thermon"]
DIFF = ["dd_" + v for v in DV]


def cum_table():
    X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    X = X.sort_values(["farm", "day", "hour"])
    X["seal"] = (X.act_vent == 0).astype(float).where(X.act_vent.notna())
    X["co2on"] = (X.act_co2 > 0).astype(float).where(X.act_co2.notna())
    X["heaton"] = (X.act_heating > 0).astype(float).where(X.act_heating.notna())
    X["thermon"] = (X.act_thermal > 0).astype(float).where(X.act_thermal.notna())
    g = X.groupby(["farm", "day"])
    for v in DV:
        X["c_" + v] = g[v].transform(lambda s: s.expanding().mean())
    return X.set_index(["farm", "day", "hour"])[["c_" + v for v in DV]]


def best_anchor_days(queries, train_q, R, WV, hrs, SIG, ec, ref, cal, lock):
    """Same search as af0.anchor_features; returns farm, day, hour, anchor_day of the best candidate."""
    roles = R.set_index(["farm", "day"]).role
    alld = {f: sorted(R[R.farm == f].day) for f in ("F13", "F47")}
    pre, zs, out, cache = {}, {}, [], {}
    for f in ("F13", "F47"):
        E = [e for e in alld[f] if (f, e) in ref]
        pre[f] = dict(E=np.array(E), W=WV.loc[[(f, e) for e in E]].values, cal=np.array([cal[(f, e)] for e in E]),
                      lab=np.array([(f, e) in ec.index and (f, e) not in lock for e in E]))
    for h in range(24):
        S = SIG[h].astype(float)
        for f in ("F13", "F47"):
            RS = S.loc[[(f, e) for e in pre[f]["E"]]]; mu, sd = RS.mean().values, RS.std().replace(0, np.nan).values
            zs[(f, h)] = (mu, sd, (RS.values - mu) / sd)

    def full_date(f, d):
        if (f, d) in cal: return cal[(f, d)]
        if (f, d) in cache: return cache[(f, d)]
        P = pre[f]; dist = np.sqrt(np.nanmean((P["W"] - WV.loc[(f, d)].values) ** 2, axis=1)); ok = dist <= .05
        if ok.any(): t = float(P["cal"][ok].mean())
        else:
            i = alld[f].index(d); t = (full_date(f, alld[f][i - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if i else 0.0
        cache[(f, d)] = t; return t
    for f, d in queries:
        P = pre[f]; keep = np.ones(len(P["E"]), bool)
        if (f, d) in train_q:
            keep &= np.abs(P["E"] - d) > af0.EXCL
        i = alld[f].index(d); b = WV.loc[(f, d)].values
        for h in range(24):
            cq = None
            if h >= 5:
                cm = np.asarray(hrs <= h); dist = np.sqrt(np.nanmean((P["W"][:, cm] - b[cm]) ** 2, axis=1)); ok = (dist <= .05) & keep
                if ok.any(): cq = float(P["cal"][ok].mean())
            if cq is None: cq = (full_date(f, alld[f][i - 1]) + 0.1) if i else 0.0
            m = keep & P["lab"] & (np.abs(P["cal"] - cq) <= 3) & (P["cal"] != cq)
            a = np.nan
            if m.any():
                mu, sd, Zr = zs[(f, h)]
                q = (SIG[h].loc[(f, d)].values.astype(float) - mu) / sd; use = ~np.isnan(q)
                sc = np.sqrt(((np.nan_to_num(Zr[m][:, use]) - q[use]) ** 2).mean(axis=1)) + .15 * np.abs(P["cal"][m] - cq)
                a = float(P["E"][m][np.argmin(sc)])
            out.append((f, d, h, a))
    return pd.DataFrame(out, columns=["farm", "day", "hour", "anchor_day"])


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    F38 = [c for c in core.FULL if c != "day"] + ["season"]
    R, WV, hrs, SIG = sg2.prepare_structure(); CT = cum_table()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    folds = [x for x in fds if x[0] == "DIAG10"]
    for f in ("F13", "F47"):
        for d in sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique()):
            folds.append(("P2LOO", len(folds), {(f, int(d))}))
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    for name, i, vd in folds:
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
        ref = {(f, int(d)) for f, d in labset if (f, int(d)) not in vd}; cal = sg2.ref_calendar(R, WV, ref)
        tq = [(f, int(d)) for f, d in tdays.itertuples(index=False)]; vq_ = [(f, int(d)) for f, d in vdays[["farm", "day"]].itertuples(index=False)]
        A = af0.anchor_features(tq + vq_, set(tq), R, WV, hrs, SIG, ec, ref, cal, lockd)
        B = best_anchor_days(tq + vq_, set(tq), R, WV, hrs, SIG, ec, ref, cal, lockd)
        qv = CT.reindex(list(zip(B.farm, B.day, B.hour))).values
        av = CT.reindex(list(zip(B.farm, B.anchor_day.fillna(-1).astype(int), B.hour))).values
        for j, v in enumerate(DV):
            B["dd_" + v] = qv[:, j] - av[:, j]
        A = A.merge(B.drop(columns="anchor_day"), on=["farm", "day", "hour"])
        tr = tr.merge(A, on=["farm", "day", "hour"], how="left").set_index(tr.index)
        va = va.merge(A, on=["farm", "day", "hour"], how="left").set_index(va.index)
        frame = pd.read_csv(os.path.join(CK1, "%s_%d.csv" % (name, i)))
        assert (frame.row_id.values == va.row_id.values).all()
        cols = F38 + af0.AF + DIFF; Xtr, Xva = tr[cols].to_numpy(np.float32), va[cols].to_numpy(np.float32); y = tr.sub_ec.to_numpy(float)
        ps = []
        for c in (5, 6, 7, 8):
            ix = np.random.default_rng(c).choice(len(tr), size=min(2000, len(tr)), replace=False)
            ps.append(pf1.pfn(Xtr[ix], y[ix], Xva, c))
        frame["pfnE"] = core.shrink(np.mean(ps, axis=0), va)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O.to_csv(os.path.join(env.LOCAL, "ec3_PF2_all.csv"), index=False)
    TM = pd.read_csv(os.path.join(env.LOCAL, "tm1_set_v1.csv")); TMs = set(zip(TM.farm, TM.day))
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))

    def bl(G, tag, s):
        return np.clip(.6 * (.6 * G["et_%d" % s] + .3 * G["lgb_%d" % s] + .1 * G["mlp_%d" % s]) + .4 * G["pfn" + tag], G.lo, G.hi)
    sets = {"TM": O[(O.validator == "DIAG10") & np.array([(f, d) in TMs for f, d in zip(O.farm, O.day)])],
            "P2LOO": O[(O.validator == "P2LOO") & (O.day >= 179)], "EL1": O[(O.validator == "EL1") & (O.day >= 179)]}
    print("\nTabPFN alone: " + " | ".join("%s " % nm + " ".join("%s %.4f" % (t, r(G["pfn" + t] - G.sub_ec)) for t in "ADE") for nm, G in sets.items()))
    for base in ("A", "D"):
        ok = True; print("\nE vs %s (0.6 R3 + 0.4 PFN)" % base)
        for nm, G in sets.items():
            cells = []
            for s in SEEDS:
                a, b = r(bl(G, base, s) - G.sub_ec), r(bl(G, "E", s) - G.sub_ec); ok &= b < a
                cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
            G2 = G.assign(dm=G.groupby(["farm", "day"]).sub_ec.transform("mean")); nn = (G2.dm < 1).values
            bm = np.mean([bl(G2, base, s) for s in SEEDS], axis=0); cm = np.mean([bl(G2, "E", s) for s in SEEDS], axis=0)
            print("  %-6s %s | normal %.4f->%.4f high %.4f->%.4f" % (nm, "  ".join(cells), r((bm - G2.sub_ec)[nn]), r((cm - G2.sub_ec)[nn]),
                  r((bm - G2.sub_ec)[~nn]), r((cm - G2.sub_ec)[~nn])))
        T = sets["TM"].copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
        bm = np.mean([bl(T, base, s) for s in SEEDS], axis=0); cm = np.mean([bl(T, "E", s) for s in SEEDS], axis=0)
        dd = pd.Series((cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2, index=T.index).groupby(T.cl).agg(["sum", "count"])
        sm, n = dd["sum"].values, dd["count"].values
        idx = np.random.default_rng(20261007).integers(0, len(sm), (20000, len(sm)))
        p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
        print("  TM P(worse) %.4f" % p)
        if base == "A":
            print("PF2 decision:", "ADOPT E" if ok and p < .025 else "KEEP A", "(all better %s, P %.4f)" % (ok, p))


if __name__ == "__main__":
    main()
