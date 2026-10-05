# -*- coding: utf-8 -*-
"""RC0 (diagnostic, fixed before running; 2026-10-05 집 클로드; GPU for speed only).
TabPFN context selection by REFERENCE (allowed: storing / referencing training data, 6.309).
Current v2 TabPFN: context = 2000 random training rows (4 seeds averaged).  RC0 asks whether a
SEASON-LOCAL context helps pass-2 days: context = all 24 hours of the K = 80 training days
nearest to the query day in the training reference calendar (SG2 calendar, training records
only); the query day's date is known at hour 0 (previous record's date + 0.1, SG2 rule), so
the context is the same for every hour of the day and uses no later input.
Never tried before (catalog: random / weighted / sealed-only contexts; sealed-only hurt 6.57).
Arms (same FULL features with DC4 season as v2, n_estimators 4, float32):
  RND  2000 random rows of the fold's training set, random_state 1..4, averaged;
  RET  80 nearest days (1920 rows), random_state 1..4 (same context), averaged.
Blend as submitted: 0.8 * R3S (stored OOF, shrunk) + 0.2 * shrink(PFN).
Sets: pass-2 days of DIAG10 folds and EL1 folds (R3S from ec2_DC5_oof / ec2_EL1_oof seed mean).
Clue (fixed): RET blend beats RND blend on pass-2 rows of BOTH sets, normal days (label day
mean < 1) not worse by > 1 % in either, and DIAG10 farm x 5-day cluster bootstrap
P(worse) < .0125."""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd, torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(HERE, "ec3_SG2_reference_knn_level_v1.py"))
sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
dc4, p3, core = sg2.dc4, sg2.p3, sg2.core
CK = os.path.join(env.LOCAL, "rc0_ckpt"); K = 80


def pfn(Xc, yc, Xq, seed):
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                   ignore_pretraining_limits=True, inference_precision=torch.float32)
    m.fit(Xc, yc)
    return np.asarray(m.predict(Xq), float)


def shrink(p, fr):
    d = fr[["farm", "day", "hour"]].copy(); d["p"] = p
    avg = d.groupby(["farm", "day"]).p.transform(lambda z: z.expanding().mean())
    return (.5 * d.p + .5 * avg).values


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    R, WV, hrs, SIG = sg2.prepare_structure()
    roles = R.set_index(["farm", "day"]).role
    alld = {f: sorted(R[R.farm == f].day) for f in ("F13", "F47")}
    ec = lab.groupby(["farm", "day"]).sub_ec.mean(); labset = set(ec.index)
    folds = [x for x in fds if x[0] == "DIAG10"]
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    for name, i, vd in folds:
        if not any(d >= 179 for f, d in vd):
            continue
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd and d >= 179 for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index).sort_values(["farm", "day", "hour"])
        ref = {(f, int(d)) for f, d in labset if (f, int(d)) not in vd}
        cal = sg2.ref_calendar(R, WV, ref); cache = {}

        def full_date(f, d):
            if (f, d) in cal: return cal[(f, d)]
            if (f, d) in cache: return cache[(f, d)]
            E = [e for e in alld[f] if (f, e) in ref]
            dist = np.sqrt(np.nanmean((WV.loc[[(f, e) for e in E]].values - WV.loc[(f, d)].values) ** 2, axis=1)); ok = dist <= .05
            if ok.any(): t = float(np.mean([cal[(f, e)] for e, o in zip(E, ok) if o]))
            else:
                j = alld[f].index(d); t = (full_date(f, alld[f][j - 1]) + (0.0 if roles.get((f, d)) == "second" else 0.1)) if j else 0.0
            cache[(f, d)] = t; return t
        Xtr, ytr = tr[FS].to_numpy(np.float32), tr.sub_ec.to_numpy(float)
        Xva = va[FS].to_numpy(np.float32)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy(); frame["validator"], frame["validation_fold"] = name, i
        rnd = np.mean([pfn(Xtr[np.random.default_rng(s).choice(len(tr), 2000, replace=False)],
                           ytr[np.random.default_rng(s).choice(len(tr), 2000, replace=False)], Xva, s) for s in (1, 2, 3, 4)], axis=0)
        frame["rnd"] = rnd
        ret = np.full(len(va), np.nan)
        tdc = tdays.copy(); tdc["cal"] = [cal.get((f, int(d)), np.nan) for f, d in zip(tdc.farm, tdc.day)]
        for (f, d), idx in va.groupby(["farm", "day"]).groups.items():
            j = alld[f].index(d); cq = full_date(f, alld[f][j - 1]) + 0.1
            C = tdc[(tdc.farm == f) & tdc.cal.notna()].copy(); C["dist"] = (C.cal - cq).abs() + 1e-3 * (C.day - d).abs()
            keep = set(zip(C.nsmallest(K, "dist").farm, C.nsmallest(K, "dist").day))
            cm = np.array([(a, int(b)) in keep for a, b in zip(tr.farm, tr.day)])
            pos = [va.index.get_loc(x) for x in idx]
            ret[pos] = np.mean([pfn(Xtr[cm], ytr[cm], Xva[pos], s) for s in (1, 2, 3, 4)], axis=0)
        frame["ret"] = ret
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    D5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); D5 = D5[D5.validator == "DIAG10"]
    E1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv"))
    r3 = pd.concat([D5.assign(validator="DIAG10")[["row_id", "validator", "r3s_7", "r3s_101", "r3s_2024"]],
                    E1.assign(validator="EL1")[["row_id", "validator", "r3s_7", "r3s_101", "r3s_2024"]]])
    r3["r3s"] = r3[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
    O = O.merge(r3[["row_id", "validator", "r3s"]], on=["row_id", "validator"], how="left")
    assert O.r3s.notna().all()
    O = O.sort_values(["validator", "farm", "day", "hour"]).reset_index(drop=True)
    O.to_csv(os.path.join(env.LOCAL, "rc0_all_v1.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    clue = True
    for v in ("DIAG10", "EL1"):
        G = O[O.validator == v].copy(); G["dm"] = G.groupby(["farm", "day"]).sub_ec.transform("mean")
        for c in ("rnd", "ret"):
            G["b_" + c] = .8 * G.r3s + .2 * shrink(G[c].values, G)
        print("\n%s pass-2 (%d days)" % (v, G.groupby(["farm", "day"]).ngroups))
        for nm, m in (("all", G.dm.notna()), ("normal", G.dm < 1), ("high", G.dm >= 1)):
            g = G[m]
            print("  %-6s PFN alone RND %.4f RET %.4f | blend R3S %.4f RND %.4f RET %.4f (%+.1f%% vs RND)" % (
                nm, r(g.rnd - g.sub_ec), r(g.ret - g.sub_ec), r(g.r3s - g.sub_ec), r(g.b_rnd - g.sub_ec), r(g.b_ret - g.sub_ec),
                100 * (r(g.b_ret - g.sub_ec) / r(g.b_rnd - g.sub_ec) - 1)))
        clue &= r(G.b_ret - G.sub_ec) < r(G.b_rnd - G.sub_ec)
        n = G.dm < 1; clue &= r((G.b_ret - G.sub_ec)[n]) <= 1.01 * r((G.b_rnd - G.sub_ec)[n])
        if v == "DIAG10":
            G["cl"] = G.farm + "_" + (G.day // 5).astype(str)
            dd = ((G.b_ret - G.sub_ec) ** 2 - (G.b_rnd - G.sub_ec) ** 2).groupby(G.cl).agg(["sum", "count"])
            sm, nn = dd["sum"].values, dd["count"].values
            idx = np.random.default_rng(20261005).integers(0, len(sm), (20000, len(sm)))
            p = float(((sm[idx].sum(1) / nn[idx].sum(1)) >= 0).mean()); print("  DIAG10 P(worse) %.4f" % p); clue &= p < .0125
    print("\nRC0 clue:", clue)


if __name__ == "__main__":
    main()
