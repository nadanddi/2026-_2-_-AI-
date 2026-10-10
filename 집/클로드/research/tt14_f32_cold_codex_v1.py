# -*- coding: utf-8 -*-
"""TT14: F32 re-evaluation under the COLD scope rule (2026-10-10 집 클로드, user: "진행해봐").
Background: 6.32 (re-run today, logs/test_f32_codex_rerun_20261010.log reproduces it exactly): adding F32 to CODEX training
gave EXT10 -1.2..-1.7% (CI excludes 0) but DIAG10 +0.55..+1.13% -> rejected under the old 'both validators' rule; 6.36
summarised it backwards.  That run used FULL-world features and the old 0.8 BASE + 0.2 CODEX blend.  Fixed before running.
Candidate CF(w): CODEX refit on (F13/F47 training rows of the fold, MASK world, season index as in TK2) + ALL F32 rows with
  row weight w, an is_f32 indicator added to physics and residual features; F32 columns absent at F32 (out_hum,
  act_shade, act_thermal, act_heating, act_fog) are NaN; F32 season = NaN (different site, no season map).
  Applied on the COLD side only:  CODEX_used = g * CODEX_REF + (1 - g) * CODEX_F32,  g = clip((in_temp - 8)/2, 0, 1);
  BASE, TabPFN, blend formula unchanged (rows with in_temp >= 10 are untouched).
  w in {0.1, 0.3}  (0.3 was the better value in 6.32 -> post-hoc risk noted), k = 2, alpha .0125.
RULE (user temperature scope rule, scope COLD): EXT10 primary - every BASE seed {7,101} x PFN family {A,B} better and farm x
  5-day block bootstrap P(worse) < .0125;  DIAG10 / EXT12 / EL1 fail iff seed-mean worse and share(better) < .0125.
  Scoring excludes the 30 strict CO2-rough days (protocol 10-10).  CODEX deterministic (seed 726).
Sanity: CODEX without F32 refit with the same code on the first DIAG10 fold must equal the stored codex_REF (< 1e-9).
Reported: rows g < 1, in_temp < 6, pass-2.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt14_f32_cold_codex_v1.py
"""
import os, sys, pathlib
import env  # noqa: F401
import numpy as np, pandas as pd
sys.argv = ["x"]
sys.path.insert(0, os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "temp_tk_season_20261003_v1"))
import world as W  # noqa: E402
import common  # noqa: E402
import resid_reset_features as R  # noqa: E402  (same module world.py uses)
CK1 = os.path.join(env.LOCAL, "tt1_ckpt")
CK = os.path.join(env.LOCAL, "tt14_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt14_season"
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
VALS = ("DIAG10", "EXT10", "EXT12", "EL1"); ALPHA = .0125; WS = (0.1, 0.3)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def f32_frame(FC, PC):
    """Same computation as resid_reset_features.build_features (which filters to F13/F47), applied to F32."""
    tX, ty, sX = common.load_raw()
    a = R._read(tX[tX.farm == "F32"])
    a = a.sort_values(["farm", "t"]).reset_index(drop=True)
    z = {c: a[c] for c in R.METADATA}
    z.update(farm_id=(a.farm == "F47").astype(float), sin=np.sin(a.hour * np.pi / 12), cos=np.cos(a.hour * np.pi / 12),
             second=(a.day >= 179).astype(float))
    for c in R.USABLE:
        v = a[c] if c in a else pd.Series(np.nan, index=a.index); g = v.groupby([a.farm, a.day])
        z[c] = v
        z[c + "_h0"] = v.where(a.hour == 0).groupby([a.farm, a.day]).ffill()
        z[c + "_mean"] = g.transform(lambda x: x.expanding().mean())
        z[c + "_std"] = g.transform(lambda x: x.expanding().std())
        z[c + "_diff"] = g.diff()
        if c in R.FILTERED:
            for h in [1, 3, 8]:
                z[c + "_reset" + str(h)] = g.transform(lambda x, h=h: x.ewm(halflife=h).mean())
    z["delta"] = a.in_temp - a.out_temp
    F = pd.DataFrame(z).merge(ty[["row_id", "sub_temp"]], on="row_id", how="inner")
    F["season"] = np.nan
    for c in set(FC + PC):
        if c not in F.columns:
            F[c] = np.nan
    F["is_f32"] = 1.0
    return F.reset_index(drop=True)


def fit_predict(tr, va, w, FC, PC):
    TM = W.TM; sf, sp = TM.FEATURE_COLUMNS, TM.PHYSICS_COLUMNS
    TM.FEATURE_COLUMNS, TM.PHYSICS_COLUMNS = FC, PC
    try:
        return TM.codex_fit_predict(tr.reset_index(drop=True), va.reset_index(drop=True), w, 726)
    finally:
        TM.FEATURE_COLUMNS, TM.PHYSICS_COLUMNS = sf, sp


def run():
    os.makedirs(CK, exist_ok=True); os.makedirs(W.OUT, exist_ok=True)
    lab, pfn, ct, phc, wb, wp, wv, sets, z = W.worlds()
    wp = np.asarray(wp, float)
    FC = [c if c != "day" else "season" for c in W.FEATURE_COLUMNS]
    PC = list(W.TM.PHYSICS_COLUMNS)
    F32 = f32_frame(FC, PC)
    print("F32 rows %d, non-null share of CODEX features %.2f" % (len(F32), F32[FC].notna().mean().mean()), flush=True)
    ref_all = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    checked = False
    for name, folds in sets:
        for k, fd in enumerate(folds):
            path = os.path.join(CK, "%s_%d.csv" % (name, k))
            if os.path.exists(path):
                continue
            tm, vm = common.split_mask(lab, fd)
            if not vm.sum():
                continue
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt14_" + name, k)
            tr = tr.reset_index(drop=True).assign(is_f32=0.0); va = va.reset_index(drop=True).assign(is_f32=0.0)
            out = pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values})
            if not checked:
                c0 = fit_predict(tr, va, wp[tm], FC, PC)
                R0 = ref_all[ref_all.validator == name].set_index("row_id").reindex(out.row_id).codex_REF.values
                d = np.abs(c0 - R0).max(); print("REPRO gate: CODEX w/o F32 vs stored max diff %.2e" % d, flush=True)
                assert d < 1e-9, "reproduction failed"; checked = True
            trF = pd.concat([tr, F32.reindex(columns=tr.columns)], ignore_index=True)
            for w in WS:
                wt = np.concatenate([wp[tm], np.full(len(F32), w)])
                out["codex_F32_%s" % w] = fit_predict(trF, va, wt, FC + ["is_f32"], PC + ["is_f32"])
            out.to_csv(path, index=False); print("%s/%d done" % (name, k), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261016)
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))])
    G = G.merge(C, on=["validator", "row_id"], how="left"); assert G["codex_F32_0.3"].notna().all()
    G = G[[(f, d) not in RD for f, d in zip(G.farm, G.day)]].copy()
    G["g"] = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    for w in WS:
        cu = G.g * G.codex_REF + (1 - G.g) * G["codex_F32_%s" % w]
        G["codex_C%s" % w] = cu
        print("\n==== CF(w=%s), COLD scope" % w); ok = True; rel = {v: [] for v in VALS}; pw = {v: [] for v in VALS}
        for s in (7, 101):
            for f in "AB":
                line = "seed %3d PFN %s |" % (s, f)
                for v in VALS:
                    X = G[G.validator == v]; y = X.sub_temp.values
                    bl = lambda c: (.4 * X["base_REF_%d" % s] + (.2 + .4 * (1 - X.g)) * X[c] + .4 * X.g * X["pfn_%s" % f]).values
                    a, b = bl("codex_REF"), bl("codex_C%s" % w); p = boot(X, a, b, rng)
                    rel[v].append(r(b - y) / r(a - y) - 1); pw[v].append(p)
                    m1 = X.g.values < 1; m6 = X.in_temp.values < 6; L = X.day.values >= 179
                    line += " %s %+.2f%% P %.3f (g<1 %.3f->%.3f n%d, <6C %s, 2차 %+.2f%%) |" % (
                        v, 100 * rel[v][-1], p, r((a - y)[m1]), r((b - y)[m1]), m1.sum(),
                        "%.2f->%.2f" % (r((a - y)[m6]), r((b - y)[m6])) if m6.any() else "-", 100 * (r((b - y)[L]) / r((a - y)[L]) - 1))
                print(line)
        for v in VALS:
            m = np.mean(rel[v])
            if v == "EXT10":
                ok &= all(x < 0 for x in rel[v]) and max(pw[v]) < ALPHA
            else:
                ok &= not (m > 0 and (1 - min(pw[v])) < ALPHA)
            print("  %-6s seed-mean %+.2f%%  P(worse) max %.4f" % (v, 100 * m, max(pw[v])))
        print("VERDICT CF(w=%s): %s" % (w, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT14_SUM") != "1":
        run()
    summarize()
