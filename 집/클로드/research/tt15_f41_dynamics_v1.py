# -*- coding: utf-8 -*-
"""TT15: borrow the substrate FOLLOW DYNAMICS of F41 for the cold side (2026-10-10 집 클로드, user: "온도가 이런 식으로 내려가면
어느 정도 더 내려가거나 복귀하더라 이런 거 학습해보면 어때" -> "진행해봐").  TD28: F41 follows air like F13/F47 (k_up .18 / k_down
.16 vs .20/.16; 6 h fall ratio .67 vs .69) and has 526 rows with in_temp < 8 (F13/F47 < 80); its level differs (sub - in
+1.17 vs -0.72) and it never goes below 6 C.  F32 is NOT used (opposite dynamics).  Fixed before running.
 A1: CODEX refit on (fold's F13/F47 MASK-world training rows) + ALL F41 rows (weight 0.3, is_f41 indicator in physics and
     residual features; F41 has only in_temp/in_hum/in_co2 of the CODEX inputs -> others NaN; season NaN).  Level adapter:
     F41 labels shifted by (mean(sub-in) of F13/F47 training) - (mean(sub-in) of F41)  (one global constant).
 A2: new causal feature 'asym' added to CODEX physics AND residual features: per record day (reset at hour 0 to that hour's
     in_temp), s_t = s_{t-1} + k * (in_t - s_{t-1}) with k = k_up if in_t > s_{t-1} else k_down; warm rates (in_t >= 8)
     = F13/F47 pooled training fit (k_up .196, k_down .159, TD28), cold rates (in_t < 8) = F41 cold fit (.374, .148).
     Uses only the same record's current and earlier inputs.
 Both applied on the COLD side only: CODEX_used = g * CODEX_REF + (1 - g) * CODEX_cand.  k = 2, alpha .0125.
RULE (user temperature scope rule, COLD): EXT10 primary - every BASE seed {7,101} x PFN family {A,B} better and farm x 5-day
  block bootstrap P(worse) < .0125; DIAG10/EXT12/EL1 fail iff seed-mean worse and share(better) < .0125.  Rough days excluded
  from scoring.  Sanity: CODEX without changes reproduces the stored codex_REF.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt15_f41_dynamics_v1.py
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
CK = os.path.join(env.LOCAL, "tt15_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt15_season"
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
VALS = ("DIAG10", "EXT10", "EXT12", "EL1"); ALPHA = .0125; CANDS = ("A1", "A2")
KW = (.196, .159); KC = (.374, .148)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def other_frame(FC, PC, farm, shift):
    """Same computation as resid_reset_features.build_features (which filters to F13/F47), applied to another greenhouse."""
    tX, ty, sX = common.load_raw()
    a = R._read(tX[tX.farm == farm])
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
    F["sub_temp"] = F.sub_temp + shift
    F["is_f32"] = 1.0
    return F.reset_index(drop=True)


def asym(df):
    """per record day, reset at hour 0; inputs of the same record up to the current hour."""
    out = np.full(len(df), np.nan)
    d = df.sort_values(["farm", "day", "hour"])
    for _, q in d.groupby(["farm", "day"]):
        s_ = np.nan
        for i, t in zip(q.index, q.in_temp.values):
            if np.isnan(t):
                out[df.index.get_loc(i)] = s_; continue
            if np.isnan(s_):
                s_ = t
            else:
                ku, kd = KC if t < 8 else KW
                s_ = s_ + (ku if t > s_ else kd) * (t - s_)
            out[df.index.get_loc(i)] = s_
    return out


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
    tX, ty, _ = common.load_raw()
    t = tX.merge(ty[["row_id", "sub_temp"]], on="row_id")
    lvl_t = (t[t.farm.isin(["F13", "F47"])].sub_temp - t[t.farm.isin(["F13", "F47"])].in_temp).mean()
    lvl_41 = (t[t.farm == "F41"].sub_temp - t[t.farm == "F41"].in_temp).mean()
    F41 = other_frame(FC, PC, "F41", lvl_t - lvl_41)
    print("F41 rows %d, level shift %+.3f, non-null share of CODEX features %.2f" % (len(F41), lvl_t - lvl_41, F41[FC].notna().mean().mean()), flush=True)
    pfn = pfn.copy(); pfn["asym"] = asym(pfn)
    print("asym feature: corr with in_temp_reset3 %.3f" % np.corrcoef(pfn.asym.fillna(pfn.in_temp), pfn.in_temp_reset3.fillna(pfn.in_temp))[0, 1], flush=True)
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
            tr, va = W.season_fold(pfn[tm], pfn[vm], wv, "tt15_" + name, k)
            tr = tr.reset_index(drop=True).assign(is_f32=0.0); va = va.reset_index(drop=True).assign(is_f32=0.0)
            out = pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values})
            if not checked:
                c0 = fit_predict(tr, va, wp[tm], FC, PC)
                R0 = ref_all[ref_all.validator == name].set_index("row_id").reindex(out.row_id).codex_REF.values
                d = np.abs(c0 - R0).max(); print("REPRO gate: CODEX unchanged vs stored max diff %.2e" % d, flush=True)
                assert d < 1e-9, "reproduction failed"; checked = True
            trF = pd.concat([tr, F41.reindex(columns=tr.columns)], ignore_index=True)
            wt = np.concatenate([wp[tm], np.full(len(F41), 0.3)])
            out["codex_A1"] = fit_predict(trF, va, wt, FC + ["is_f32"], PC + ["is_f32"])
            out["codex_A2"] = fit_predict(tr, va, wp[tm], FC + ["asym"], PC + ["asym"])
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
    G = G.merge(C, on=["validator", "row_id"], how="left"); assert G["codex_A2"].notna().all()
    G = G[[(f, d) not in RD for f, d in zip(G.farm, G.day)]].copy()
    G["g"] = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    for w in CANDS:
        cu = G.g * G.codex_REF + (1 - G.g) * G["codex_%s" % w]
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
        print("VERDICT %s: %s" % (w, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT15_SUM") != "1":
        run()
    summarize()
