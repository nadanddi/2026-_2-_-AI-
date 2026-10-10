# -*- coding: utf-8 -*-
"""TT16: H-P3 re-test - pretrained other-greenhouse model as a CODEX feature (2026-10-10 집 클로드, user: "1번").
Original H-P3 (re14, catalog 6b.29): all six cells improved -0.1..-0.3% but rejected on DIAG10 P .19-.22 (critic 10-10:
flawed by low power - benefit expected on the cold side - and by the leaky old validators).  Fixed before running.
Pretrain: re13 recipe unchanged (re13_pooled51_v1.build_panel features: in_temp/in_hum/in_co2 histories, MASK-style panel
  from train_X only; PARAMS; all 49 non-target greenhouses incl. F32; no farm category).  NEW pretrain seeds 202 / 303.
  pre49 = its prediction for F13/F47 rows; added as ONE column to the CODEX LGB residual features (physics Ridge unchanged).
Candidates (k = 2, alpha .0125):
  HP-ALL : CODEX_used = CODEX_pre on every row            -> scope ALL  (primary DIAG10 AND EXT10)
  HP-COLD: CODEX_used = g * CODEX_REF + (1 - g) * CODEX_pre -> scope COLD (primary EXT10)
RULE (user temperature scope rule): primary validator(s): every pretrain seed {202,303} x BASE seed {7,101} x PFN family {A,B}
  (8 cells) better and farm x 5-day block bootstrap P(worse) < .0125 in every cell; others fail iff seed-mean worse and
  share(better) < .0125.  Rough days excluded from scoring.  Sanity: CODEX without pre49 reproduces the stored codex_REF.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tt16_hp3_pretrain_feature_v1.py
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
CK = os.path.join(env.LOCAL, "tt16_ckpt")
W.OUT = pathlib.Path(env.LOCAL) / "tt16_season"
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
VALS = ("DIAG10", "EXT10", "EXT12", "EL1"); ALPHA = .0125; PSEEDS = (202, 303)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


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
    import re13_pooled51_v1 as R13
    P = R13.build_panel()
    feats = [c for c in P.columns if c not in ("farm", "day", "hour", "row_id", "sub_temp", "farm_cat")]
    oth = P[~P.farm.isin(R13.TARGETS)]
    tgt = P[P.farm.isin(R13.TARGETS)].set_index("row_id").reindex(pfn.row_id)
    print("other greenhouses %d rows, target rows matched %d / %d" % (len(oth), tgt[feats[0]].notna().sum(), len(pfn)), flush=True)
    pre = {}
    for sd in PSEEDS:
        m = R13.fit(R13.PARAMS, oth[feats], oth.sub_temp.values, sd)
        pre[sd] = m.predict(tgt[feats])
        print("pretrain seed %d: alone RMSE on F13/F47 training rows %.3f" % (sd, np.sqrt(np.nanmean((pre[sd] - pfn.sub_temp.values) ** 2))), flush=True)
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
            out = pd.DataFrame({"validator": name, "row_id": lab.row_id[vm].values})
            for sd in PSEEDS:
                pf = pfn.copy(); pf["pre49"] = pre[sd]
                tr, va = W.season_fold(pf[tm], pf[vm], wv, "tt16_" + name, k)
                if not checked:
                    c0 = fit_predict(tr, va, wp[tm], FC, PC)
                    R0 = ref_all[ref_all.validator == name].set_index("row_id").reindex(out.row_id).codex_REF.values
                    d = np.abs(c0 - R0).max(); print("REPRO gate: CODEX w/o pre49 vs stored max diff %.2e" % d, flush=True)
                    assert d < 1e-9, "reproduction failed"; checked = True
                out["codex_pre_%d" % sd] = fit_predict(tr, va, wp[tm], FC + ["pre49"], PC)
            out.to_csv(path, index=False); print("%s/%d done" % (name, k), flush=True)


def boot(X, a, b, rng):
    cl = (X.farm + "_" + (X.day // 5).astype(str)).values
    dd = pd.Series((b - X.sub_temp.values) ** 2 - (a - X.sub_temp.values) ** 2).groupby(cl).agg(["sum", "count"])
    idx = rng.integers(0, len(dd), (20000, len(dd)))
    return float(((dd["sum"].values[idx].sum(1) / dd["count"].values[idx].sum(1)) >= 0).mean())


def summarize():
    rng = np.random.default_rng(20261017)
    G = pd.concat([pd.read_csv(os.path.join(CK1, f)) for f in sorted(os.listdir(CK1))], ignore_index=True)
    C = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))])
    G = G.merge(C, on=["validator", "row_id"], how="left"); assert G["codex_pre_303"].notna().all()
    G = G[[(f, d) not in RD for f, d in zip(G.farm, G.day)]].copy()
    G["g"] = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    for cand, prim in (("HP-ALL", ("DIAG10", "EXT10")), ("HP-COLD", ("EXT10",))):
        print("
==== %s (primary %s)" % (cand, "+".join(prim))); ok = True; rel = {v: [] for v in VALS}; pw = {v: [] for v in VALS}
        for ps in PSEEDS:
            cu = G["codex_pre_%d" % ps] if cand == "HP-ALL" else G.g * G.codex_REF + (1 - G.g) * G["codex_pre_%d" % ps]
            G["cu"] = cu
            for s in (7, 101):
                for f in "AB":
                    line = "pre %d seed %3d PFN %s |" % (ps, s, f)
                    for v in VALS:
                        X = G[G.validator == v]; y = X.sub_temp.values
                        bl = lambda c: (.4 * X["base_REF_%d" % s] + (.2 + .4 * (1 - X.g)) * X[c] + .4 * X.g * X["pfn_%s" % f]).values
                        a, b = bl("codex_REF"), bl("cu"); p = boot(X, a, b, rng)
                        rel[v].append(r(b - y) / r(a - y) - 1); pw[v].append(p)
                        m1 = X.g.values < 1; m6 = X.in_temp.values < 6; L = X.day.values >= 179
                        line += " %s %+.2f%% P %.3f (g<1 %.3f->%.3f, <6C %s, 2차 %+.2f%%) |" % (v, 100 * rel[v][-1], p, r((a - y)[m1]), r((b - y)[m1]),
                                 "%.2f->%.2f" % (r((a - y)[m6]), r((b - y)[m6])) if m6.any() else "-", 100 * (r((b - y)[L]) / r((a - y)[L]) - 1))
                    print(line)
        for v in VALS:
            m = np.mean(rel[v])
            if v in prim:
                ok &= all(x < 0 for x in rel[v]) and max(pw[v]) < ALPHA
            else:
                ok &= not (m > 0 and (1 - min(pw[v])) < ALPHA)
            print("  %-6s seed-mean %+.2f%%  all better %s  P(worse) max %.4f" % (v, 100 * m, all(x < 0 for x in rel[v]), max(pw[v])))
        print("VERDICT %s: %s" % (cand, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if os.environ.get("TT16_SUM") != "1":
        run()
    summarize()
