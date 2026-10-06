# -*- coding: utf-8 -*-
"""EC stage-3 LH1: give the LightGBM and MLP members the FULL feature set (47: + hour-0 values, today-so-far
means and zero shares of the actuators) instead of BASE (23).  Fixed before running; 2026-10-07 집 클로드.
Motivation: LGB/MLP never saw the actuator history (6.288 note); WT0 hinted LGB weight hurts pass 2.
Members: ET (FULL_R3) and PFN reused from WT0/WT1 checkpoints (seeds 47 / 1414 / 6464, PFN contexts 5-8);
LGB-tweedie and MLP refitted here with FULL_R3 for the same folds and seeds.
CUR  = clip(shrink(0.8 (.6 ET + .3 LGB_base + .1 MLP_base) + 0.2 PFN))
LH1  = clip(shrink(0.8 (.6 ET + .3 LGB_full + .1 MLP_full) + 0.2 PFN))      (also reported with PFN 0.4)
Sets: TM (111 test-matched days, DIAG10), P2LOO pass-2, EL1 pass-2.
Decision (fixed): LH1 adopted iff every seed improves on all three sets AND TM seed-mean farm x 5-day block
bootstrap P(worse) < .025.
Run (checkpointed; parallel runner allowed):  PYTHONPATH="" py -3.12 -u ec3_LH1_lgb_mlp_full_features_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
SEEDS = wt0.SEEDS; CK1 = os.path.join(env.LOCAL, "wt1_ckpt"); CK = os.path.join(env.LOCAL, "lh1_ckpt")


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
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
        frame = pd.read_csv(os.path.join(CK1, "%s_%d.csv" % (name, i)))
        assert (frame.row_id.values == va.row_id.values).all()
        for s in SEEDS:
            l = core.predict_model(core.lg(s, "tweedie"), tr, va, FS)
            mlp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                core.MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3, max_iter=800,
                                                  early_stopping=True, n_iter_no_change=25, validation_fraction=.12, random_state=s))
            m = core.predict_model(mlp, tr, va, FS)
            frame["lgbF_%d" % s] = core.shrink(l, va); frame["mlpF_%d" % s] = core.shrink(m, va)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O.to_csv(os.path.join(env.LOCAL, "ec3_LH1_all.csv"), index=False)
    TM = pd.read_csv(os.path.join(env.LOCAL, "tm1_set_v1.csv")); TMs = set(zip(TM.farm, TM.day))
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))

    def cfg(G, which, s, w=.2):
        l, m = ("lgbF_%d", "mlpF_%d") if which == "LH1" else ("lgb_%d", "mlp_%d")
        r3 = .6 * G["et_%d" % s] + .3 * G[l % s] + .1 * G[m % s]
        return np.clip((1 - w) * r3 + w * G.pfn, G.lo, G.hi)
    sets = {"TM": O[(O.validator == "DIAG10") & np.array([(f, d) in TMs for f, d in zip(O.farm, O.day)])],
            "P2LOO": O[(O.validator == "P2LOO") & (O.day >= 179)], "EL1": O[(O.validator == "EL1") & (O.day >= 179)]}
    ok = True
    for w in (.2, .4):
        print("\nPFN share %.1f: CUR -> LH1" % w)
        for nm, G in sets.items():
            cells = []
            for s in SEEDS:
                a, b = r(cfg(G, "CUR", s, w) - G.sub_ec), r(cfg(G, "LH1", s, w) - G.sub_ec)
                if w == .2:
                    ok &= b < a
                cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
            G2 = G.assign(dm=G.groupby(["farm", "day"]).sub_ec.transform("mean"))
            bm = np.mean([cfg(G2, "CUR", s, w) for s in SEEDS], axis=0); cm = np.mean([cfg(G2, "LH1", s, w) for s in SEEDS], axis=0)
            nn = (G2.dm < 1).values
            print("  %-6s %s | normal %.4f->%.4f high %.4f->%.4f" % (nm, "  ".join(cells), r((bm - G2.sub_ec)[nn]), r((cm - G2.sub_ec)[nn]),
                  r((bm - G2.sub_ec)[~nn]), r((cm - G2.sub_ec)[~nn])))
    T = sets["TM"].copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
    bm = np.mean([cfg(T, "CUR", s) for s in SEEDS], axis=0); cm = np.mean([cfg(T, "LH1", s) for s in SEEDS], axis=0)
    dd = pd.Series((cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2, index=T.index).groupby(T.cl).agg(["sum", "count"])
    sm, n = dd["sum"].values, dd["count"].values
    idx = np.random.default_rng(20261007).integers(0, len(sm), (20000, len(sm)))
    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    print("\nTM seed-mean (PFN .2) %.4f -> %.4f  P(worse) %.4f" % (r(bm - T.sub_ec), r(cm - T.sub_ec), p))
    print("LH1 decision:", "ADOPT" if ok and p < .025 else "KEEP BASE", "(all better %s, P %.4f)" % (ok, p))


if __name__ == "__main__":
    main()
