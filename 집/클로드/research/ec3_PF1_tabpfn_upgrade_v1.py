# -*- coding: utf-8 -*-
"""EC stage-3 PF1: TabPFN upgrades (the strongest member; WT1/WT2 6.355/6.369).  Fixed before running;
2026-10-07 집 클로드 (user: look for bigger gains).  GPU for speed only.
Variants (TabPFN v2, n_estimators 4, float32, contexts 5..8 averaged, random training rows):
  A  baseline: 2000-row context, FULL + season (38 features)                        [as submitted]
  B  6000-row context, same features
  C  2000-row context, FULL + season + DP1 (47)
  D  2000-row context, FULL + season + 7 reference-anchor features (AF0: SG2-style hour-causal search over
     training reference records; TRAINING rows exclude their own +-3 record days)
Blend for judging: clip(shrink(0.6 R3_DP1 + 0.4 PFN)) per seed 47 / 1414 / 6464 (R3 members from WT0
checkpoints; adopted share .4).
Sets: TM (111 test-matched days, DIAG10), P2LOO pass-2, EL1 pass-2.
Decision (fixed, k = 3, alpha .025/3): a variant replaces A iff every seed improves over A on all three sets
AND the TM seed-mean farm x 5-day block bootstrap P(worse) < .0083; if several, the largest TM gain.
Run (checkpointed):  PYTHONPATH="" py -3.12 -u ec3_PF1_tabpfn_upgrade_v1.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd, torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("af0", os.path.join(HERE, "ec3_AF0_anchor_features_v1.py"))
af0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(af0)
sg2, dc4, p3, core = af0.sg2, af0.dc4, af0.p3, af0.core
spec2 = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(dp1)
SEEDS = (47, 1414, 6464); CK0 = os.path.join(env.LOCAL, "wt0_ckpt"); CK = os.path.join(env.LOCAL, "pf1_ckpt")


def pfn(Xc, yc, Xq, seed):
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                   ignore_pretraining_limits=True, inference_precision=torch.float32)
    m.fit(Xc, yc)
    out = []
    for i in range(0, len(Xq), 2000):
        out.append(np.asarray(m.predict(Xq[i:i + 2000]), float))
    return np.concatenate(out)


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    F38 = [c for c in core.FULL if c != "day"] + ["season"]; F47 = F38 + dp1.NEW
    R, WV, hrs, SIG = sg2.prepare_structure()
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
        ref = {(f, int(d)) for f, d in labset if (f, int(d)) not in vd}
        cal = sg2.ref_calendar(R, WV, ref)
        tq = [(f, int(d)) for f, d in tdays.itertuples(index=False)]; vq_ = [(f, int(d)) for f, d in vdays[["farm", "day"]].itertuples(index=False)]
        Aft = af0.anchor_features(tq + vq_, set(tq), R, WV, hrs, SIG, ec, ref, cal, lockd)
        tr = tr.merge(Aft, on=["farm", "day", "hour"], how="left").set_index(tr.index)
        va = va.merge(Aft, on=["farm", "day", "hour"], how="left").set_index(va.index)
        F_D = F38 + af0.AF
        frame = pd.read_csv(os.path.join(CK0, "%s_%d.csv" % (name, i)))
        assert (frame.row_id.values == va.row_id.values).all()
        y = tr.sub_ec.to_numpy(float)
        for tag, cols, nctx in (("A", F38, 2000), ("B", F38, 6000), ("C", F47, 2000), ("D", F_D, 2000)):
            Xtr, Xva = tr[cols].to_numpy(np.float32), va[cols].to_numpy(np.float32)
            ps = []
            for c in (5, 6, 7, 8):
                ix = np.random.default_rng(c).choice(len(tr), size=min(nctx, len(tr)), replace=False)
                ps.append(pfn(Xtr[ix], y[ix], Xva, c))
            frame["pfn" + tag] = core.shrink(np.mean(ps, axis=0), va)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O.to_csv(os.path.join(env.LOCAL, "ec3_PF1_all.csv"), index=False)
    TM = pd.read_csv(os.path.join(env.LOCAL, "tm1_set_v1.csv")); TMs = set(zip(TM.farm, TM.day))
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))

    def bl(G, tag, s):
        r3 = .6 * G["et_%d" % s] + .3 * G["lgb_%d" % s] + .1 * G["mlp_%d" % s]
        return np.clip(.6 * r3 + .4 * G["pfn" + tag], G.lo, G.hi)
    sets = {"TM": O[(O.validator == "DIAG10") & np.array([(f, d) in TMs for f, d in zip(O.farm, O.day)])],
            "P2LOO": O[(O.validator == "P2LOO") & (O.day >= 179)], "EL1": O[(O.validator == "EL1") & (O.day >= 179)]}
    print("\nTabPFN alone (shrunk) RMSE: " + " | ".join("%s " % nm + " ".join("%s %.4f" % (t, r(G["pfn" + t] - G.sub_ec)) for t in "ABCD") for nm, G in sets.items()))
    best = None
    for tag in "BCD":
        ok = True; print("\n%s vs A (0.6 R3 + 0.4 PFN)" % tag)
        for nm, G in sets.items():
            cells = []
            for s in SEEDS:
                a, b = r(bl(G, "A", s) - G.sub_ec), r(bl(G, tag, s) - G.sub_ec); ok &= b < a
                cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
            G2 = G.assign(dm=G.groupby(["farm", "day"]).sub_ec.transform("mean")); nn = (G2.dm < 1).values
            bm = np.mean([bl(G2, "A", s) for s in SEEDS], axis=0); cm = np.mean([bl(G2, tag, s) for s in SEEDS], axis=0)
            print("  %-6s %s | normal %.4f->%.4f high %.4f->%.4f" % (nm, "  ".join(cells), r((bm - G2.sub_ec)[nn]), r((cm - G2.sub_ec)[nn]),
                  r((bm - G2.sub_ec)[~nn]), r((cm - G2.sub_ec)[~nn])))
        T = sets["TM"].copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
        bm = np.mean([bl(T, "A", s) for s in SEEDS], axis=0); cm = np.mean([bl(T, tag, s) for s in SEEDS], axis=0)
        dd = pd.Series((cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2, index=T.index).groupby(T.cl).agg(["sum", "count"])
        sm, n = dd["sum"].values, dd["count"].values
        idx = np.random.default_rng(20261007).integers(0, len(sm), (20000, len(sm)))
        p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()); gain = r(cm - T.sub_ec) / r(bm - T.sub_ec) - 1
        print("  TM P(worse) %.4f -> %s" % (p, "PASS" if ok and p < .0083 else "FAIL"))
        if ok and p < .0083 and (best is None or gain < best[1]):
            best = (tag, gain)
    print("\nPF1 decision:", "use variant %s" % best[0] if best else "keep A")


if __name__ == "__main__":
    main()
