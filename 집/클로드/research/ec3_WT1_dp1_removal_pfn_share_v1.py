# -*- coding: utf-8 -*-
"""EC stage-3 WT1: two configuration questions raised by TM1 (6.353), judged on the TEST-MATCHED set
plus the pass-2 validators.  Fixed before running; 2026-10-06 집 클로드 (user: "1번 진행").
Configurations (final = clip(shrink(0.8 R3 + w PFN)) with w as stated, R3 = .6 ET + .3 LGB + .1 MLP):
  CUR  R3 with DP1 features (WT0 stored members, seeds 47 / 1414 / 6464) + 0.2 PFN   [= submission_13 core]
  A    R3 WITHOUT DP1 (season v2 features; refitted here, same seeds and folds)   + 0.2 PFN
  B3   CUR R3 + 0.3 PFN (0.7 / 0.3);   B4  CUR R3 + 0.4 PFN (0.6 / 0.4)
PFN: TabPFN v2 on FULL + season (as the model), 2000 random training rows, contexts 5..8 (never used in a
submission), n_estimators 4, float32, GPU (speed only), averaged.
Sets: TM (DIAG10 out-of-fold rows of the 111 test-matched days, tm1_set_v1.csv; both passes), P2LOO
pass-2 rows, EL1 pass-2 rows.
Decision per candidate (k = 3, alpha .025 / 3 = .0083): ADOPT iff (a) every seed improves on TM, (b) no
seed x {P2LOO, EL1} cell is worse by more than 1 %, (c) TM seed-mean farm x 5-day block bootstrap
P(worse) < .0083.  Candidates are judged against CUR independently.
Run (checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_WT1_dp1_removal_pfn_share_v1.py
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd, torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
SEEDS = wt0.SEEDS; CK0 = wt0.CK; CK = os.path.join(env.LOCAL, "wt1_ckpt")


def pfn(Xc, yc, Xq, seed):
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=4, random_state=seed,
                                                   ignore_pretraining_limits=True, inference_precision=torch.float32)
    m.fit(Xc, yc)
    return np.asarray(m.predict(Xq), float)


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"]; BS = [c for c in core.BASE if c != "day"] + ["season"]
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
        frame = pd.read_csv(os.path.join(CK0, "%s_%d.csv" % (name, i)))
        assert (frame.row_id.values == va.row_id.values).all()
        for s in SEEDS:
            e, l, m = wt0.members(tr, va, s, FS, BS)
            frame["nodp_%d" % s] = core.shrink(.6 * e + .3 * l + .1 * m, va)
        Xtr, ytr, Xva = tr[FS].to_numpy(np.float32), tr.sub_ec.to_numpy(float), va[FS].to_numpy(np.float32)
        ps = []
        for c in (5, 6, 7, 8):
            ix = np.random.default_rng(c).choice(len(tr), size=min(2000, len(tr)), replace=False)
            ps.append(pfn(Xtr[ix], ytr[ix], Xva, c))
        frame["pfn"] = core.shrink(np.mean(ps, axis=0), va)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    O = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    O.to_csv(os.path.join(env.LOCAL, "ec3_WT1_all.csv"), index=False)
    TM = pd.read_csv(os.path.join(env.LOCAL, "tm1_set_v1.csv")); TMs = set(zip(TM.farm, TM.day))
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))

    def cfg(G, name, s):
        r3dp = .6 * G["et_%d" % s] + .3 * G["lgb_%d" % s] + .1 * G["mlp_%d" % s]
        r3 = G["nodp_%d" % s] if name == "A" else r3dp
        w = {"CUR": .2, "A": .2, "B3": .3, "B4": .4}[name]
        return np.clip((1 - w) * r3 + w * G.pfn, G.lo, G.hi)
    sets = {"TM": O[(O.validator == "DIAG10") & np.array([(f, d) in TMs for f, d in zip(O.farm, O.day)])],
            "P2LOO": O[(O.validator == "P2LOO") & (O.day >= 179)], "EL1": O[(O.validator == "EL1") & (O.day >= 179)]}
    for nm, G in sets.items():
        print("%s rows %d days %d" % (nm, len(G), G.groupby(["farm", "day"]).ngroups))
    for cand in ("A", "B3", "B4"):
        print("\n%s vs CUR" % cand)
        ok_tm = True; ok_guard = True
        for nm, G in sets.items():
            cells = []
            for s in SEEDS:
                a, b = r(cfg(G, "CUR", s) - G.sub_ec), r(cfg(G, cand, s) - G.sub_ec)
                cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
                if nm == "TM":
                    ok_tm &= b < a
                else:
                    ok_guard &= b <= 1.01 * a
            print("  %-6s %s" % (nm, "  ".join(cells)))
        T = sets["TM"].copy(); T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
        bm = np.mean([cfg(T, "CUR", s) for s in SEEDS], axis=0); cm = np.mean([cfg(T, cand, s) for s in SEEDS], axis=0)
        dd = pd.Series((cm - T.sub_ec) ** 2 - (bm - T.sub_ec) ** 2, index=T.index).groupby(T.cl).agg(["sum", "count"])
        sm, n = dd["sum"].values, dd["count"].values
        idx = np.random.default_rng(20261006).integers(0, len(sm), (20000, len(sm)))
        p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
        T["dm"] = T.groupby(["farm", "day"]).sub_ec.transform("mean")
        nn = T.dm < 1
        print("  TM seed-mean %.4f -> %.4f, normal %.4f -> %.4f, P(worse) %.4f" % (r(bm - T.sub_ec), r(cm - T.sub_ec),
              r((bm - T.sub_ec)[nn]), r((cm - T.sub_ec)[nn]), p))
        print("  %s decision: %s (TM all seeds better %s, guards ok %s, P %.4f vs .0083)" % (cand, "ADOPT" if ok_tm and ok_guard and p < .0083 else "KEEP CUR", ok_tm, ok_guard, p))


if __name__ == "__main__":
    main()
