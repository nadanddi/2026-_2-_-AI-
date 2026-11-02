# -*- coding: utf-8 -*-
"""EC stage-3 DB1: separate R3 models per 동 (fixed before running; 2026-10-04 집 클로드).
Basis: 6.208 - EC level persists within the true 동 (.89-.94) and high EC is mostly 동 B;
the single model may average the two 동s' level relations.
Per outer fold: training records split by 동 (pair role: first A / second B; singletons
by the h=23 input classifier, se3.structure()); DC5 R3 recipe fitted separately on A
rows and on B rows (season, FS/BS as DC5).  Validation rows: both models predict;
blend by the causal pB (exact weather match to the previous record over hours 0..h ->
1, else per-hour input classifier):  pred = pB * R3_B + (1 - pB) * R3_A.
Judge (EC rule 2026-10-04, 6.247): every seed x DIAG10/A/B better, DIAG10 P(worse) <
.025; guard: EL1 or DIAG10 pass-2 rows worse by >= 2 % -> HOLD; EXT reported.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_DB1_dong_split_models_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("se3", os.path.join(HERE, "ec3_SE3_same_dong_residual_carry_v1.py"))
se3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(se3)
dc5, dc4, p3, core = se3.dc5, se3.dc4, se3.p3, se3.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "db1_ckpt")
TAG = "db"


def main():
    os.makedirs(CK, exist_ok=True)
    pB, rec = se3.structure()
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab["dong"] = [rec.loc[(f, d), "dong"] for f, d in zip(lab.farm, lab.day)]
    lab["pB"] = pB.reindex(lab.row_id).values
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    BS = [c for c in core.BASE if c != "day"] + ["season"]
    el = []
    for f in ("F13", "F47"):
        days = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(days), 5):
            el.append(("EL1", len(el), {(f, int(d)) for d in days[k:k + 5]}))
    for name, i, vd in list(fds) + el:
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
        frame = va[["row_id", "farm", "day", "hour", "sub_ec", "pB"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        trA, trB = tr[tr.dong == "A"], tr[tr.dong == "B"]
        for s in SEEDS:
            pa = dc5.r3(trA, va, s, FS, BS); pb = dc5.r3(trB, va, s, FS, BS)
            frame["%s_%d" % (TAG, s)] = va.pB.values * pb + (1 - va.pB.values) * pa
        frame.to_csv(path, index=False)
        print("%s/%d done (train A %d / B %d days)" % (name, i, trA[["farm", "day"]].drop_duplicates().shape[0], trB[["farm", "day"]].drop_duplicates().shape[0]), flush=True)
    judge_report(TAG, "DB1")


def judge_report(TAG, NAME):
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    O.to_csv(os.path.join(env.LOCAL, "ec3_%s_all.csv" % NAME), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261004)
    judge = True
    print("\npooled RMSE R3S -> %s (judge: DIAG10/A/B; others reported)" % NAME)
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]; cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["%s_%d" % (TAG, s)] - G.sub_ec)
            if v in ("DIAG10", "A", "B"):
                judge &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["%s_%d" % (TAG, s)] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    L = D.day >= 179; E = O[O.validator == "EL1"]; guard = []
    for s in SEEDS:
        guard.append(100 * (r(E["%s_%d" % (TAG, s)] - E.sub_ec) / r(E["r3s_%d" % s] - E.sub_ec) - 1))
        guard.append(100 * (r((D["%s_%d" % (TAG, s)] - D.sub_ec)[L]) / r((D["r3s_%d" % s] - D.sub_ec)[L]) - 1))
    hi = D.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1
    print("  DIAG10 late R3S %.4f -> %s %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), NAME, np.mean([r((D["%s_%d" % (TAG, s)] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 high-EC days (rows %d): label %.3f  R3S %.3f  %s %.3f" % (hi.sum(), D.sub_ec[hi].mean(), D.r3s_7[hi].mean(), NAME, D["%s_7" % TAG][hi].mean()))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("  guard changes (EL1, DIAG10-late per seed, %%): %s" % [round(x, 2) for x in guard])
    ok = judge and all(p < 0.025 for p in ps); hold = any(x >= 2.0 for x in guard)
    print("\n%s decision:" % NAME, ("HOLD (guard)" if hold else "PASS") if ok else "FAIL", "(judge %s, guard trip %s)" % (ok, hold))


if __name__ == "__main__":
    main()
