# -*- coding: utf-8 -*-
"""CMP2: the 0.083 setting (HX1b: pass-2 NORMAL days, P2LOO) with four models side by side.  (2026-10-09 집 클로드, user:
"0.083 정도 나왔던 상황에 두 모델을 두고 실험하면 성능이 어떻게 나와?").  Descriptive comparison, no adoption.
Validation days: the remaining (non-high) labelled days; P2LOO = each pass-2 normal day alone (41 folds), DIAG10 = house
folds minus high days (pass-1 + pass-2 normal days).  High day = labelled day mean >= 1.2 (HX1 HIGH set, 26 non-lock days).
Exclusions as HX1b: same farm +-1, lock-40 +-1, other farm d-3..d+3.
Models (submission seeds 7 / 101 / 2024; PFN contexts 1-4 as the submission; GPU only for speed):
  R3_NOHIGH    R3 trained without high days                                   (the 0.083 recipe)
  FULL_NOHIGH  clip(shrink(.8 R3 + .2 PFN)) + SG2, all trained / referenced without high days  (= submission_15 config)
  R3_WITH      R3 trained with high days
  FULL_WITH    same pipeline with high days in training and SG2 reference        (= submission_14 config)
Scored on normal days only (the 0.083 setting).  The cost on high days is NOT visible here.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u cmp2_sub14_vs_sub15_normal_days_v1.py {2|1|sum}   (2 = P2LOO first)
"""
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "2"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ct2", os.path.join(HERE, "ct2_thermal_on_submission_pipeline_v1.py"))
ct2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct2)
sg2, wt0, dp1, dc4, p3, core = ct2.sg2, ct2.wt0, ct2.dp1, ct2.dc4, ct2.p3, ct2.core
SEEDS = (7, 101, 2024); CTX = (1, 2, 3, 4)
CK = os.path.join(env.LOCAL, "cmp2_ckpt")
W = (.6, .3, .1)
OTHER = {"F13": "F47", "F47": "F13"}
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    high = {(f, int(d)) for f, d in json.load(open(os.path.join(env.LOCAL, "hx1_day_sets.json"), encoding="utf-8"))["HIGH"]}
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    PF = [c for c in core.FULL if c != "day"] + ["season"]
    R, WV, hrs, SIG = sg2.prepare_structure()
    LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
    lockd = {(s["farm"], int(s["day"])) for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]} | set(lock)
    ecall = lab.groupby(["farm", "day"]).sub_ec.mean()
    normal = lab[[(f, int(d)) not in high for f, d in zip(lab.farm, lab.day)]]
    if MODE == "2":
        folds = [("P2LOO", 300 + k, {x}) for k, x in enumerate(sorted({(f, int(d)) for f, d in zip(normal.farm, normal.day) if d >= 179}))]
    else:
        folds = [(n, i, {k for k in vd if k not in high}) for n, i, vd in fds if n == "DIAG10"]
    for name, i, vd in folds:
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = ({(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)}
                | {(f, d + j) for f, d in lock for j in (-1, 0, 1)})
        tr_all = lab[np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)]) & ~va_m]
        va0 = lab[va_m]
        G = va0[["row_id", "farm", "day", "hour", "sub_ec"]].copy().reset_index(drop=True)
        G["validator"], G["validation_fold"] = name, i
        for tag, tr in (("NOHIGH", tr_all[[(f, int(d)) not in high for f, d in zip(tr_all.farm, tr_all.day)]]), ("WITH", tr_all)):
            tr = tr.copy(); va = va0.copy()
            tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
            season, vq = dc4.season_index(tdays, vdays, wv)
            tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
            va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
            vaR = va.reset_index(drop=True)
            lo, hi = tr.sub_ec.min(), tr.sub_ec.max()
            Xtr, ytr, Xva = tr[PF].to_numpy(np.float32), tr.sub_ec.to_numpy(float), va[PF].to_numpy(np.float32)
            ps = []
            for c in CTX:
                ix = np.random.default_rng(c).choice(len(tr), min(2000, len(tr)), replace=False)
                ps.append(ct2.pfn(Xtr[ix], ytr[ix], Xva, c))
            pf = np.mean(ps, axis=0)
            ec = ecall if tag == "WITH" else ecall[[k not in high for k in ecall.index]]
            ref = {k for k in ec.index if k not in vd}; cal = sg2.ref_calendar(R, WV, ref)
            for s in SEEDS:
                e, l, m = wt0.members(tr, va, s, FS, BS)
                r3 = W[0] * e + W[1] * l + W[2] * m
                G["R3_%s_%d" % (tag, s)] = np.clip(core.shrink(r3, va), lo, hi)
                G["pre_%s_%d" % (tag, s)] = np.clip(ct2.shrink(.8 * r3 + .2 * pf, vaR), lo, hi)
                G["FULL_%s_%d" % (tag, s)] = np.clip(sg2.correction(G, "pre_%s_%d" % (tag, s), vd, lockd, ec, R, WV, hrs, SIG, ref, cal), lo, hi)
        G.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)


def summarize():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    print("folds:", G.groupby("validator").validation_fold.nunique().to_dict())
    for v, part in (("P2LOO", "pass-2 normal days"), ("DIAG10", "all normal days"), ("DIAG10", "pass-2 normal days")):
        g = G[G.validator == v]
        if part.startswith("pass-2"): g = g[g.day >= 179]
        if g.empty:
            continue
        y = g.sub_ec.to_numpy(float)
        print("\n== %s, %s (%d days)" % (v, part, g[["farm", "day"]].drop_duplicates().shape[0]))
        for c, lab_ in (("R3_NOHIGH", "① R3, 고EC 제외 (0.083 방식)"), ("FULL_NOHIGH", "② 전체 구성, 고EC 제외 (submission_15)"),
                        ("R3_WITH", "③ R3, 고EC 포함"), ("FULL_WITH", "④ 전체 구성, 고EC 포함 (submission_14)")):
            ps = [g["%s_%d" % (c, s)].to_numpy(float) for s in SEEDS]
            print("  %-42s RMSE %.4f  (seeds %s)  bias %+.3f" % (lab_, r(np.mean(ps, 0) - y), " ".join("%.4f" % r(p - y) for p in ps), np.mean(np.mean(ps, 0) - y)))


if __name__ == "__main__":
    if MODE != "sum":
        main()
    summarize()
