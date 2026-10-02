# -*- coding: utf-8 -*-
"""Runner for PAR1 (rules unchanged: ec2_PAR1_parity_prev_label_v1.py, commit 0f3654b).
Execution changes only (2026-10-02 집 클로드), after the first run hit the tool's
background time limit with nothing saved:
  * the R3S baseline is NOT refitted: it is read from DC5 (main validators) and EL1
    OOF files - same recipe, deterministic (DC5/EL1 reproduce stored R3 to 1e-15)
  * one checkpoint file per fold in local/par1_ckpt/ (resume-safe)
  * meant to be started as a detached process.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec2_PAR1b_runner.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("par1", os.path.join(HERE, "ec2_PAR1_parity_prev_label_v1.py"))
par1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(par1)
dc4, dc5, p3, core = par1.dc4, par1.dc5, par1.p3, par1.core
SEEDS = par1.SEEDS
CK = os.path.join(env.LOCAL, "par1_ckpt")


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    cal = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_9_days.csv"))[["farm", "day", "cal"]]
    lab = lab.merge(cal, on=["farm", "day"], how="left")
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    BS = [c for c in core.BASE if c != "day"] + ["season"]
    FP, BP = FS + ["par_ec", "par_gap"], BS + ["par_ec", "par_gap"]
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
        tr, va = par1.add_feats(lab[tr_m], lab[va_m], wv)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec", "cal", "par_gap"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["par_%d" % s] = dc5.r3(tr, va, s, FP, BP)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    # assemble + judge
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"})
    e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    O.to_csv(os.path.join(env.LOCAL, "ec2_PAR1_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261002)
    allbetter = True
    print("\npooled RMSE R3S -> R3S + parity prev label")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        g = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(g["r3s_%d" % s] - g.sub_ec), r(g["par_%d" % s] - g.sub_ec)
            if v != "EL1":
                allbetter &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy()
    D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["par_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    E = O[O.validator == "EL1"]
    elok = all(r(E["par_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    for nm, m in (("DIAG10 late", D.day >= 179), ("DIAG10 early", D.day < 179)):
        g = D[m]
        print("  %-13s R3S %.4f PAR %.4f" % (nm, np.mean([r(g["r3s_%d" % s] - g.sub_ec) for s in SEEDS]), np.mean([r(g["par_%d" % s] - g.sub_ec) for s in SEEDS])))
    ok = allbetter and all(p < 0.025 for p in ps) and elok
    print("\nPAR1 decision:", "PASS" if ok else "FAIL", "(rule %s, EL1 %s)" % (allbetter and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
