# -*- coding: utf-8 -*-
"""EC stage-3 ST5: causal 'second record of a same-date pair' flag as an R3S feature
(fixed before running; 2026-10-03 집 클로드).
Basis: C6.205 (a date has 1-2 records; identical-weather neighbours = same date,
second ~ 동 B).  MN1/MN2: pass-2 pair SECOND records are badly under-predicted
(EC RMSE .488 vs singles .206 on 7 DIAG10 days; label EC .98 vs first .47, vent 3.5
vs 10.4); 10 of the 60 evaluation days are pass-2 pair seconds.
Feature is2_h (row level): 1 if this record's outdoor weather (4 channels) at hours
0..h equals the previous record's (record d-1) at hours 0..h exactly, else 0 -> uses
only the same greenhouse's current and previous inputs (legal).  Added to FS and BS
of the DC5 R3 recipe (FULL/BASE - day + season).  Seeds 7/101/2024, all validators
+ EL1.  Baseline R3S = stored DC5 / EL1 OOF.
Rule (user's): all 3 seeds x 5 validators better and DIAG10 P(worse) < .025; EL1
better for all seeds also required.  Pass-2 pair-second rows reported separately.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_ST5_is_second_feature_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
spec8 = importlib.util.spec_from_file_location("st8", os.path.join(HERE, "ec3_ST8_dong_state_corrector_v1.py"))
st8 = importlib.util.module_from_spec(spec8); spec8.loader.exec_module(st8)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "st5_ckpt")


def main():
    os.makedirs(CK, exist_ok=True)
    A = st8.load_inputs()
    A["is2"] = st8.causal_is_second(A).astype(float)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.merge(A[["row_id", "is2"]], on="row_id", how="left")
    assert lab.is2.notna().all()
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season", "is2"]
    BS = [c for c in core.BASE if c != "day"] + ["season", "is2"]
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
        tdays = tr[["farm", "day"]].drop_duplicates()
        vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
        vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec", "is2"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            frame["st5_%d" % s] = dc5.r3(tr, va, s, FS, BS)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    roles = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"))
    O = O.merge(roles, on=["farm", "day"], how="left")
    O.to_csv(os.path.join(env.LOCAL, "ec3_ST5_all.csv"), index=False)
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261003)
    allb = True
    print("\npooled RMSE R3S -> R3S + causal is-second flag")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["st5_%d" % s] - G.sub_ec)
            if v != "EL1":
                allb &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["st5_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    E = O[O.validator == "EL1"]
    elok = all(r(E["st5_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    L = D.day >= 179
    print("  DIAG10 late R3S %.4f -> ST5 %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["st5_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    for nm, m in (("DIAG10 pass-2 seconds", (D.day >= 179) & (D.role == "second")), ("DIAG10 pass-1 seconds", (D.day < 179) & (D.role == "second")),
                  ("EL1 seconds", None)):
        G = D[m] if m is not None else E.merge(roles, on=["farm", "day"])[lambda x: x.role == "second"]
        if len(G):
            print("  %s (rows %d): R3S %.4f -> ST5 %.4f" % (nm, len(G), np.mean([r(G["r3s_%d" % s] - G.sub_ec) for s in SEEDS]), np.mean([r(G["st5_%d" % s] - G.sub_ec) for s in SEEDS])))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nST5 decision:", "PASS" if allb and all(p < 0.025 for p in ps) and elok else "FAIL",
          "(rule %s, EL1 %s)" % (allb and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
