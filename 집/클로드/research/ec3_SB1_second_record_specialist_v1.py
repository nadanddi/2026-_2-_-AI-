# -*- coding: utf-8 -*-
"""EC stage-3 SB1: specialist model for same-date pair SECOND records (fixed before
running; 2026-10-03 집 클로드).
Basis: 6.218 - pass-2 pair second records (동 B, sealed, cold; 10 of the 60
evaluation days) are the worst day type (EC RMSE .488 vs .206); a flag feature
(ST5) did not help because the general model sees only 7 such pass-2 days among
~360.  Pass 1 has ~110 second records, predicted well (RMSE .157).
Specialist: ExtraTrees (core.et(seed), FS = FULL - day + season) fitted ONLY on
the fold's training rows whose record is a pair second (role from identical-weather
structure, C6.205), post-processed by p3.final with those rows as reference.
Applied only to validation rows with the causal flag is2_h = 1 (outdoor weather of
hours 0..h equals the previous record's): candidate = 0.5 * R3S + 0.5 * specialist;
all other rows = R3S.  Weight 0.5 fixed now.
Baseline R3S = stored DC5 / EL1 OOF.  Rule (user's): all 3 seeds x 5 validators
better and DIAG10 P(worse) < .025; EL1 better for all seeds also required.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_SB1_second_record_specialist_v1.py
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
CK = os.path.join(env.LOCAL, "sb1_ckpt")
W_SP = 0.5


def main():
    os.makedirs(CK, exist_ok=True)
    A = st8.load_inputs()
    A["is2"] = st8.causal_is_second(A).astype(float)
    roles = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"))
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.merge(A[["row_id", "is2"]], on="row_id", how="left").merge(roles, on=["farm", "day"], how="left")
    assert lab.is2.notna().all() and lab.role.notna().all()
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"]
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
        ts = tr[tr.role == "second"]
        frame = va[["row_id", "farm", "day", "hour", "sub_ec", "is2", "role"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        frame["n_spec_days"] = ts[["farm", "day"]].drop_duplicates().shape[0]
        for s in SEEDS:
            frame["sp_%d" % s] = p3.final(core.predict_model(core.et(s), ts, va, FS), ts, va)
        frame.to_csv(path, index=False)
        print("%s/%d done (specialist days %d)" % (name, i, frame.n_spec_days.iloc[0]), flush=True)
    P = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    base = pd.concat([d5, e1[["row_id", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]], ignore_index=True)
    O = P.merge(base, on=["row_id", "validator", "validation_fold"], how="inner")
    assert len(O) == len(P), (len(O), len(P))
    for s in SEEDS:
        O["sb_%d" % s] = np.where(O.is2 == 1, (1 - W_SP) * O["r3s_%d" % s] + W_SP * O["sp_%d" % s], O["r3s_%d" % s])
    O.to_csv(os.path.join(env.LOCAL, "ec3_SB1_all.csv"), index=False)
    print("rows flagged is2: %.3f; flagged rows that are true seconds %.3f" % ((O.is2 == 1).mean(), (O[O.is2 == 1].role == "second").mean()))
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261003)
    allb = True
    print("\npooled RMSE R3S -> R3S with second-record specialist (w .5 on flagged rows)")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["sb_%d" % s] - G.sub_ec)
            if v != "EL1":
                allb &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["sb_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    E = O[O.validator == "EL1"]
    elok = all(r(E["sb_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    L = D.day >= 179
    print("  DIAG10 late R3S %.4f -> SB1 %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["sb_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    for nm, G in (("DIAG10 pass-2 seconds", D[(D.day >= 179) & (D.role == "second")]), ("DIAG10 pass-1 seconds", D[(D.day < 179) & (D.role == "second")]),
                  ("EL1 seconds", E[E.role == "second"])):
        if len(G):
            print("  %s (rows %d): R3S %.4f -> SB1 %.4f | specialist alone %.4f" % (nm, len(G), np.mean([r(G["r3s_%d" % s] - G.sub_ec) for s in SEEDS]),
                  np.mean([r(G["sb_%d" % s] - G.sub_ec) for s in SEEDS]), np.mean([r(G["sp_%d" % s] - G.sub_ec) for s in SEEDS])))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nSB1 decision:", "PASS" if allb and all(p < 0.025 for p in ps) and elok else "FAIL",
          "(rule %s, EL1 %s)" % (allb and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
