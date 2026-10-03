# -*- coding: utf-8 -*-
"""EC stage-3 SE3: carry the previous same-동 day RESIDUAL into R3S (fixed before
running; 2026-10-03 집 클로드).
Basis: SE2 (6.211) - along the true calendar (C6.205) the R3S day residual persists
within a 동 over 1-3 dates (rho F13 A .33 / B .31, F47 A .40 / B .24; mixing 동
gives ~0).  ST8 (6.209) carried previous LABELS and failed; residuals are untested.
Leak-free residuals: for every outer fold (DIAG10/A/B/EXT10/EXT12 + EL1), the outer
TRAINING days get inner cross-fitted R3 residuals (R3 seed 7 = DC5 recipe with the
outer fold's season index; 5 inner folds = 5-record chunks round robin, +-1 purge).
No validation label touches any residual used for that fold.
Correction for a validation row (farm f, record d, hour h):
  date index = record order with pair seconds sharing the date (C6.205), same pass.
  rA / rB = inner residual (day mean) of the most recent outer-training labelled
            day of 동 A / B within 1..3 dates before d's date (else 0); 동 of past
            days = pair role, singletons by the h=23 input classifier (ST8 rule).
  pB_h = current record's P(동 B) at hour h (ST8 causal rule: exact weather match to
         the previous record over hours 0..h -> 1, else per-hour classifier).
  candidate = max(R3S + clip(0.3 * (pB*rB + (1-pB)*rA), -.3, .3), 0)
  beta 0.3 fixed now (~ SE2 rho), not fitted.
Baseline R3S = stored DC5 / EL1 OOF.  Rule (user's): all 3 seeds x 5 validators
better and DIAG10 P(worse) < .025; EL1 better for all seeds also required.
Run (detached, checkpointed):  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ec3_SE3_same_dong_residual_carry_v1.py
"""
import env  # noqa: F401
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
spec8 = importlib.util.spec_from_file_location("st8", os.path.join(HERE, "ec3_ST8_dong_state_corrector_v1.py"))
st8 = importlib.util.module_from_spec(spec8); spec8.loader.exec_module(st8)
dc4, p3, core = dc5.dc4, dc5.p3, dc5.core
SEEDS = (7, 101, 2024)
CK = os.path.join(env.LOCAL, "se3_ckpt")
BETA, CLIP, MAXLAG = 0.3, 0.3, 3


def structure():
    """pB per row_id (causal) and per-record date index / dong (past days)."""
    A = st8.load_inputs()
    roles = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"))
    A = A.merge(roles, on=["farm", "day"], how="left")
    A["is2"] = st8.causal_is_second(A)
    g = A.groupby(["farm", "day"])
    EM = []
    for c in st8.C:
        A["em_" + c] = g[c].transform(lambda s: s.expanding().mean())
        EM.append("em_" + c)
    A["pB"] = np.nan
    for f in ("F13", "F47"):
        mf = A.farm == f
        mu, sd = A.loc[mf & (A.src == "tr"), EM].mean(), A.loc[mf & (A.src == "tr"), EM].std()
        Z = ((A.loc[mf, EM] - mu) / sd).fillna(0)
        for h in range(24):
            trm = mf & (A.src == "tr") & (A.hour == h) & A.role.isin(["first", "second"])
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(Z.loc[trm].values, (A.loc[trm, "role"] == "second").astype(int))
            hm = mf & (A.hour == h)
            A.loc[hm, "pB"] = clf.predict_proba(Z.loc[hm].values)[:, 1]
    A["pB"] = np.where(A.is2, 1.0, A.pB)
    last = A[A.hour == 23].set_index(["farm", "day"]).pB
    rec = roles.sort_values(["farm", "day"]).copy()
    rec["dong"] = [("B" if r == "second" else "A") if r in ("first", "second") else ("B" if last.get((f, d), .5) >= .5 else "A")
                   for f, d, r in zip(rec.farm, rec.day, rec.role)]
    rec["date"] = rec.groupby("farm").role.transform(lambda s: np.cumsum(s.values != "second") - 1)
    return A.set_index("row_id").pB, rec.set_index(["farm", "day"])


def inner_residuals(tr, s=7):
    days = tr[["farm", "day"]].drop_duplicates().sort_values(["farm", "day"]).reset_index(drop=True)
    days["chunk"] = days.groupby("farm").cumcount() // 5
    days["inner"] = days.chunk % 5
    key = dict(zip(zip(days.farm, days.day), days.inner))
    FS = [c for c in core.FULL if c != "day"] + ["season"]
    BS = [c for c in core.BASE if c != "day"] + ["season"]
    out = {}
    for k in range(5):
        vd = {fd for fd, i in key.items() if i == k}
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)}
        itr = tr[[(f, d) not in forb for f, d in zip(tr.farm, tr.day)]]
        iva = tr[[(f, d) in vd for f, d in zip(tr.farm, tr.day)]]
        p = dc5.r3(itr, iva, s, FS, BS)
        res = pd.Series(iva.sub_ec.values - p).groupby([iva.farm.values, iva.day.values]).mean()
        out.update(res.to_dict())
    return out


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    wv = dc4.weather_vectors(full)
    el = []
    for f in ("F13", "F47"):
        days = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(days), 5):
            el.append(("EL1", len(el), {(f, int(d)) for d in days[k:k + 5]}))
    for name, i, vd in list(fds) + el:
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr = lab[np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])].copy()
        tdays = tr[["farm", "day"]].drop_duplicates()
        season, _ = dc4.season_index(tdays, tdays.iloc[:1], wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
        res = inner_residuals(tr)
        pd.DataFrame([(f, d, r) for (f, d), r in res.items()], columns=["farm", "day", "res"]).to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    pB, rec = structure()
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "farm", "day", "hour", "sub_ec", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"}); e1["validator"] = "EL1"
    O = pd.concat([d5, e1[d5.columns]], ignore_index=True)
    O["pB"] = pB.reindex(O.row_id).values
    assert O.pB.notna().all()
    corr = np.zeros(len(O))
    for (v, k), G in O.groupby(["validator", "validation_fold"]):
        R = pd.read_csv(os.path.join(CK, "%s_%d.csv" % (v, k))).set_index(["farm", "day"]).res
        for (f, d), idx in G.groupby(["farm", "day"]).groups.items():
            dt, ps = rec.loc[(f, d), "date"], d >= 179
            rA = rB = 0.0
            gotA = gotB = False
            for (ff, q), r in R[R.index.get_level_values(0) == f].sort_index(ascending=False).items():
                if q >= d or (q >= 179) != ps:
                    continue
                lag = dt - rec.loc[(f, q), "date"]
                if lag < 1:
                    continue
                if lag > MAXLAG:
                    break
                if rec.loc[(f, q), "dong"] == "A" and not gotA:
                    rA, gotA = r, True
                if rec.loc[(f, q), "dong"] == "B" and not gotB:
                    rB, gotB = r, True
            p = O.loc[idx, "pB"].values
            corr[np.asarray(idx)] = np.clip(BETA * (p * rB + (1 - p) * rA), -CLIP, CLIP)
    O["corr"] = corr
    for s in SEEDS:
        O["se_%d" % s] = np.maximum(O["r3s_%d" % s] + O["corr"], 0)
    O.to_csv(os.path.join(env.LOCAL, "ec3_SE3_all.csv"), index=False)
    print("rows with nonzero correction %.2f, mean |corr| %.4f" % ((O["corr"] != 0).mean(), O["corr"].abs().mean()))
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261003)
    allb = True
    print("\npooled RMSE R3S -> R3S + same-dong previous residual carry (beta .3)")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["se_%d" % s] - G.sub_ec)
            if v != "EL1":
                allb &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["se_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    E = O[O.validator == "EL1"]
    elok = all(r(E["se_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    L = D.day >= 179
    print("  DIAG10 late R3S %.4f -> SE3 %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["se_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nSE3 decision:", "PASS" if allb and all(p < 0.025 for p in ps) and elok else "FAIL",
          "(rule %s, EL1 %s)" % (allb and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
