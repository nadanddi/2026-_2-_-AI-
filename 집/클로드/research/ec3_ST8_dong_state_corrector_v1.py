# -*- coding: utf-8 -*-
"""EC stage-3 ST8: same-동 previous-label day-level corrector on R3S, with the
TRUE 동 (pair order) instead of record parity (fixed before running; 2026-10-03
집 클로드).
Basis: C6.205 records are calendar ordered, a date has 1 or 2 records (identical
24h weather neighbours), so 'parity = 동' is wrong in pass 1; 6.208: with the
objective pair 동, EC day-mean persistence is .89-.94 (other 동 .42-.45), high EC
almost only in 동 B.  PAR1 (6.166) / ESTATE_PAR (6.179) grouped by parity.
To avoid tree identifier over-fitting (PAR1), a LOW-variance linear corrector:
  pB_h  current record's P(동 B) at hour h: 1 if the record's outdoor weather of
        hours 0..h equals the previous record's (= second of a pair, causal);
        else a per-hour logistic classifier (C=1) on expanding means (0..h) of 10
        indoor/actuator channels, standardized per farm, trained on train_X pair
        records (inputs + pair role only, no labels, no test_X).
  dong of PREVIOUS records: pair role (first A / second B), singletons: classifier
        at h=23 (their full day is past input).
  lastA / lastB: EC label day mean of the most recent AVAILABLE labelled record of
        that 동 within 12 record days before (available = not in the validation
        fold +-1 and not in the lock +-1, same as model training).
  x1 = pB*(lastB - cm), x2 = (1-pB)*(lastA - cm), missing -> 0; cm = expanding
        mean of R3S (seed s) over hours 0..h.  Ridge(alpha 1, intercept) on
        target y - R3S; correction clipped to +-0.5; candidate = max(R3S + corr, 0).
  Corrector fit: DIAG10 OOF rows (features w.r.t. their own DIAG10 fold) whose
        days are NOT in the evaluated validator fold's days +-1; applied to that
        fold's rows (features w.r.t. that fold).
Baseline R3S = stored DC5 / EL1 OOF.  Rule (user's): all 3 seeds x 5 validators
better and DIAG10 P(worse) < .025; EL1 better for all seeds also required.
Structure (roles) uses train_X+test_X weather equality only (C6.205); no labels of
evaluation rows, no test_X statistics in any fitted object.
"""
import env  # noqa: F401
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc5)
p3 = dc5.p3
SEEDS = (7, 101, 2024)
C = ["in_temp", "in_hum", "in_co2", "act_heating", "act_vent", "act_thermal", "act_shade", "act_circfan", "act_fog", "act_co2"]
W4 = ["out_temp", "out_hum", "out_rad", "out_wspd"]


def load_inputs():
    tr = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + W4 + C).assign(src="tr")
    te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"] + W4).assign(src="te")
    A = pd.concat([tr, te])
    A = A[A.row_id.str[:3].isin(["F13", "F47"])].copy()
    A["farm"], A["day"], A["hour"] = A.row_id.str[:3], A.row_id.str[4:7].astype(int), A.row_id.str[8:10].astype(int)
    return A.sort_values(["farm", "day", "hour"]).reset_index(drop=True)


def causal_is_second(A):
    """row-level: weather of hours 0..h equals the previous record's hours 0..h."""
    out = np.zeros(len(A), bool)
    for f, G in A.groupby("farm"):
        W = {d: g.set_index("hour")[W4] for d, g in G.groupby("day")}
        for d, g in G.groupby("day"):
            if d - 1 not in W:
                continue
            prev = W[d - 1]
            eq = True
            for idx, row in g.iterrows():
                h = row.hour
                if h in prev.index:
                    a, b = row[W4].values.astype(float), prev.loc[h].values.astype(float)
                    ok = ~np.isnan(a) & ~np.isnan(b)
                    eq = eq and ok.any() and np.max(np.abs(a[ok] - b[ok])) < 1e-9
                else:
                    eq = False
                out[idx] = eq
    return out


def main():
    A = load_inputs()
    roles = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"))
    A = A.merge(roles, on=["farm", "day"], how="left")
    A["is2"] = causal_is_second(A)
    g = A.groupby(["farm", "day"])
    EM = []
    for c in C:
        A["em_" + c] = g[c].transform(lambda s: s.expanding().mean())
        EM.append("em_" + c)
    # per farm x hour classifier on train_X pair records
    A["pB"] = np.nan
    for f in ("F13", "F47"):
        mf = (A.farm == f)
        mu, sd = A.loc[mf & (A.src == "tr"), EM].mean(), A.loc[mf & (A.src == "tr"), EM].std()
        Z = ((A.loc[mf, EM] - mu) / sd).fillna(0)
        for h in range(24):
            trm = mf & (A.src == "tr") & (A.hour == h) & A.role.isin(["first", "second"])
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(Z.loc[trm].values, (A.loc[trm, "role"] == "second").astype(int))
            hm = mf & (A.hour == h)
            A.loc[hm, "pB"] = clf.predict_proba(Z.loc[hm].values)[:, 1]
    A["pB"] = np.where(A.is2, 1.0, A.pB)
    # dong of every record (for previous records): pair role, else classifier at h=23
    last = A[A.hour == 23].set_index(["farm", "day"]).pB
    rec = roles.copy()
    rec["dong"] = [("B" if r == "second" else "A") if r in ("first", "second") else ("B" if last.get((f, d), .5) >= .5 else "A")
                   for f, d, r in zip(rec.farm, rec.day, rec.role)]
    dong = rec.set_index(["farm", "day"]).dong
    Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
    Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
    Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
    lock = {(z["farm"], int(z["day"])) for z in json.loads(open(p3.LOCK, encoding="utf-8").read())["selected"]}
    ECd = Y.groupby(["farm", "day"]).sub_ec.mean()
    ECd = ECd[[k not in lock for k in ECd.index]]
    # OOF frame
    d5 = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))[["row_id", "farm", "day", "hour", "sub_ec", "validator", "validation_fold"] + ["r3s_%d" % s for s in SEEDS]]
    e1 = pd.read_csv(os.path.join(env.LOCAL, "ec2_EL1_oof.csv")).rename(columns={"fold": "validation_fold"})
    e1["validator"] = "EL1"
    O = pd.concat([d5, e1[d5.columns]], ignore_index=True)
    O = O.merge(A[["row_id", "pB"]], on="row_id", how="left")
    assert O.pB.notna().all()
    O = O.sort_values(["validator", "validation_fold", "farm", "day", "hour"]).reset_index(drop=True)
    forb = {}
    for (v, k), G in O.groupby(["validator", "validation_fold"]):
        vd = set(zip(G.farm, G.day))
        forb[(v, k)] = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
    lastA, lastB = np.full(len(O), np.nan), np.full(len(O), np.nan)
    for (v, k, f, d), idx in O.groupby(["validator", "validation_fold", "farm", "day"]).groups.items():
        fb = forb[(v, k)]
        for q in range(d - 1, d - 13, -1):
            if (f, q) in ECd.index and (f, q) not in fb and (f, q) in dong.index:
                if dong[(f, q)] == "A" and np.isnan(lastA[idx[0]]):
                    lastA[idx] = ECd[(f, q)]
                if dong[(f, q)] == "B" and np.isnan(lastB[idx[0]]):
                    lastB[idx] = ECd[(f, q)]
    O["lastA"], O["lastB"] = lastA, lastB
    grp = O.groupby(["validator", "validation_fold", "farm", "day"])
    for s in SEEDS:
        cm = grp["r3s_%d" % s].transform(lambda x: x.expanding().mean())
        O["x1_%d" % s] = np.where(O.lastB.notna(), O.pB * (O.lastB - cm), 0.0)
        O["x2_%d" % s] = np.where(O.lastA.notna(), (1 - O.pB) * (O.lastA - cm), 0.0)
    D = O[O.validator == "DIAG10"]
    for s in SEEDS:
        corr = np.zeros(len(O))
        for (v, k), G in O.groupby(["validator", "validation_fold"]):
            vd = {(f, d + j) for f, d in set(zip(G.farm, G.day)) for j in (-1, 0, 1)}
            T = D[[(f, d) not in vd for f, d in zip(D.farm, D.day)]]
            m = Ridge(alpha=1.0).fit(T[["x1_%d" % s, "x2_%d" % s]].values, (T.sub_ec - T["r3s_%d" % s]).values)
            corr[G.index] = np.clip(m.predict(G[["x1_%d" % s, "x2_%d" % s]].values), -.5, .5)
            if v == "DIAG10" and k == 0:
                print("seed %d DIAG10 fold0 corrector coef %s intercept %.4f" % (s, np.round(m.coef_, 3), m.intercept_))
        O["st_%d" % s] = np.maximum(O["r3s_%d" % s] + corr, 0)
    O.to_csv(os.path.join(env.LOCAL, "ec3_ST8_all.csv"), index=False)
    print("coverage: lastA %.2f lastB %.2f; pB from exact pair %.3f" % (O.lastA.notna().mean(), O.lastB.notna().mean(), (O.pB == 1).mean()))
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    rng = np.random.default_rng(20261003)
    allb = True
    print("\npooled RMSE R3S -> R3S + same-dong previous-label corrector")
    for v in ("DIAG10", "A", "B", "EXT10", "EXT12", "EL1"):
        G = O[O.validator == v]
        cells = []
        for s in SEEDS:
            a, b = r(G["r3s_%d" % s] - G.sub_ec), r(G["st_%d" % s] - G.sub_ec)
            if v != "EL1":
                allb &= b < a
            cells.append("s%d %.4f->%.4f (%+.1f%%)" % (s, a, b, 100 * (b / a - 1)))
        print("  %-6s %s" % (v, "  ".join(cells)))
    D = O[O.validator == "DIAG10"].copy(); D["cl"] = D.farm + "_" + (D.day // 5).astype(str)
    ps = []
    for s in SEEDS:
        dd = (D["st_%d" % s] - D.sub_ec) ** 2 - (D["r3s_%d" % s] - D.sub_ec) ** 2
        cl = dd.groupby(D.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
        idx = rng.integers(0, len(sm), (20000, len(sm)))
        ps.append(float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean()))
    E = O[O.validator == "EL1"]
    elok = all(r(E["st_%d" % s] - E.sub_ec) < r(E["r3s_%d" % s] - E.sub_ec) for s in SEEDS)
    L = D.day >= 179
    hi = D.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1
    print("  DIAG10 late R3S %.4f -> ST8 %.4f" % (np.mean([r((D["r3s_%d" % s] - D.sub_ec)[L]) for s in SEEDS]), np.mean([r((D["st_%d" % s] - D.sub_ec)[L]) for s in SEEDS])))
    print("  DIAG10 high-EC days (rows %d): label %.3f  R3S %.3f  ST8 %.3f" % (hi.sum(), D.sub_ec[hi].mean(), D.r3s_7[hi].mean(), D.st_7[hi].mean()))
    print("  DIAG10 P(worse) by seed:", [round(p, 4) for p in ps])
    print("\nST8 decision:", "PASS" if allb and all(p < 0.025 for p in ps) and elok else "FAIL",
          "(rule %s, EL1 %s)" % (allb and all(p < 0.025 for p in ps), elok))


if __name__ == "__main__":
    main()
