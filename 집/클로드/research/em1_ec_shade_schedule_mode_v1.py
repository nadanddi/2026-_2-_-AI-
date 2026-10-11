# -*- coding: utf-8 -*-
"""EM1 (2026-10-12 집 클로드; sub-agent meta forensics lead): EC residual is positive on days whose shade screen runs in a
0/100 SCHEDULE (step changes only) rather than proportional control (partial rho +.23 after daily-mean controls; causal 0-9 h ->
10-23 h +.21; pass-2 F13 +.43, F47 +.64).  Possibly the known high-EC condition (6.194, 6.240), but never tried as an EC
residual rule.  Evaluation-set mode share is NOT used (eval composition statistic, banned).
Causal indicator per hour h: changes = nonzero |diff| of act_shade over hours 0..h of the record; S_h = 1 if at least one change
and min(changes) >= T, else 0.
Rules (researcher sweep): additive  p' = sg + b * S_h,      b {.02, .05, .1, .15, .2}
                         multiplic. p' = sg * (1 + m * S_h), m {.05, .1, .2, .3}
                         x T {30, 50, 90}  -> 27 settings.
Base: stored RV1 OOF (CUR + SG2, column sg), pass-2 rows.  Selection on F13 pass-2 (DIAG10 seed mean), judgement on F47 and vice
versa.  PASS iff both directions: held-out farm improves in every seed of DIAG10, P2LOO, EL1, AND pooled record bootstrap
P(worse) < .025 on DIAG10.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u em1_ec_shade_schedule_mode_v1.py
"""
import os, itertools
import env  # noqa: F401
import numpy as np, pandas as pd
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
GRID = [("add", b, T) for b in (.02, .05, .1, .15, .2) for T in (30, 50, 90)] + [("mul", m, T) for m in (.05, .1, .2, .3) for T in (30, 50, 90)]


def main():
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "act_shade"]); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int); X = X.sort_values(["rec", "h"])
    X["d"] = X.groupby("rec").act_shade.diff().abs()
    X["dm"] = X.d.where(X.d > 1e-9)
    X["minstep"] = X.groupby("rec").dm.transform(lambda s: s.expanding().min())
    O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.day >= 179)].copy()
    O["rec"] = O.row_id.str[:7]; O = O.merge(X[["row_id", "minstep"]], on="row_id", how="left")
    print("pass-2 rows %d, share with any shade change so far %.2f" % (len(O[O.validator == "DIAG10"]), O[O.validator == "DIAG10"].minstep.notna().mean()))

    def apply(D, g):
        kind, v, T = g; S = (D.minstep.fillna(-1) >= T).astype(float).values
        return (D.sg + v * S).values if kind == "add" else (D.sg * (1 + v * S)).values
    D = O[O.validator == "DIAG10"]; rows = []
    for g in GRID:
        pr = apply(D, g); r = {}
        for f in ("F13", "F47"):
            m = (D.farm == f).values; r[f] = 100 * np.mean([rm((pr - D.sub_ec)[m & (D.seed == s).values]) / rm((D.sg - D.sub_ec)[m & (D.seed == s).values]) - 1 for s in D.seed.unique()])
        rows.append(dict(kind=g[0], v=g[1], T=g[2], **r))
    G = pd.DataFrame(rows); print(G.round(2).to_string())
    rng = np.random.default_rng(20261214); ok = True; per = []
    for sel, held in (("F13", "F47"), ("F47", "F13")):
        b = G.sort_values(sel).iloc[0]; g = (b.kind, b.v, b.T)
        print("select on %s: %s v %.2f T %d (%s %+.2f%%) -> held %s %+.2f%%" % (sel, g[0], g[1], g[2], sel, b[sel], held, b[held]))
        for v in ("DIAG10", "P2LOO", "EL1"):
            XX = O[(O.validator == v) & (O.farm == held)]; pr = apply(XX, g)
            cl = [rm((pr - XX.sub_ec)[(XX.seed == s).values]) / rm((XX.sg - XX.sub_ec)[(XX.seed == s).values]) - 1 for s in XX.seed.unique()]
            ok &= all(c < 0 for c in cl); print("    %-6s %s cells %s" % (v, held, " ".join("%+.2f%%" % (100 * c) for c in cl)))
            if v == "DIAG10":
                per.append(pd.Series((pr - XX.sub_ec.values) ** 2 - (XX.sg - XX.sub_ec).values ** 2).groupby([XX.seed.values, XX.rec.values]).sum().groupby(level=1).mean())
    P = pd.concat(per); P = P[P.abs() > 1e-12]; idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean()); ok &= pw < .025
    print("pooled record bootstrap P(worse) %.4f (%d records, improved %d) -> VERDICT EM1: %s" % (pw, len(P), int((P < 0).sum()), "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
