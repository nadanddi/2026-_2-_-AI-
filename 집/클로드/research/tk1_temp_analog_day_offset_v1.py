# -*- coding: utf-8 -*-
"""TK1 (2026-10-12 집 클로드; allowed source 'stored training reference'): temperature day offset borrowed from ANALOG training
days.  Not tried before for temperature (catalog grep: kNN analog only for EC, 6.101/6.303-6.310).
Why: 57-77% of temperature error is a day-level offset (house-specific state); if days with similar inputs (0..h) share the
offset sign, the stored training reference can carry it.
Rule (causal, per hour h): sig_h = running mean over hours 0..h of a feature set, standardised by training stats;
  neighbours = k nearest training records of OTHER folds (purge same farm +-1 day, other farm +-5) by distance of sig_h;
  off = mean over neighbours of their WHOLE-DAY mean residual (y - q, q = their own OOF), pred_h = q_h + s * off.
Grid (researcher sweep): k {5, 10, 20, 40} x s {.25, .5, 1} x feature set {env, act, all} -> 36.
Selection on F13 pass-2 DIAG10 (cell mean), judgement on F47 pass-2, and vice versa; PASS iff both directions improve the
held-out farm in all 4 cells of DIAG10 AND of EL1 and G (same rule applied), and record bootstrap P(worse) < .025 on DIAG10.
Caveat: neighbour residuals come from OOF models that saw the target fold (mild); descriptive screen first.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tk1_temp_analog_day_offset_v1.py
"""
import os, sys, itertools, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(HERE, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
SETS = {"env": ["in_temp", "out_temp", "in_hum", "out_hum", "out_rad", "in_co2"],
        "act": ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog"]}
SETS["all"] = SETS["env"] + SETS["act"]
GRID = list(itertools.product((5, 10, 20, 40), (.25, .5, 1.0), ("env", "act", "all")))


def run_layout(M, cells, X):
    M = M.reset_index(drop=True); M["rec"] = M.farm + "_" + M.day.map("{:03d}".format)
    J = M[["row_id", "rec", "farm", "day", "hour", "fold", "sub_temp"]].merge(X, on="row_id", how="left")
    recs = J.drop_duplicates("rec")[["rec", "farm", "day", "fold"]].reset_index(drop=True); ri = {r: i for i, r in enumerate(recs.rec)}
    n = len(recs); out = {g: [] for g in GRID}
    for sname, cols in SETS.items():
        V = np.full((n, 24, len(cols)), np.nan)
        for c_i, c in enumerate(cols):
            P = J.pivot_table(index="rec", columns="hour", values=c, dropna=False).reindex(index=recs.rec, columns=range(24))
            V[:, :, c_i] = P.values
        V = np.where(np.isnan(V), np.nanmean(V, axis=(0, 1))[None, None, :] if np.isfinite(np.nanmean(V, axis=(0, 1))).all() else 0.0, V); V = np.nan_to_num(V)
        C = np.cumsum(V, 1) / np.arange(1, 25)[None, :, None]
        mu = C.reshape(-1, len(cols)).mean(0); sd = C.reshape(-1, len(cols)).std(0) + 1e-9; C = (C - mu) / sd
        J.attrs[sname] = C
    for q in cells:
        res = pd.Series(J.sub_temp.values - q).groupby(J.rec.values).mean().reindex(recs.rec).values
        for sname in SETS:
            C = J.attrs[sname]; offs = {k: np.zeros((n, 24)) for k in (5, 10, 20, 40)}
            for i in range(n):
                f, d, fo = recs.farm[i], recs.day[i], recs.fold[i]
                ok = (recs.fold.values != fo) & ~(((recs.farm.values == f) & (np.abs(recs.day.values - d) <= 1)) | ((recs.farm.values != f) & (np.abs(recs.day.values - d) <= 5)))
                idx = np.where(ok)[0]
                D = np.sqrt(((C[idx] - C[i][None]) ** 2).sum(-1))  # (cand, 24)
                o = np.argsort(D, 0)
                for k in offs:
                    offs[k][i] = res[idx[o[:k]]].mean(0)
            for (k, sh, sn) in GRID:
                if sn != sname:
                    continue
                add = offs[k][[ri[r] for r in J.rec], J.hour.values]
                out[(k, sh, sn)].append(q + sh * add)
    return J, out


def main():
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
    R = A2.A1.frames(); rng = np.random.default_rng(20261212); res = {}
    for L in ("DIAG10", "EL1", "G"):
        M, cells = R[L]; J, out = run_layout(M, cells, X); res[L] = (J, cells, out); print("layout", L, "done", flush=True)
    J, cells, out = res["DIAG10"]; y = J.sub_temp.values; p2 = J.day.values >= 179
    rows = []
    for g in GRID:
        r = {f: 100 * np.mean([rm((a - y)[p2 & (J.farm.values == f)]) / rm((q - y)[p2 & (J.farm.values == f)]) - 1 for a, q in zip(out[g], cells)]) for f in ("F13", "F47")}
        rows.append(dict(k=g[0], s=g[1], set=g[2], **r))
    G = pd.DataFrame(rows); print(G.round(2).to_string())
    for fac in ("k", "s", "set"):
        print("  main effect %-3s:" % fac, " | ".join("%s: F13 %+.2f%% F47 %+.2f%%" % (v, gg.F13.mean(), gg.F47.mean()) for v, gg in G.groupby(fac)))
    ok = True; per_all = []
    for sel, held in (("F13", "F47"), ("F47", "F13")):
        b = G.sort_values(sel).iloc[0]; g = (int(b.k), b.s, b.set)
        print("select on %s: k %d s %.2f %s (%s %+.2f%%) -> held %s %+.2f%%" % (sel, *g, sel, b[sel], held, b[held]))
        for L in ("DIAG10", "EL1", "G"):
            JJ, cc, oo = res[L]; yy = JJ.sub_temp.values; m = (JJ.day.values >= 179) & (JJ.farm.values == held)
            cl = [rm((a - yy)[m]) / rm((q - yy)[m]) - 1 for a, q in zip(oo[g], cc)]; ok &= all(c < 0 for c in cl)
            print("    %-6s held %s cells %s" % (L, held, " ".join("%+.2f%%" % (100 * c) for c in cl)))
            if L == "DIAG10":
                per_all.append(pd.concat([pd.Series(((a - yy) ** 2 - (q - yy) ** 2)[m]).groupby(JJ.rec.values[m]).sum() for a, q in zip(oo[g], cc)], axis=1).mean(1))
    P = pd.concat(per_all); P = P[P.abs() > 1e-12]; idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean()); ok &= pw < .025
    print("pooled record bootstrap P(worse) %.4f (%d records, improved %d) -> VERDICT TK1: %s" % (pw, len(P), int((P < 0).sum()), "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
