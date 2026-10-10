# -*- coding: utf-8 -*-
"""TA7 (2026-10-12 집 클로드; researcher sweep, memory researcher-sweeps): temperature anchor with MEASURED components (TM1)
swept over reasoned ranges.  Fixed before running:
  grid   p_min in {.8, .9, .95}            (TM1: only p > .95 clearly better than the model on pass 1)
         bias  in {const, dT}              (const: yhat0 = A - .36 [submitted];  dT: yhat0 = A + c0 + k dT_air, c0/k from TM1 pass 1)
         decay in {lin8, meas}             (lin8: w(h) = 1 - h/8 on h < 8 [submitted]; meas: w(h) = beta_h measured on pass 1, 0..23)
         clip  in {1, 2, inf}              (TM1: response ~linear up to |gap| 2)
         gate  in {1.5, 3, inf}            (|dT_air| gate; with bias=dT the step is modelled, so a wider gate is reasonable)
  -> 108 settings.  Correction: pred_h = q_h + w(h) * clip(yhat0 - q_0, +-clip), q = round-11 recipe (W40G-S + chain corrector).
SELECTION on PASS-1 rows of the clean layout DIAG10 only (seed-mean RMSE).  JUDGEMENT of the selected setting on pass-2 rows:
  all 4 cells better than the SUBMITTED rule (p .8, const, lin8, clip 1, gate 1.5) AND record-level bootstrap P(worse) < .025.
Whole pass-2 response table is reported (main-effect means per factor) so the choice can be read against the curve.
Anchors: linker v7 mirror-CV choices (local/linker_v7/cv_all400_choice_v7m.csv).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ta7_anchor_sweep_v1.py
"""
import os, sys, itertools, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(HERE, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
LK = os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")
PRM = np.load(os.path.join(env.LOCAL, "linker_v7", "tm1_params.npy")); C0, K = PRM[0], PRM[1]; BETA = PRM[2:26]
rm = A2.rm


def build(M, L):
    """per-row arrays: record index, anchor A, dT, p, and record-level q0 lookup."""
    key = (M.farm + "_" + M.day.map("{:03d}".format)).values
    A = L.set_index("rec")
    return key, A


def correct(M, q, key, A, p_min, bias, decay, clip, gate):
    out = q.copy(); h = M.hour.values
    sel = A[(A.p >= p_min) & (A.dT_air.abs() <= gate)]
    q0 = pd.Series(q[h == 0], index=key[h == 0])
    for rec in sel.index.intersection(q0.index):
        a, dT = sel.at[rec, "anchor_st23"], sel.at[rec, "dT_air"]
        yh = a - .36 if bias == "const" else a + C0 + K * dT
        gap = np.clip(yh - q0[rec], -clip, clip)
        m = key == rec; hh = h[m]
        w = np.where(hh < 8, 1 - hh / 8, 0.0) if decay == "lin8" else BETA[hh]
        out[m] = q[m] + w * gap
    return out


def main():
    L = pd.read_csv(LK); L = L[L.prev_is_labelled.astype(bool) & L.anchor_st23.notna() & (L.steps_to_anchor == 1)].copy()
    R = A2.A1.frames(); rng = np.random.default_rng(20261202)
    M, cells = R["DIAG10"]; key, A = build(M, L); y = M.sub_temp.values; p1 = M.day.values < 179; p2 = ~p1
    grid = list(itertools.product((.8, .9, .95), ("const", "dT"), ("lin8", "meas"), (1, 2, np.inf), (1.5, 3, np.inf)))
    rows = []
    for g in grid:
        r1 = np.mean([rm((correct(M, q, key, A, *g) - y)[p1]) for q in cells]); r2 = np.mean([rm((correct(M, q, key, A, *g) - y)[p2]) for q in cells])
        rows.append(dict(p_min=g[0], bias=g[1], decay=g[2], clip=g[3], gate=g[4], pass1=r1, pass2=r2))
    T = pd.DataFrame(rows); base = T[(T.p_min == .8) & (T.bias == "const") & (T.decay == "lin8") & (T.clip == 1) & (T.gate == 1.5)].iloc[0]
    q_ref = np.mean([rm((q - y)[p2]) for q in cells]); q_ref1 = np.mean([rm((q - y)[p1]) for q in cells])
    T["d1"] = 100 * (T.pass1 / base.pass1 - 1); T["d2"] = 100 * (T.pass2 / base.pass2 - 1)
    print("no-anchor RMSE pass1 %.4f pass2 %.4f | submitted rule pass1 %.4f pass2 %.4f" % (q_ref1, q_ref, base.pass1, base.pass2))
    for fac in ("p_min", "bias", "decay", "clip", "gate"):
        print("  main effect %-6s:" % fac, " ".join("%s: pass1 %+.2f%% pass2 %+.2f%%" % (v, g.d1.mean(), g.d2.mean()) for v, g in T.groupby(fac)))
    best = T.sort_values("pass1").iloc[0]
    print("selected on pass-1:", best[["p_min", "bias", "decay", "clip", "gate"]].to_dict(), "| pass1 %+.2f%% pass2 %+.2f%% (vs submitted)" % (best.d1, best.d2))
    print("pass-2 rank of the selected setting: %d of %d; top-5 by pass-1 and their pass-2:" % (int((T.pass2 < best.pass2).sum()) + 1, len(T)))
    print(T.sort_values("pass1").head(5)[["p_min", "bias", "decay", "clip", "gate", "d1", "d2"]].round(2).to_string())
    T.to_csv(os.path.join(env.LOCAL, "linker_v7", "ta7_grid.csv"), index=False)
    g = (best.p_min, best.bias, best.decay, best.clip, best.gate); gb = (.8, "const", "lin8", 1, 1.5)
    for Lname in ("DIAG10", "G", "EL1", "EXT10", "EXT12"):
        M, cells = R[Lname]; key, A = build(M, L); y = M.sub_temp.values; m2 = M.day.values >= 179; rel = []; per = []
        for q in cells:
            a = correct(M, q, key, A, *g); b = correct(M, q, key, A, *gb)
            rel.append(rm((a - y)[m2]) / rm((b - y)[m2]) - 1)
            per.append(pd.Series(((a - y) ** 2 - (b - y) ** 2)[m2]).groupby(key[m2]).sum())
        P = pd.concat(per, axis=1).mean(axis=1); P = P[P.abs() > 1e-12]
        line = "  %-6s selected vs submitted, pass-2 cells %s | changed records %d, improved %d" % (Lname, " ".join("%+.2f%%" % (100 * x) for x in rel), len(P), int((P < 0).sum()))
        if Lname == "DIAG10" and len(P):
            idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean())
            line += " | record P(worse) %.4f -> %s" % (pw, "PASS" if all(x < 0 for x in rel) and pw < .025 else "FAIL")
        print(line, flush=True)


if __name__ == "__main__":
    main()
