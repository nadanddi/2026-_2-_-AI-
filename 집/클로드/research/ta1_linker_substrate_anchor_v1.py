# -*- coding: utf-8 -*-
"""TA1 (SCREEN, 2026-10-11 집 클로드, user: "전부 해봐 ... 막힌 곳 해결").  Temperature anchor from the linker (6.559):
for a record whose same-house previous date is found with confidence (1 step, prev labelled, p >= .8; linker run with the
record's 10-record chunk hidden and the records on both sides deleted, scratch linker/cv_all400_choice_v1.csv), the
previous date's 23 h SUBSTRATE temperature (training label) anchors the early hours:
    pred_h = q_h + w * (1 - h / 8) * (A - q_0)   for h = 0..7     (q = round-11 recipe: W40G-S + chain corrector)
w is chosen on PASS-1 rows of DIAG10 only (grid .1 .. 1.0, seed-mean over the 4 cells) and then fixed; scoring on pass-2 rows
(temperature pass-2 protocol) and all rows.  Fixed before running.  Rough days excluded.  k = 1.
Layouts DIAG10, EL1, F..L (stored members); EXT10/EXT12 reported.
PASS (screen) iff better than q in every cell of DIAG10, EL1, F..L on pass-2 rows AND pooled pass-2 (per-block differences
averaged over DIAG10 + F..L) P(worse) < .025; all-rows reported.  A pass needs a fresh check (new layout / LB test).
Caveats: linker CV choices exist for 396 labelled records; anchors fire only where the CV choice is confident.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u ta1_linker_substrate_anchor_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("tc1", os.path.join(HERE, "tc1_chain_history_early_hours_v1.py")); T = importlib.util.module_from_spec(s); s.loader.exec_module(T)
s = importlib.util.spec_from_file_location("tc1b", os.path.join(HERE, "tc1b_chain_history_common_train_v1.py")); B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
s = importlib.util.spec_from_file_location("tc2", os.path.join(HERE, "tc2_chain_hours_extend_v1.py")); C2 = importlib.util.module_from_spec(s); s.loader.exec_module(C2)
T.LAY["L"] = ("tt27_ckpt", "L_", ("base_R_1414", "base_R_1515"), "codex_R")
LK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\96fe1d15-2691-4cbd-a117-4a3e2d96bdf9\scratchpad\linker\cv_all400_choice_v1.csv"
rm = T.rm


def anchors():
    A = pd.read_csv(LK)
    A = A[(A.prev_is_labelled.astype(bool)) & (A.p >= .8) & (A.steps_to_anchor == 1) & A.anchor_st23.notna()]
    return {(r[:3], int(r[4:7])): float(a) for r, a in zip(A.rec, A.anchor_st23)}


def apply(M, q, AN, w):
    out = q.copy(); M = M.assign(q=q)
    for (f, d), idx in M.groupby(["farm", "day"]).groups.items():
        a = AN.get((f, d))
        if a is None:
            continue
        rows = M.loc[idx]; q0 = rows.loc[rows.hour == 0, "q"]
        if not len(q0):
            continue
        q0 = float(q0.iloc[0]); h = rows.hour.values; m = h <= 7
        j = np.array([M.index.get_loc(i) for i in rows.index])
        out[j[m]] = q[j[m]] + w * (1 - h[m] / 8) * (a - q0)
    return out


def frames():
    CT = T.chain_table(); D = T.layout_frame("DIAG10", CT); res = {}
    for L, (d, pre, bs, cx) in T.LAY.items():
        M = T.layout_frame(L, CT); cells = []
        for bi, b in enumerate(bs):
            for fam in "AB":
                p0 = B.w40(M, b, cx, fam); cells.append(C2.corrected(M, p0, D, bi, fam, 7))
        res[L] = (M, cells)
    return res


def main():
    AN = anchors(); print("confident 1-step anchors on labelled records: %d" % len(AN), flush=True)
    R = frames(); rng = np.random.default_rng(20261123)
    M, cells = R["DIAG10"]; p1 = (M.day.values < 179); y = M.sub_temp.values
    best = None
    for w in np.round(np.arange(.1, 1.01, .1), 2):
        sc = np.mean([rm((apply(M, q, AN, w) - y)[p1]) for q in cells])
        best = (w, sc) if best is None or sc < best[1] else best
        print("  pass-1 DIAG10 w %.1f RMSE %.4f" % (w, sc))
    w = best[0]; print("chosen w = %.1f (pass-1 only)" % w, flush=True)
    ok = True; diffs = []
    for L, (M, cells) in R.items():
        y = M.sub_temp.values; m2 = M.day.values >= 179; fired = np.array([(f, d) in AN for f, d in zip(M.farm, M.day)])
        c2, ca = [], []
        for k, q in enumerate(cells):
            a = apply(M, q, AN, w)
            c2.append(rm((a - y)[m2]) / rm((q - y)[m2]) - 1); ca.append(rm(a - y) / rm(q - y) - 1)
            if k == 0 and L not in ("EL1", "EXT10", "EXT12"):
                key = (M.farm + "_" + (M.day // 5).astype(str)).values[m2]
                diffs.append(pd.Series(((a - y) ** 2 - (q - y) ** 2)[m2]).groupby(key).mean())
        e = fired & (M.hour.values <= 7)
        print("  %-6s pass-2 %s | all %s | anchored early rows %d (pass-2 %d)" % (L, " ".join("%+.2f%%" % (100 * x) for x in c2),
              " ".join("%+.2f%%" % (100 * x) for x in ca), e.sum(), (e & m2).sum()), flush=True)
        if L not in ("EXT10", "EXT12"):
            ok &= all(x < 0 for x in c2)
    S = pd.concat(diffs, axis=1).mean(axis=1).dropna().values; idx = rng.integers(0, len(S), (20000, len(S))); p = float((S[idx].mean(1) >= 0).mean())
    print("TA1 (w %.1f) pass-2: all cells %s, pooled P(worse) %.4f -> %s" % (w, ok, p, "PASS" if ok and p < .025 else "FAIL"))


if __name__ == "__main__":
    main()
