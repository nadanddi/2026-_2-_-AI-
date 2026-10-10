# -*- coding: utf-8 -*-
"""TC5 (SCREEN, 2026-10-11 집 클로드, user: "전부 진행해봐" after round 12).  The round-11 chain corrector picks the same-house
previous date among the previous 1-3 records by midnight continuity only (hit ~.5).  The linker (6.559, rule-fixed v3) picks it
with date / calendar / continuity / exclusion cues (hit .64-.80).  TC5 swaps the predecessor used by the chain features.
Fixed before running.
  predecessor for record r = linker v3 CV choice at hour 0 (scratch-free copy: local/linker_v3/cv_all400_choice_v3.csv, record's
  10-record chunk hidden + both neighbours deleted).  Features exactly TC1's (lit1/lit2 unchanged, ch_in23 / ch_eve / ch_cost from
  the linker's predecessor; ch_cost = TC1 midnight cost to that predecessor).  MASK world: if the predecessor is an evaluation
  record its inputs are not used for training-row features (NaN), as TC1.
  Records without a linker choice keep TC1's predecessor.
Baseline: TC1c corrector (submitted recipe).  Candidate LNK: same recipe, linker predecessor (k = 1).
Layouts DIAG10, EL1, F..L (chain features use no labels -> no fold-hiding issue), EXT10/EXT12 reported.
PASS (screen) iff LNK beats TC1c in every cell of DIAG10, EL1, F..L AND pooled P(worse) < .025 (per-block differences averaged
over DIAG10 + F..L).  A pass needs a leaderboard hypothesis test (fresh layouts exhausted for this cost).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tc5_chain_linker_predecessor_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("tc1", os.path.join(HERE, "tc1_chain_history_early_hours_v1.py")); T = importlib.util.module_from_spec(s); s.loader.exec_module(T)
s = importlib.util.spec_from_file_location("tc1b", os.path.join(HERE, "tc1b_chain_history_common_train_v1.py")); B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
s = importlib.util.spec_from_file_location("tc2", os.path.join(HERE, "tc2_chain_hours_extend_v1.py")); C2 = importlib.util.module_from_spec(s); s.loader.exec_module(C2)
T.LAY["L"] = ("tt27_ckpt", "L_", ("base_R_1414", "base_R_1515"), "codex_R")
LK = os.path.join(env.LOCAL, "linker_v3", "cv_all400_choice_v3.csv")
rm = T.rm


def linker_table():
    """TC1 chain table with the predecessor replaced by the linker's choice (MASK world: train_X inputs only)."""
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    V = T.V; sd = X[V].std()
    h0 = X[X.hour == 0].set_index(["farm", "day"])[V]; h23 = X[X.hour == 23].set_index(["farm", "day"])[V]
    eve = X[X.hour >= 18].groupby(["farm", "day"]).in_temp.mean()
    L = pd.read_csv(LK); rows = []
    for rec, prev in zip(L.rec, L.prev):
        f, d, p = rec[:3], int(rec[4:7]), int(prev[4:7])
        if (f, p) in h23.index and (f, d) in h0.index:
            c = np.nanmean(np.abs((h0.loc[(f, d)] - h23.loc[(f, p)]) / sd).values)
            rows.append(dict(farm=f, day=d, ch_in23=h23.in_temp.get((f, p)), ch_eve=eve.get((f, p)), ch_cost=c))
        else:                                   # predecessor is an evaluation record -> MASK (NaN), as TC1
            rows.append(dict(farm=f, day=d, ch_in23=np.nan, ch_eve=np.nan, ch_cost=np.nan))
    return pd.DataFrame(rows)


def main():
    CT1 = T.chain_table(); LT = linker_table()
    CT5 = CT1.set_index(["farm", "day"]); LT = LT.set_index(["farm", "day"])
    CT5.loc[LT.index, ["ch_in23", "ch_eve", "ch_cost"]] = LT[["ch_in23", "ch_eve", "ch_cost"]].values
    CT5 = CT5.reset_index()
    agree = (CT1.set_index(["farm", "day"]).ch_in23.reindex(LT.index) == LT.ch_in23).mean()
    print("records with linker predecessor %d; same 23h in_temp as TC1 predecessor %.0f%%" % (len(LT), 100 * agree), flush=True)
    D1 = T.layout_frame("DIAG10", CT1); D5 = T.layout_frame("DIAG10", CT5)
    rng = np.random.default_rng(20261125); ok = True; diffs = []
    for L, (d, pre, bs, cx) in T.LAY.items():
        M1 = T.layout_frame(L, CT1); M5 = T.layout_frame(L, CT5); y = M1.sub_temp.values; cells = []
        for bi, b in enumerate(bs):
            for fam in "AB":
                p0 = B.w40(M1, b, cx, fam)
                c1 = C2.corrected(M1, p0, D1, bi, fam, 7); c5 = C2.corrected(M5, p0, D5, bi, fam, 7)
                cells.append(rm(c5 - y) / rm(c1 - y) - 1)
                if bi == 0 and fam == "A" and L not in ("EL1", "EXT10", "EXT12"):
                    key = (M1.farm + "_" + (M1.day // 5).astype(str)).values
                    diffs.append(pd.Series((c5 - y) ** 2 - (c1 - y) ** 2).groupby(key).mean())
        print("  %-6s LNK vs TC1c %s" % (L, " ".join("%+.2f%%" % (100 * x) for x in cells)), flush=True)
        if L not in ("EXT10", "EXT12"):
            ok &= all(x < 0 for x in cells)
    S = pd.concat(diffs, axis=1).mean(axis=1).values; idx = rng.integers(0, len(S), (20000, len(S))); p = float((S[idx].mean(1) >= 0).mean())
    print("LNK vs TC1c: all cells %s, pooled P(worse) %.4f -> %s" % (ok, p, "PASS" if ok and p < .025 else "FAIL"))


if __name__ == "__main__":
    main()
