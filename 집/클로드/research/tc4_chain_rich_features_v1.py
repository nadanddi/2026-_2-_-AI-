# -*- coding: utf-8 -*-
"""TC4 (SCREEN, 2026-10-11 집 클로드, user: "온도 다른 거 뭐 더 있나 찾아봐").  The chain-aware early-hour corrector (round 11,
.5127 -> .5089) uses only the chosen predecessor's evening AIR temperature.  TC4 adds more of the same house's previous evening
(all inputs of earlier records; causal).  Fixed before running.
Added features (chosen predecessor = TC1 rule): ch_hum23 (in_hum 23 h), ch_heat_eve / ch_therm_eve (act_heating / act_thermal
mean 18-23 h), ch_cool (in_temp 18 h - 23 h, evening cooling), ch_co2_23; second step: ch2_in23 (the predecessor's own chain
predecessor's 23 h in_temp, same rule applied to the predecessor) and ch2_cost.
Baseline: TC1c corrector (hours 0-7, recipe as submitted).  Candidate RICH: same recipe + added features (k = 1).
Layouts DIAG10, EL1, F..L (L = tt27, screen), EXT10/EXT12 reported.
PASS (screen) iff RICH beats the TC1c corrector in every cell of DIAG10, EL1, F..L AND pooled (per-block differences averaged
over DIAG10 + F..L) P(worse) < .025.  A pass needs a fresh layout with refitted members.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tc4_chain_rich_features_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("tc1", os.path.join(HERE, "tc1_chain_history_early_hours_v1.py")); T = importlib.util.module_from_spec(s); s.loader.exec_module(T)
s = importlib.util.spec_from_file_location("tc1b", os.path.join(HERE, "tc1b_chain_history_common_train_v1.py")); B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
s = importlib.util.spec_from_file_location("tc2", os.path.join(HERE, "tc2_chain_hours_extend_v1.py")); C2 = importlib.util.module_from_spec(s); s.loader.exec_module(C2)
T.LAY["L"] = ("tt27_ckpt", "L_", ("base_R_1414", "base_R_1515"), "codex_R")
ADD = ["ch_hum23", "ch_heat_eve", "ch_therm_eve", "ch_cool", "ch_co2_23", "ch2_in23", "ch2_cost"]
rm = T.rm


def rich_table():
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    V = T.V; sd = X[V].std()
    h0 = X[X.hour == 0].set_index(["farm", "day"])[V]; h23 = X[X.hour == 23].set_index(["farm", "day"])
    h18 = X[X.hour == 18].set_index(["farm", "day"]).in_temp; eve = X[X.hour >= 18].groupby(["farm", "day"])[["act_heating", "act_thermal"]].mean()
    allx = pd.concat([pd.read_csv(os.path.join(env.DATA, f), usecols=["row_id"]) for f in ("train_X.csv", "test_X.csv")])
    allx = allx[allx.row_id.str[:3].isin(["F13", "F47"])]; rec = sorted({(r[:3], int(r[4:7])) for r in allx.row_id})
    pred = {}; cost = {}
    for f in ("F13", "F47"):
        days = [d for ff, d in rec if ff == f]
        for i, d in enumerate(days):
            best, bc = None, np.inf
            if (f, d) in h0.index:
                for p in days[max(0, i - 3):i]:
                    if (f, p) in h23.index:
                        c = np.nanmean(np.abs((h0.loc[(f, d)] - h23.loc[(f, p), V]) / sd).values)
                        if np.isfinite(c) and c < bc:
                            best, bc = p, c
            if best is not None:
                pred[(f, d)] = best; cost[(f, d)] = bc
    rows = []
    for (f, d) in rec:
        p = pred.get((f, d)); r = dict(farm=f, day=d)
        if p is not None:
            r.update(ch_hum23=h23.in_hum.get((f, p), np.nan), ch_co2_23=h23.in_co2.get((f, p), np.nan),
                     ch_heat_eve=eve.act_heating.get((f, p), np.nan), ch_therm_eve=eve.act_thermal.get((f, p), np.nan),
                     ch_cool=h18.get((f, p), np.nan) - h23.in_temp.get((f, p), np.nan))
            q = pred.get((f, p))
            r.update(ch2_in23=h23.in_temp.get((f, q), np.nan) if q is not None else np.nan, ch2_cost=cost.get((f, p), np.nan))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    CT = T.chain_table().merge(rich_table(), on=["farm", "day"], how="left")
    D = T.layout_frame("DIAG10", CT)
    rng = np.random.default_rng(20261122); ok = True; diffs = []
    base_feat = list(T.FEAT)
    for L, (d, pre, bs, cx) in T.LAY.items():
        M = T.layout_frame(L, CT); y = M.sub_temp.values; cells = []
        for bi, b in enumerate(bs):
            for fam in "AB":
                p0 = B.w40(M, b, cx, fam)
                T.FEAT = base_feat; c7 = C2.corrected(M, p0, D, bi, fam, 7)
                T.FEAT = base_feat + ADD; cr = C2.corrected(M, p0, D, bi, fam, 7); T.FEAT = base_feat
                cells.append(rm(cr - y) / rm(c7 - y) - 1)
                if bi == 0 and fam == "A" and L not in ("EL1", "EXT10", "EXT12"):
                    key = (M.farm + "_" + (M.day // 5).astype(str)).values
                    diffs.append(pd.Series((cr - y) ** 2 - (c7 - y) ** 2).groupby(key).mean())
        print("  %-6s RICH vs TC1c %s" % (L, " ".join("%+.2f%%" % (100 * x) for x in cells)), flush=True)
        if L not in ("EXT10", "EXT12"):
            ok &= all(x < 0 for x in cells)
    S = pd.concat(diffs, axis=1).mean(axis=1).values; idx = rng.integers(0, len(S), (20000, len(S))); p = float((S[idx].mean(1) >= 0).mean())
    print("RICH vs TC1c: all cells %s, pooled P(worse) %.4f -> %s" % (ok, p, "PASS" if ok and p < .025 else "FAIL"))


if __name__ == "__main__":
    main()
