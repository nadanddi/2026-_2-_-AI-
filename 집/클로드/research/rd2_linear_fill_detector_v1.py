# -*- coding: utf-8 -*-
"""RD2: straight-line (linear-fill) detector for restored stretches in the
F13/F47 TRAIN inputs.  2026-10-02 집 클로드.

Why: in the 48 integer farms, 43% of long fractional in_temp runs are exact
straight lines between the neighbouring hours (rd1_restore_signature_v1) -
the organisers' restoration includes linear filling.  F13/F47 train inputs
are said to be restored/transformed (problem statement 4); test inputs are
not.  So test inputs give the natural rate of straight stretches.

Detector (fixed before running)
  channel x in {in_temp (0.1 res, tol 0.1), in_hum (1, tol 1), in_co2 (1, tol 1)}
  a "straight stretch" = maximal run of consecutive hours (contiguous time,
  same farm) whose 2nd difference |x[t+1] - 2x[t] + x[t-1]| <= tol + 1e-9;
  stretch length = qualifying 2nd differences + 2 points.  Constant runs
  (slope 0) count as straight too and are reported separately.
  Length threshold K per channel = the smallest K in 4..12 whose TEST row-flag
  rate is <= 0.5% (chosen on test only, before looking at train rates).
Reports
  1. per channel and K: row-flag rate train vs test (excess = restoration)
  2. overlap with the old physics-rule flags (eda_forensic_11 Vany, 4.7)
  3. G_C2 DIAG10 OOF temperature error on flagged rows vs others (and +-3 h)
  4. EC v2 DIAG10 OOF error on flagged rows vs others (Codex phase-3 OOF)
Decision (for a later pre-registered weighting test): worth testing only if
  train flag rate >= 2x test rate in at least one channel AND G_C2 RMSE on
  flagged rows >= 1.3x the unflagged rows.

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u rd2_linear_fill_detector_v1.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd

from harness import load

CH = {"in_temp": 0.1, "in_hum": 1.0, "in_co2": 1.0}
KS = range(4, 13)


def straight_runs(g, c, tol):
    """per-row: length of the straight stretch the row belongs to (0 if none), and is-constant."""
    v = g[c].values
    t = g.t.values
    n = len(v)
    ok = np.zeros(n, bool)       # ok[i]: 2nd difference centred at i qualifies
    for i in range(1, n - 1):
        if t[i + 1] - t[i] == 1 and t[i] - t[i - 1] == 1 and not np.isnan(v[i - 1:i + 2]).any():
            ok[i] = abs(v[i + 1] - 2 * v[i] + v[i - 1]) <= tol + 1e-9
    L = np.zeros(n, int)
    const = np.zeros(n, bool)
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        a, b = i - 1, j + 1
        L[a:b + 1] = np.maximum(L[a:b + 1], b - a + 1)
        if np.all(np.diff(v[a:b + 1]) == 0):
            const[a:b + 1] = True
        i = j + 1
    return L, const


def main():
    panel, _, _ = load()
    P = panel[panel.farm.isin(["F13", "F47"])].sort_values(["farm", "day", "hour"]).reset_index(drop=True)
    P["t"] = P.day * 24 + P.hour
    for c, tol in CH.items():
        Ls, Cs = [], []
        for _, g in P.groupby("farm"):
            L, C = straight_runs(g, c, tol)
            Ls.append(L)
            Cs.append(C)
        P["L_" + c] = np.concatenate(Ls)
        P["C_" + c] = np.concatenate(Cs)
    tr, te = P[~P.is_test], P[P.is_test]
    print("rows train %d test %d" % (len(tr), len(te)))

    print("\n1. row-flag rate (stretch length >= K): train / test")
    Kc = {}
    for c in CH:
        line = []
        for K in KS:
            a, b = (tr["L_" + c] >= K).mean(), (te["L_" + c] >= K).mean()
            line.append("K%d %.2f/%.2f%%" % (K, 100 * a, 100 * b))
            if c not in Kc and b <= 0.005:
                Kc[c] = K
        print("  %-8s " % c + "  ".join(line))
    print("  chosen K (test rate <= 0.5%):", Kc)
    flag = np.zeros(len(P), bool)
    for c, K in Kc.items():
        f = P["L_" + c] >= K
        P["F_" + c] = f
        flag |= f.values
        print("  %-8s K=%d  train %.2f%% (%d rows, constant-only %d)  test %.2f%%  ratio %.1fx"
              % (c, K, 100 * f[~P.is_test].mean(), f[~P.is_test].sum(),
                 (f & P["C_" + c])[~P.is_test].sum(), 100 * f[P.is_test].mean(),
                 f[~P.is_test].mean() / max(f[P.is_test].mean(), 1e-9)))
    P["flag"] = flag
    print("  any channel: train %.2f%% test %.2f%%" % (100 * P.flag[~P.is_test].mean(), 100 * P.flag[P.is_test].mean()))
    dd = P[~P.is_test & P.flag].groupby(["farm", "day"]).size().sort_values(ascending=False)
    print("  flagged train farm-days %d; top: %s" % (len(dd), dd.head(12).to_dict()))

    # 2. overlap with physics-rule flags
    fl = pd.read_csv(os.path.join(env.LOCAL, "eda_forensic_11_flags.csv"))
    P = P.merge(fl[["row_id", "Vany"]], on="row_id", how="left")
    T = P[~P.is_test]
    v = T.Vany.fillna(False).astype(bool)
    print("\n2. old rule flags (Vany) train rows %d; overlap with linear flags %d; linear-only %d"
          % (v.sum(), (v & T.flag).sum(), (~v & T.flag).sum()))

    # 3. G_C2 temperature error
    _, lab, _ = load()
    z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
    tt = lab.in_temp.values
    gg = np.where(np.isnan(tt), 1, np.clip((tt - 8) / 2, 0, 1))
    s = "DIAG10"
    base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], 0)
    cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], 0)
    pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{s}.npy")).mean(0)
    lab = lab.assign(e=(0.6 - 0.2 * (1 - gg)) * base + (0.2 + 0.4 * (1 - gg)) * cx + 0.2 * gg * pfn - lab.sub_temp)
    M = lab[["row_id", "e"]].merge(P[["row_id", "flag", "farm", "t", "Vany"]], on="row_id", how="left")
    M = M[M.e.notna()]
    near = np.zeros(len(M), bool)
    ft = set(zip(M.farm[M.flag], M.t[M.flag]))
    for k in range(-3, 4):
        near |= np.array([(f, t + k) in ft for f, t in zip(M.farm, M.t)])
    r = lambda e: float(np.sqrt(np.mean(e ** 2)))
    vv = M.Vany.fillna(False).astype(bool)
    print("\n3. G_C2 DIAG10 RMSE: flagged %.3f (n=%d) | +-3h of flag %.3f (n=%d) | others %.3f | ratio %.2fx"
          % (r(M.e[M.flag]), M.flag.sum(), r(M.e[near]), near.sum(), r(M.e[~near]),
             r(M.e[M.flag]) / r(M.e[~near])))
    print("   linear-only flags (not old rules): %.3f (n=%d)" % (r(M.e[M.flag & ~vv]), (M.flag & ~vv).sum()))
    print("   mean signed error flagged %+.3f others %+.3f" % (M.e[M.flag].mean(), M.e[~near].mean()))

    # 4. EC v2
    o = pd.read_csv(os.path.join(env.ROOT, u"집", u"코덱스", "local", "ec_restart_phase3_20261001_v1",
                                 "oof_predictions.csv"), encoding="utf-8-sig")
    o = o[o.validator == "DIAG10"].merge(P[["row_id", "flag"]], on="row_id", how="left")
    print("\n4. EC v2 DIAG10 RMSE: flagged %.3f (n=%d) others %.3f"
          % (r((o.v2 - o.sub_ec)[o.flag == True]), (o.flag == True).sum(), r((o.v2 - o.sub_ec)[o.flag != True])))

    ok = any((P[~P.is_test]["F_" + c].mean() >= 2 * max(P[P.is_test]["F_" + c].mean(), 1e-9)) for c in Kc) \
        and r(M.e[M.flag]) >= 1.3 * r(M.e[~near])
    print("\nworth a pre-registered weighting test:", ok)
    P.loc[~P.is_test & P.flag, ["row_id"]].to_csv(os.path.join(env.LOCAL, "rd2_linear_flags.csv"), index=False)


if __name__ == "__main__":
    main()
