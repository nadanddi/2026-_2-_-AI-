# -*- coding: utf-8 -*-
"""TB1 (2026-10-12 집 클로드; user "그렇게 해봐": judge the mid_Tin lead (6.596) in an evaluation-like geometry).
Geometry: linker v7 MIRROR CV (cv_all400_choice_v7m.csv) already hides 10-record chunks + deleted neighbours + the other ID's
mirror chunk, like the evaluation set.  Pass-2 has only ~21 training records per farm, so judgement pools pass 1 + pass 2 with
leave-one-CHUNK-out cross-fitting; pass 2 reported separately (descriptive).
Deployable multi-step anchor (linker choices only, no truth):
  t -> p1 (unlabelled, same ID, p >= .8) -> p2 ... -> pk (labelled, p >= .8), k = 2..4, every step p >= .8, intermediate records
  same ID (their inputs = same greenhouse earlier inputs; allowed); pk may be other-farm TRAINING (allowed).
  gap = A23(pk) - .36 - q_t(0 h);  mid = mean in_temp over intermediate records p1..p(k-1) (all hours) - mean in_temp(pk)
  G0: pred_h = base_h + b_h * clip(gap, +-2)            G1: pred_h = base_h + b_h * clip(gap, +-2) + g_h * mid
  b_h, g_h: per-hour LS on the OTHER chunks (records of the target chunk excluded), pooled over the 4 DIAG10 cells.
  base = round-11 recipe q + submitted 1-step anchor rule (only touches records with a labelled 1-step anchor; disjoint set).
PASS (G1) iff vs base: all 4 cells better on the pooled set AND record bootstrap P(worse) < .025 AND pass-2 cell mean better;
and G1 vs G0 also all 4 cells better with P(worse) < .025 (mid adds beyond the bare multi-step anchor).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u tb1_temp_multistep_anchor_midtin_v1.py
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta7", os.path.join(HERE, "ta7_anchor_sweep_v1.py")); A7 = importlib.util.module_from_spec(s); s.loader.exec_module(A7)
A2 = A7.A2; rm = A2.rm


def chains(L):
    Li = L.set_index("rec"); out = {}
    for t in Li.index:
        r = Li.loc[t]
        if r.prev_is_labelled or r.xf or r.p < .8:
            continue
        path = [r.prev]; ok = False
        for _ in range(3):
            c = path[-1]
            if c not in Li.index:
                break
            rc = Li.loc[c]
            if rc.p < .8:
                break
            if rc.prev_is_labelled:
                if np.isfinite(rc.anchor_st23):
                    path.append(rc.prev); ok = True; out[t] = dict(path=path, k=len(path), A=rc.anchor_st23)
                break
            if rc.xf:
                break
            path.append(rc.prev)
        _ = ok
    return out


def main():
    L = pd.read_csv(A7.LK)
    La = L[L.prev_is_labelled.astype(bool) & L.anchor_st23.notna() & (L.steps_to_anchor == 1)].copy()
    C = chains(L)
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp"]); X["rec"] = X.row_id.str[:7]
    TM = X.groupby("rec").in_temp.mean()
    for t, d in C.items():
        mids = d["path"][:-1]; d["mid"] = TM.reindex(mids).mean() - TM.get(d["path"][-1], np.nan)
    C = {t: d for t, d in C.items() if np.isfinite(d["mid"])}
    ks = pd.Series({t: d["k"] for t, d in C.items()})
    print("multi-step chains: %d records (k: %s), pass-2 %d" % (len(C), ks.value_counts().sort_index().to_dict(), int((ks.index.str[4:7].astype(int) >= 179).sum())), flush=True)
    R = A2.A1.frames(); M, cells = R["DIAG10"]; key, A = A7.build(M, La); h = M.hour.values; y = M.sub_temp.values
    base = [A7.correct(M, q, key, A, .8, "const", "lin8", 1, 1.5) for q in cells]
    lo = {f: int(M[M.farm == f].day.min()) for f in ("F13", "F47")}
    chunk = {t: (t[:3], (int(t[4:7]) - lo[t[:3]]) // 10) for t in C}
    recs = [t for t in C if (key == t).any()]
    q0 = [pd.Series(q[h == 0], index=key[h == 0]) for q in cells]
    gap = [{t: float(np.clip(C[t]["A"] - .36 - qq[t], -2, 2)) for t in recs} for qq in q0]
    mid = {t: C[t]["mid"] for t in recs}
    rows_of = {t: np.where(key == t)[0] for t in recs}

    def fit(train, use_mid):
        # per hour LS of (y - base) on [gap] or [gap, mid], pooled cells
        Bs = np.zeros((24, 2))
        for hh in range(24):
            Xr, Yr = [], []
            for ci, b in enumerate(base):
                for t in train:
                    i = rows_of[t][h[rows_of[t]] == hh]
                    if len(i):
                        Xr.append([gap[ci][t], mid[t]] if use_mid else [gap[ci][t], 0.0]); Yr.append(y[i[0]] - b[i[0]])
            Xr = np.array(Xr); Yr = np.array(Yr)
            if use_mid:
                Bs[hh] = np.linalg.lstsq(Xr.T @ Xr + 1e-6 * np.eye(2), Xr.T @ Yr, rcond=None)[0]
            else:
                Bs[hh, 0] = (Xr[:, 0] @ Yr) / max(Xr[:, 0] @ Xr[:, 0], 1e-9)
        return Bs
    preds = {"G0": [b.copy() for b in base], "G1": [b.copy() for b in base]}
    for ch in sorted(set(chunk[t] for t in recs)):
        te = [t for t in recs if chunk[t] == ch]; tr = [t for t in recs if chunk[t] != ch]
        for arm, um in (("G0", False), ("G1", True)):
            Bs = fit(tr, um)
            for ci in range(len(base)):
                for t in te:
                    i = rows_of[t]; preds[arm][ci][i] = base[ci][i] + Bs[h[i], 0] * gap[ci][t] + Bs[h[i], 1] * mid[t]
    Bfull = fit(recs, True); print("full-fit G1 b_h: " + " ".join("%d:%.2f" % (i, v) for i, v in enumerate(Bfull[:, 0]) if i % 3 == 0) +
                                   " | g_h: " + " ".join("%d:%.2f" % (i, v) for i, v in enumerate(Bfull[:, 1]) if i % 3 == 0))
    sel = np.isin(key, recs); p2 = sel & (M.day.values >= 179); rng = np.random.default_rng(20261215); ok = True
    for nm, a, b in (("G0 vs base", "G0", None), ("G1 vs base", "G1", None), ("G1 vs G0", "G1", "G0")):
        A_ = preds[a]; B_ = base if b is None else preds[b]
        rel = [rm((x - y)[sel]) / rm((z - y)[sel]) - 1 for x, z in zip(A_, B_)]
        rel2 = [rm((x - y)[p2]) / rm((z - y)[p2]) - 1 for x, z in zip(A_, B_)] if p2.any() else [np.nan]
        P = pd.concat([pd.Series(((x - y) ** 2 - (z - y) ** 2)[sel]).groupby(key[sel]).sum() for x, z in zip(A_, B_)], axis=1).mean(1); P = P[P.abs() > 1e-12]
        idx = rng.integers(0, len(P), (20000, len(P))); pw = float((P.values[idx].sum(1) >= 0).mean())
        print("  %-11s pooled cells %s | pass-2 mean %+.2f%% | records %d improved %d | P(worse) %.4f | top share %.0f%%" % (
            nm, " ".join("%+.2f%%" % (100 * v) for v in rel), 100 * np.nanmean(rel2), len(P), int((P < 0).sum()), pw, 100 * P.min() / P.sum() if P.sum() < 0 else np.nan), flush=True)
        if nm != "G0 vs base":
            ok &= all(v < 0 for v in rel) and pw < .025 and (nm == "G1 vs G0" or np.nanmean(rel2) < 0)
    # whole-layout effect (all DIAG10 pass-1+2 rows) for scale
    print("  whole DIAG10 RMSE change G1 vs base: pass-1 %+.2f%%, pass-2 %+.2f%%" % (
        100 * np.mean([rm((x - y)[M.day.values < 179]) / rm((z - y)[M.day.values < 179]) - 1 for x, z in zip(preds["G1"], base)]),
        100 * np.mean([rm((x - y)[M.day.values >= 179]) / rm((z - y)[M.day.values >= 179]) - 1 for x, z in zip(preds["G1"], base)])))
    print("VERDICT TB1 (G1):", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
