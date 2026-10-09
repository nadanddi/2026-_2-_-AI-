# -*- coding: utf-8 -*-
"""MX1: mixture of experts for EC - a NORMAL model trained without high-EC days, a HIGH (magnitude) model trained on
high-EC days only, and a causal gate g = P(high day | inputs 0..h).  (2026-10-09 집 클로드, user: "일반적인 날은
고EC 제외 모델로 학습하고, 조건적으로 크기 판독 모델이 고EC만 학습해보는 건 어때").  Fixed before running.
High day = training label day mean >= 1.2 (defined inside each fold from TRAINING labels only).
Per fold and seed s (all R3 = .6 ET + .3 LGB-tweedie + .1 MLP, features FULL w/o day + season + DP1 + THS5 curtain
features of CT2; shrink; clip to training range):
  BASE  R3 trained on all training days                      (comparator = current way + curtain features)
  N     R3 trained on non-high training days
  H     R3 trained on high training days only
  g     LightGBM classifier (binary, 300 trees, lr .05, 15 leaves, min_child 40, seed s) on all training rows,
        same features, target = row's day is high; predict_proba on validation rows
  MIX   (1 - g) N + g H;      ORACLE (descriptive) = N on true normal days, H on true high days
NEW seeds 3737 / 5858 / 7979.  If a fold has < 3 high training days: H = BASE, g = 0 (MIX = N).  Exclusions same farm +-1, lock-40 +-1, other farm d-3..d+3.
Folds: DIAG10 / A / B (judged, ALL rows incl. high days); EL1, P2LOO pass-2 (guard).
RULE (EC rule, k = 1): PASS iff every seed MIX better than BASE on DIAG10, A, B (9/9) AND DIAG10 seed-mean
(farm, day // 5) block bootstrap share(MIX not better) < .025.  GUARD: EL1 or P2LOO pass-2 worse by >= 2 % -> hold.
Also reported: normal / high days, gate AUC, ORACLE, days better, top-day concentration.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u mx1_normal_high_mixture_v1.py {1|2|sum}
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
MODE = sys.argv[1] if len(sys.argv) > 1 else "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("ct1", os.path.join(HERE, "ct1_thermal_schedule_features_v1.py"))
ct1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct1)
wt0, dp1, dc4, p3, core = ct1.wt0, ct1.dp1, ct1.dc4, ct1.p3, ct1.core
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
SEEDS = (3737, 5858, 7979)
CK = os.path.join(env.LOCAL, "mx1_ckpt")
W = (.6, .3, .1)
OTHER = {"F13": "F47", "F47": "F13"}
THS5 = ["th_full_hours_td", "th_night_closed", "th_h9", "th_h9_partial", "th_sched_score"]
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def r3(tr, va, s, FS, BS):
    e, l, m = wt0.members(tr, va, s, FS, BS)
    return W[0] * e + W[1] * l + W[2] * m


def main():
    os.makedirs(CK, exist_ok=True)
    ct1.STAGE = MODE
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    lab = lab.merge(ct1.thermal_features(full), on="row_id", how="left", validate="one_to_one").set_index(lab.index)
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW + THS5
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW + THS5
    for name, i, vd in ct1.build_folds(lab, fds):
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        vd = {(f, d) for f, d in vd if ((lab.farm == f) & (lab.day == d)).any()}
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = ({(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)}
                | {(f, d + j) for f, d in lock for j in (-1, 0, 1)})
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        assert not (tr_m & va_m).any() and va_m.any()
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        hi_tr = (tr.groupby(["farm", "day"]).sub_ec.transform("mean") >= 1.2).values
        lo, hi = tr.sub_ec.min(), tr.sub_ec.max()
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        frame["n_high_train_days"] = tr[hi_tr][["farm", "day"]].drop_duplicates().shape[0]
        for s in SEEDS:
            frame["BASE_%d" % s] = np.clip(core.shrink(r3(tr, va, s, FS, BS), va), lo, hi)
            frame["N_%d" % s] = np.clip(core.shrink(r3(tr[~hi_tr], va, s, FS, BS), va), lo, hi)
            if frame.n_high_train_days.iloc[0] >= 3:
                frame["H_%d" % s] = np.clip(core.shrink(r3(tr[hi_tr], va, s, FS, BS), va), lo, hi)
                clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=15, min_child_samples=40,
                                         subsample=.8, subsample_freq=1, colsample_bytree=.8, random_state=s, verbose=-1, n_jobs=4)
                clf.fit(tr[FS], hi_tr.astype(int))
                frame["g_%d" % s] = clf.predict_proba(va[FS])[:, 1]
            else:   # fixed rule: fewer than 3 high training days -> no high expert (MIX = N)
                frame["H_%d" % s] = frame["BASE_%d" % s]; frame["g_%d" % s] = 0.0
        assert frame.notna().all().all()
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)


def summarize():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    nf = G.groupby("validator").validation_fold.nunique().to_dict(); print("folds:", nf)
    G["dm"] = G.groupby(["validator", "validation_fold", "farm", "day"]).sub_ec.transform("mean")
    for s in SEEDS:
        G["MIX_%d" % s] = (1 - G["g_%d" % s]) * G["N_%d" % s] + G["g_%d" % s] * G["H_%d" % s]
        G["ORA_%d" % s] = np.where(G.dm >= 1.2, G["H_%d" % s], G["N_%d" % s])
    mean = lambda g, c: np.mean([g["%s_%d" % (c, s)] for s in SEEDS], axis=0)
    cells, share, guard = [], None, []
    for v, part in (("DIAG10", "all"), ("A", "all"), ("B", "all"), ("EL1", "pass2"), ("P2LOO", "pass2"),
                    ("DIAG10", "normal"), ("DIAG10", "high"), ("DIAG10", "pass2")):
        g = G[G.validator == v]
        if part == "pass2": g = g[g.day >= 179]
        if part == "normal": g = g[g.dm < 1.2]
        if part == "high": g = g[g.dm >= 1.2]
        if g.empty:
            continue
        y = g.sub_ec.to_numpy(float)
        sr = [(r(g["BASE_%d" % s] - y), r(g["MIX_%d" % s] - y)) for s in SEEDS]
        mB, mM, mN, mH, mO = (mean(g, c) for c in ("BASE", "MIX", "N", "H", "ORA"))
        d = r(mM - y) / r(mB - y) - 1
        try:
            auc = roc_auc_score((g.dm >= 1.2).values, mean(g, "g"))
        except ValueError:
            auc = np.nan
        print("%-6s %-6s days %3d  BASE %.4f  MIX %.4f (%+.1f%%) seeds %s | N %.4f  H %.4f  ORACLE %.4f | gate AUC %.2f, mean g %.3f" % (
            v, part, g[["farm", "day"]].drop_duplicates().shape[0], r(mB - y), r(mM - y), 100 * d,
            "".join("+" if b < a else "-" for a, b in sr), r(mN - y), r(mH - y), r(mO - y), auc, mean(g, "g").mean()))
        if part == "all":
            cells += [b < a for a, b in sr]
        if (v, part) == ("DIAG10", "all"):
            share = ct1.boot_share(g, (mB - y) ** 2, (mM - y) ** 2)
            D = g.assign(gain=(mB - y) ** 2 - (mM - y) ** 2).groupby(["farm", "day"]).gain.sum().sort_values(ascending=False)
            print("        days better %d / %d, top-5 share of gain %.0f%%" % ((D > 0).sum(), len(D), 100 * D.head(5).sum() / D.sum() if D.sum() else np.nan))
        if v in ("EL1", "P2LOO") and d >= .02:
            guard.append(v)
    complete = nf.get("DIAG10") == 10 and nf.get("A") == 5 and nf.get("B") == 5
    if not complete or share is None:
        print("INCOMPLETE - no verdict"); return
    ok = len(cells) == 9 and all(cells) and share < .025
    print("\nVERDICT: seed x {DIAG10,A,B} better %d/%d, DIAG10 share %.4f -> %s%s" % (
        sum(cells), len(cells), share, "PASS" if ok else "FAIL",
        ("  GUARD HOLD: %s" % guard) if guard else ("" if nf.get("P2LOO") == 46 else "  (guard sets not finished)")))


if __name__ == "__main__":
    if MODE != "sum":
        main()
    summarize()
