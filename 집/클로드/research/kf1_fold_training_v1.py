# -*- coding: utf-8 -*-
"""KF1: train the final EC model with 10-fold machinery instead of one full fit?  (2026-10-08 집 클로드,
user: "학습 단계에서 10폴드로 학습해야 좋지 않을까? ... EC 먼저 해보자").  Fixed before running.

Recipe = current R3 (.6 ET + .3 LGB-tweedie + .1 MLP; season DC4 + DP1 features; members as WT0),
then shrink + clip(train range), seeds 47 / 1414 / 6464.  Configurations, per outer fold and seed s:
  FULL   one fit of each member on all outer-training rows                     (= current way)
  BAG10  outer-training days split per farm into 5-day chunks dealt round-robin into 10 groups; member k
         fitted on the training rows minus group k (90 %), same random_state s; prediction = mean of 10
  ESCV   ET as FULL (same fitted model); LGB rounds and MLP epochs chosen by inner 5-fold CV (5-day chunks
         round-robin into 5 groups, inner train drops the inner-val days +-1), LGB: early stopping 100 on
         inner val (max 3000 rounds, n* = round(mean best iteration)); MLP: same net, early_stopping off,
         one partial_fit per epoch up to 800, patience 25 on inner-val RMSE (epoch* = round(mean best
         epoch)); then LGB(n*) and MLP(epoch* epochs) refitted on all outer-training rows.
Outer folds: DIAG10 (10), A (5), B (5) [judged], EL1 pass-2 (10) [guard].  Exclusions: same farm +-1 day,
lock-40 +-1 (house rule) AND the other farm's days d-3..d+3 (FX1 finding 2026-10-08: F47 test days =
F13 test days - 2), identical for all configurations.
Rule (EC rule of 2026-10-04, k = 2 candidates, alpha = .025 / 2 = .0125), candidate X vs FULL:
  PASS iff every seed improves on DIAG10, A and B (seed-wise X_s vs FULL_s)
       AND DIAG10 seed-mean (farm, day // 5) block bootstrap share(X not better) < .0125 (20,000 draws).
  GUARD (not a pass condition): EL1 pass-2 or DIAG10 pass-2 seed-mean worse by >= 2 % -> hold, report.
No submission files.  Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u kf1_fold_training_v1.py {1|2|sum}
  stage 1 = DIAG10 + A + B, stage 2 = EL1 (parallel-safe), sum = summary only.
"""
import env  # noqa: F401
import importlib.util, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
import lightgbm as lgb
SEEDS = wt0.SEEDS
CK = os.path.join(env.LOCAL, "kf1_ckpt")
W = (.6, .3, .1)
CONFIGS = ("FULL", "BAG10", "ESCV")
OTHER = {"F13": "F47", "F47": "F13"}
ALPHA = .025 / 2
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def chunk_groups(df, k):
    """(farm, day) -> group, 5-day chunks per farm dealt round-robin into k groups."""
    g = {}
    for f in ("F13", "F47"):
        days = sorted(df[df.farm == f].day.unique())
        for i in range(0, len(days), 5):
            for d in days[i:i + 5]:
                g[(f, int(d))] = (i // 5) % k
    return np.array([g[(f, int(d))] for f, d in zip(df.farm, df.day)])


def mlp_net(s, **kw):
    opts = dict(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3, max_iter=800,
                early_stopping=True, n_iter_no_change=25, validation_fraction=.12, random_state=s)
    opts.update(kw)
    return MLPRegressor(**opts)


def escv_choose(tr, s, BS):
    grp = chunk_groups(tr, 5); keys = list(zip(tr.farm, tr.day.astype(int)))
    best_l, best_m = [], []
    for k in range(5):
        vd = {kk for kk, gg in zip(keys, grp) if gg == k}
        near = {(f, d + j) for f, d in vd for j in (-1, 0, 1)}
        itr = np.array([kk not in near for kk in keys]); iva = grp == k
        a, b = tr[itr], tr[iva]
        m = core.lg(s, "tweedie"); m.set_params(n_estimators=3000)
        m.fit(a[BS], a.sub_ec.to_numpy(), eval_set=[(b[BS], b.sub_ec.to_numpy())],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        best_l.append(m.best_iteration_)
        pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(a[BS])
        Xa, Xb = pre.transform(a[BS]), pre.transform(b[BS]); ya, yb = a.sub_ec.to_numpy(), b.sub_ec.to_numpy()
        net = mlp_net(s, early_stopping=False)
        best, best_ep, wait = np.inf, 0, 0
        for ep in range(1, 801):
            net.partial_fit(Xa, ya)
            e = r(net.predict(Xb) - yb)
            if e < best - 1e-9:
                best, best_ep, wait = e, ep, 0
            else:
                wait += 1
                if wait >= 25:
                    break
        best_m.append(best_ep)
    return int(round(np.mean(best_l))), max(1, int(round(np.mean(best_m)))), best_l, best_m


def build_folds(lab, fds):
    if STAGE == "1":
        return [x for x in fds if x[0] in ("DIAG10", "A", "B")]
    folds = []
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    return folds


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    for name, i, vd in build_folds(lab, fds):
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        vd = {(f, d) for f, d in vd if ((lab.farm == f) & (lab.day == d)).any()}
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = ({(f, d + j) for f, d in vd for j in (-1, 0, 1)}
                | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)}
                | {(f, d + j) for f, d in lock for j in (-1, 0, 1)})
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        assert not (tr_m & va_m).any() and va_m.any()
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        frame["lo"], frame["hi"] = tr.sub_ec.min(), tr.sub_ec.max()
        grp10 = chunk_groups(tr, 10)
        for s in SEEDS:
            e, l, m = wt0.members(tr, va, s, FS, BS)
            for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                frame["FULL_%s_%d" % (nm, s)] = core.shrink(v, va)
            frame["ESCV_et_%d" % s] = frame["FULL_et_%d" % s]
            acc = {"et": 0., "lgb": 0., "mlp": 0.}
            for k in range(10):
                eb, lb, mb = wt0.members(tr[grp10 != k], va, s, FS, BS)
                acc["et"] += eb / 10; acc["lgb"] += lb / 10; acc["mlp"] += mb / 10
            for nm in acc:
                frame["BAG10_%s_%d" % (nm, s)] = core.shrink(acc[nm], va)
            n_l, n_m, bl, bm = escv_choose(tr, s, BS)
            ml = core.lg(s, "tweedie"); ml.set_params(n_estimators=n_l)
            frame["ESCV_lgb_%d" % s] = core.shrink(core.predict_model(ml, tr, va, BS), va)
            mm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               mlp_net(s, early_stopping=False, max_iter=n_m, n_iter_no_change=n_m + 1))
            frame["ESCV_mlp_%d" % s] = core.shrink(core.predict_model(mm, tr, va, BS), va)
            frame["ESCV_nlgb_%d" % s], frame["ESCV_nmlp_%d" % s] = n_l, n_m
            print("%s/%d seed %d  ESCV lgb %d %s mlp %d %s" % (name, i, s, n_l, bl, n_m, bm), flush=True)
        assert frame.notna().all().all()
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)


def boot_share(G, a, b, n=20000, seed=0):
    blk = (G.farm + "_" + (G.day // 5).astype(str)).values
    keys, inv = np.unique(blk, return_inverse=True)
    sa = np.bincount(inv, weights=a, minlength=len(keys)); sb = np.bincount(inv, weights=b, minlength=len(keys))
    rng = np.random.default_rng(seed); worse = 0
    for _ in range(n // 1000):
        ix = rng.integers(0, len(keys), size=(1000, len(keys)))
        worse += int((sb[ix].sum(1) >= sa[ix].sum(1)).sum())
    return worse / n


def summarize():
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    print("folds:", G.groupby("validator").validation_fold.nunique().to_dict())

    def pred(g, c, s):
        return np.clip(W[0] * g["%s_et_%d" % (c, s)] + W[1] * g["%s_lgb_%d" % (c, s)] + W[2] * g["%s_mlp_%d" % (c, s)],
                       g.lo, g.hi).to_numpy(float)
    res = {}
    for v, lo_day in (("DIAG10", 0), ("A", 0), ("B", 0), ("DIAG10", 179), ("EL1", 179)):
        g = G[(G.validator == v) & (G.day >= lo_day)]
        if g.empty:
            continue
        y = g.sub_ec.to_numpy(float)
        print("\n== %s day>=%d  n=%d" % (v, lo_day, len(g)))
        seedR, mean = {}, {}
        for c in CONFIGS:
            ps = [pred(g, c, s) for s in SEEDS]
            seedR[c] = [r(p - y) for p in ps]; mean[c] = np.mean(ps, axis=0)
            d = r(mean[c] - y) / r(mean["FULL"] - y) - 1
            print("   %-5s %.4f  seeds [%s]  vs FULL %+.2f%%" % (c, r(mean[c] - y), " ".join("%.4f" % x for x in seedR[c]), 100 * d))
            res[(v, lo_day, c)] = (seedR[c], d)
        if v == "DIAG10" and lo_day == 0:
            for c in ("BAG10", "ESCV"):
                res[("P", c)] = boot_share(g, (mean["FULL"] - y) ** 2, (mean[c] - y) ** 2)
                print("   %s share(not better than FULL) %.4f" % (c, res[("P", c)]))
    for c in ("BAG10", "ESCV"):
        cells = [(v, a < b) for v in ("DIAG10", "A", "B") if (v, 0, c) in res
                 for a, b in zip(res[(v, 0, c)][0], res[(v, 0, "FULL")][0])]
        allbetter = len(cells) == 9 and all(x for _, x in cells)
        pw = res.get(("P", c), 1.0)
        guard = [k for k in (("DIAG10", 179), ("EL1", 179)) if (k[0], k[1], c) in res and res[(k[0], k[1], c)][1] >= .02]
        print("\n%s: seed x {DIAG10,A,B} better %d/%d, DIAG10 share %.4f -> %s%s" %
              (c, sum(x for _, x in cells), len(cells), pw, "PASS" if allbetter and pw < ALPHA else "FAIL",
               ("  GUARD hit: %s" % guard) if guard else ""))


if __name__ == "__main__":
    if STAGE != "sum":
        main()
    summarize()
