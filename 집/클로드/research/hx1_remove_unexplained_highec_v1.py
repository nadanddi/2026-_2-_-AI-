# -*- coding: utf-8 -*-
"""HX1: what if the 'unexplainable' high-EC days are artificial (not in the test)?  Train without them and
validate.  (2026-10-09 집 클로드, user: "설명 불가능한 정도의 고EC날 제거하고 검증 돌려봐").  Fixed before running.

Day sets (labels of the 360 non-lock days; defined BEFORE this run from the stored WT0 DIAG10 out-of-fold
predictions of the current R3 recipe, seed mean, = local/ec3_WT0_all.csv):
  HIGH = day-mean EC >= 1.2                                           (26 days)
  UNEX = HIGH and day-mean (label - OOF prediction) >= 0.3            (14 days; 5 in pass 2)
Configurations (current R3 = .6 ET + .3 LGB-tweedie + .1 MLP, season DC4 + DP1, seeds 47 / 1414 / 6464):
  BASE  all training days            U  training without UNEX days            A  training without HIGH days
Outer folds DIAG10 (10), A (5), B (5), EL1 pass-2 (10); exclusions same farm +-1, lock-40 +-1 and the other
farm's d-3..d+3 (as KF1).  Validation rows are never dropped; summaries are computed on row subsets.
PRIMARY rows = NORMAL rows (day not in HIGH): the question is "if the test has no such days, does
removing them from training help on the other days?"
Rule (EC rule 2026-10-04, k = 2, alpha .0125), X in {U, A} vs BASE on NORMAL rows:
  PASS iff every seed improves on DIAG10, A and B AND DIAG10 seed-mean (farm, day // 5) block bootstrap
  share(X not better) < .0125.  Descriptive: EL1 pass-2 NORMAL rows, all rows, HIGH rows, UNEX rows.
Diagnostic of a hypothesis; no submission.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u hx1_remove_unexplained_highec_v1.py {1|2|sum}
"""
import env  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
SEEDS = wt0.SEEDS
CK = os.path.join(env.LOCAL, "hx1_ckpt")
W = (.6, .3, .1)
CONFIGS = ("BASE", "U", "A")
OTHER = {"F13": "F47", "F47": "F13"}
ALPHA = .025 / 2
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def day_sets():
    O = pd.read_csv(os.path.join(env.LOCAL, "ec3_WT0_all.csv")); O = O[O.validator == "DIAG10"]
    O["p"] = np.mean([np.clip(W[0] * O["et_%d" % s] + W[1] * O["lgb_%d" % s] + W[2] * O["mlp_%d" % s], O.lo, O.hi)
                      for s in SEEDS], axis=0)
    D = O.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean"))
    high = {(f, int(d)) for (f, d), v in D.y.items() if v >= 1.2}
    unex = {(f, int(d)) for (f, d), row in D.iterrows() if row.y >= 1.2 and row.y - row.p >= .3}
    assert len(high) == 26 and len(unex) == 14, (len(high), len(unex))
    return high, unex


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
    high, unex = day_sets()
    with open(os.path.join(CK, "..", "hx1_day_sets.json"), "w", encoding="utf-8") as fh:
        json.dump({"HIGH": sorted(map(list, high)), "UNEX": sorted(map(list, unex))}, fh)
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
        keys = list(zip(tr.farm, tr.day.astype(int)))
        subsets = {"BASE": np.ones(len(tr), bool),
                   "U": np.array([k not in unex for k in keys]), "A": np.array([k not in high for k in keys])}
        for s in SEEDS:
            for c in CONFIGS:
                e, l, m = wt0.members(tr[subsets[c]], va, s, FS, BS)
                for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                    frame["%s_%s_%d" % (c, nm, s)] = core.shrink(v, va)
            print("%s/%d seed %d" % (name, i, s), flush=True)
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
    high, unex = day_sets()
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    nf = G.groupby("validator").validation_fold.nunique().to_dict(); print("folds:", nf)
    complete = nf.get("DIAG10") == 10 and nf.get("A") == 5 and nf.get("B") == 5
    k = list(zip(G.farm, G.day))
    G["is_high"] = [x in high for x in k]; G["is_unex"] = [x in unex for x in k]

    def pred(g, c, s):
        return np.clip(W[0] * g["%s_et_%d" % (c, s)] + W[1] * g["%s_lgb_%d" % (c, s)] + W[2] * g["%s_mlp_%d" % (c, s)],
                       g.lo, g.hi).to_numpy(float)
    res = {}
    views = [("DIAG10", "NORMAL"), ("A", "NORMAL"), ("B", "NORMAL"), ("EL1", "NORMAL p2"),
             ("DIAG10", "ALL rows"), ("DIAG10", "HIGH"), ("DIAG10", "UNEX"), ("EL1", "ALL p2")]
    for v, sub in views:
        g = G[G.validator == v]
        if sub.startswith("NORMAL"): g = g[~g.is_high]
        if sub == "HIGH": g = g[g.is_high]
        if sub == "UNEX": g = g[g.is_unex]
        if v == "EL1": g = g[g.day >= 179]
        if g.empty:
            continue
        y = g.sub_ec.to_numpy(float)
        print("\n== %s %s  n=%d days=%d" % (v, sub, len(g), g[["farm", "day"]].drop_duplicates().shape[0]))
        seedR, mean = {}, {}
        for c in CONFIGS:
            ps = [pred(g, c, s) for s in SEEDS]
            seedR[c] = [r(p - y) for p in ps]; mean[c] = np.mean(ps, axis=0)
            print("   %-4s %.4f  seeds [%s]  vs BASE %+.2f%%" % (c, r(mean[c] - y), " ".join("%.4f" % x for x in seedR[c]),
                                                                 100 * (r(mean[c] - y) / r(mean["BASE"] - y) - 1)))
            res[(v, sub, c)] = seedR[c]
        if (v, sub) == ("DIAG10", "NORMAL"):
            for c in ("U", "A"):
                res[("P", c)] = boot_share(g, (mean["BASE"] - y) ** 2, (mean[c] - y) ** 2)
                print("   %s share(not better than BASE) %.4f" % (c, res[("P", c)]))
    if not complete:
        print("\nINCOMPLETE folds - no verdict"); return
    for c in ("U", "A"):
        cells = [a < b for v in ("DIAG10", "A", "B") for a, b in zip(res[(v, "NORMAL", c)], res[(v, "NORMAL", "BASE")])]
        ok = len(cells) == 9 and all(cells) and res[("P", c)] < ALPHA
        print("%s: seed x {DIAG10,A,B} better %d/9, share %.4f -> %s" % (c, sum(cells), res[("P", c)], "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    if STAGE != "sum":
        main()
    summarize()
