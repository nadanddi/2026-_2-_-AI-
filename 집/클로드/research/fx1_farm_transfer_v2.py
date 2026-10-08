# -*- coding: utf-8 -*-
"""FX1 v2: does learning from the OTHER greenhouse help or hurt EC?  (diagnostic; 2026-10-08 집 클로드,
user question "F13에서 학습한 내용이 F47에도 도움이 되나, 아니면 오차를 만드나?")
Fixed before running.  v1 stopped after 1 fold (no result looked at) because the plan-stage critic found
that the other greenhouse's SAME-DATE labels stayed in training: the test days of F47 are exactly the F13
test days minus 2 (checked on test_X), so at test time neither greenhouse has labels for the matching
period, but v1 validation only removed same-farm +-1 days.

Changes from v1:
  * every validation day (f, d) also removes the OTHER farm's days d-3 .. d+3 from training (covers the
    d-2 / d+2 date offset +-1), in ALL configurations; lock-40 exclusion unchanged.
  * P is refitted in the same run (no stored WT0 members), so all configurations share one code path.
  * O (other farm only) added, descriptive.
  * ONE primary set: P2LOO pass-2 rows (structure-matched to the test, all test days are pass 2).
    Others (DIAG10 all / pass-2, EL1 pass-2) are descriptive.

Configurations (R3 = .6 ET + .3 LGB-tweedie + .1 MLP, season DC4 + DP1 features, members as WT0;
seeds 47 / 1414 / 6464; season index computed from the POOLED training days for every configuration, so
S/O differ from P only in the LABELLED rows the members are fitted on):
  P  pooled F13+F47, no farm feature (current recipe)
  S  same farm only            O  other farm only (descriptive)
  I  pooled + farm indicator is_f47
final = clip(R3, lo, hi), lo/hi = pooled training range of the fold (same for all configurations).

Primary reading, per farm f and X in {S, I} vs P on P2LOO pass-2 rows of f
(alpha = .025 / (2 farms x 2 alternatives) = .00625, one-sided in the stated direction):
  S: "other farm HELPS f" iff P beats S on every seed AND bootstrap share(P not better) < .00625
     "other farm HURTS f" iff S beats P on every seed AND bootstrap share(S not better) < .00625
  I: "farm flag helps" / "hurts" with the same rule.   Otherwise "판별 불가".
Bootstrap: seed-mean predictions, resampling (farm, day // 5) blocks, 20,000 draws; a resampling share,
not a null-hypothesis p-value (house convention).  Diagnostic only: no adoption, no submission.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u fx1_farm_transfer_v2.py {1|2|sum}
  stage 1 = DIAG10 folds, stage 2 = P2LOO + EL1 folds (can run in parallel), sum = summary only.
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
STAGE = sys.argv[1] if len(sys.argv) > 1 else "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
SEEDS = wt0.SEEDS
CK = os.path.join(env.LOCAL, "fx1v2_ckpt")
W = (.6, .3, .1)
CONFIGS = ("P", "S", "O", "I")
OTHER = {"F13": "F47", "F47": "F13"}
ALPHA = .025 / 4


def boot_share(G, a, b, n=20000, seed=0):
    """share of block resamples in which squared-error sum b >= a (b not better than a)."""
    blk = (G.farm + "_" + (G.day // 5).astype(str)).values
    keys, inv = np.unique(blk, return_inverse=True)
    sa = np.bincount(inv, weights=a, minlength=len(keys)); sb = np.bincount(inv, weights=b, minlength=len(keys))
    rng = np.random.default_rng(seed); worse = 0
    for _ in range(n // 1000):
        ix = rng.integers(0, len(keys), size=(1000, len(keys)))
        worse += int((sb[ix].sum(1) >= sa[ix].sum(1)).sum())
    return worse / n


def build_folds(lab, fds):
    if STAGE == "1":
        return [x for x in fds if x[0] == "DIAG10"]
    folds = []
    for f in ("F13", "F47"):
        for d in sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique()):
            folds.append(("P2LOO", 10 + len(folds), {(f, int(d))}))
    for f in ("F13", "F47"):
        ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
        for k in range(0, len(ds), 5):
            folds.append(("EL1", 10 + len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    return folds


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    lab["is_f47"] = (lab.farm == "F47").astype(float)
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    for name, i, vd in build_folds(lab, fds):
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = ({(f, d + j) for f, d in vd for j in (-1, 0, 1)}
                | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)}
                | {(f, d + j) for f, d in lock for j in (-1, 0, 1)})
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        assert not (tr_m & va_m).any()
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        frame["lo"], frame["hi"] = tr.sub_ec.min(), tr.sub_ec.max()
        frame["n_tr_same"] = [int((tr.farm == f).sum()) for f in va.farm]
        frame["n_tr_other"] = [int((tr.farm == OTHER[f]).sum()) for f in va.farm]
        for s in SEEDS:
            for c in CONFIGS:
                for nm in ("et", "lgb", "mlp"):
                    frame["%s_%s_%d" % (c, nm, s)] = np.nan
            for c, fs, bs in (("P", FS, BS), ("I", FS + ["is_f47"], BS + ["is_f47"])):
                e, l, m = wt0.members(tr, va, s, fs, bs)
                for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                    frame["%s_%s_%d" % (c, nm, s)] = core.shrink(v, va)
            for f in ("F13", "F47"):
                vm = (va.farm == f).values
                if not vm.any():
                    continue
                vaf = va[vm]
                for c, src in (("S", f), ("O", OTHER[f])):
                    e, l, m = wt0.members(tr[tr.farm == src], vaf, s, FS, BS)
                    for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                        frame.loc[vm, "%s_%s_%d" % (c, nm, s)] = core.shrink(v, vaf)
            print("%s/%d seed %d" % (name, i, s), flush=True)
        assert frame.filter(regex="^[PSOI]_").notna().all().all()
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)


def summarize():
    files = sorted(os.listdir(CK))
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in files], ignore_index=True)
    print("folds:", G.groupby("validator").validation_fold.nunique().to_dict())
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))

    def pred(g, c, s):
        return np.clip(W[0] * g["%s_et_%d" % (c, s)] + W[1] * g["%s_lgb_%d" % (c, s)] + W[2] * g["%s_mlp_%d" % (c, s)],
                       g.lo, g.hi).to_numpy(float)
    for v, lo_day, primary in (("P2LOO", 179, True), ("DIAG10", 179, False), ("DIAG10", 0, False), ("EL1", 179, False)):
        print("\n== %s rows day>=%d %s" % (v, lo_day, "(PRIMARY)" if primary else "(descriptive)"))
        for f in ("F13", "F47"):
            g = G[(G.validator == v) & (G.farm == f) & (G.day >= lo_day)]
            if g.empty:
                continue
            y = g.sub_ec.to_numpy(float)
            seedR, mean = {}, {}
            for c in CONFIGS:
                ps = [pred(g, c, s) for s in SEEDS]
                seedR[c] = [r(p - y) for p in ps]; mean[c] = np.mean(ps, axis=0)
            print("%s n=%d days=%d  same-farm train rows %.0f, other-farm %.0f" %
                  (f, len(g), g.day.nunique(), g.n_tr_same.mean(), g.n_tr_other.mean()))
            for c in CONFIGS:
                print("   %s RMSE %.4f  seeds [%s]  vs P %+.2f%%" % (c, r(mean[c] - y), " ".join("%.4f" % x for x in seedR[c]),
                                                                      100 * (r(mean[c] - y) / r(mean["P"] - y) - 1)))
            for c in ("S", "I", "O"):
                better = sum(a < b for a, b in zip(seedR[c], seedR["P"])); worse = sum(a > b for a, b in zip(seedR[c], seedR["P"]))
                eP, eC = (mean["P"] - y) ** 2, (mean[c] - y) ** 2
                c_not_better, p_not_better = boot_share(g, eP, eC), boot_share(g, eC, eP)
                verdict = "-"
                if primary and c in ("S", "I"):
                    verdict = "판별 불가"
                    good, bad = ("다른 온실 학습이 도움", "다른 온실 학습이 오차를 만듦") if c == "S" else ("온실 표시가 해로움", "온실 표시가 도움")
                    if worse == 3 and p_not_better < ALPHA: verdict = good
                    if better == 3 and c_not_better < ALPHA: verdict = bad
                print("   %s vs P: seeds better %d/3, share(%s not better) %.4f, share(P not better) %.4f  %s"
                      % (c, better, c, c_not_better, p_not_better, verdict))


if __name__ == "__main__":
    if STAGE != "sum":
        main()
    summarize()
