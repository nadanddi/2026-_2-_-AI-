# -*- coding: utf-8 -*-
"""FX1: does learning from the OTHER greenhouse help or hurt EC?  (diagnostic; 2026-10-08 집 클로드,
user question "F13에서 학습한 내용이 F47에도 도움이 되나, 아니면 오차를 만드나?")
Fixed before running.

Configurations (current R3 recipe = .6 ET + .3 LGB-tweedie + .1 MLP, season DC4 + DP1 features, as WT0;
seeds 47 / 1414 / 6464; same folds, same +-1 day and lock-40 exclusions; season index computed exactly as
in WT0 from the POOLED training days, so only the rows the members are fitted on differ):
  P  pooled F13+F47, no farm feature        = stored WT0 members (local/ec3_WT0_all.csv), not refitted
  S  same farm only: members for farm f are fitted on f's training rows only, predicted on f's rows
  I  pooled + farm indicator column (is_f47 = 1 for F47) added to all three members' features
final = clip(.6 e + .3 l + .1 m, lo, hi) with lo/hi = the pooled training range stored in WT0 (same for all).

Sets (stage 1 = DIAG10 all rows; stage 2 = P2LOO and EL1 pass-2 rows, run after stage 1 if time allows).
Reading rule per farm f in {F13, F47} and per alternative X in {S, I} vs P, on f's rows only:
  "other farm HELPS f"   iff P beats S on every seed AND seed-mean 5-day-block bootstrap P(S not worse) < .0125
  "other farm HURTS f"   iff S beats P on every seed AND bootstrap P(S worse) < .0125
  otherwise "판별 불가".  Same wording for I ("farm flag helps/hurts").  .0125 = .025 / 2 (two farms).
This is a diagnostic: no adoption, no submission.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u fx1_farm_transfer_v1.py [stage]
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
CK = os.path.join(env.LOCAL, "fx1_ckpt")
W = (.6, .3, .1)


def r3(e, l, m, lo, hi):
    return np.clip(W[0] * e + W[1] * l + W[2] * m, lo, hi)


def boot_p_worse(G, a, b, n=20000, seed=0):
    """P(b not better than a) by 5-day blocks of (farm, day); a = reference errors^2 sum per block."""
    blk = (G.farm + "_" + (G.day // 5).astype(str)).values
    keys, inv = np.unique(blk, return_inverse=True)
    sa = np.bincount(inv, weights=a, minlength=len(keys)); sb = np.bincount(inv, weights=b, minlength=len(keys))
    rng = np.random.default_rng(seed); worse = 0
    for _ in range(n // 1000):
        ix = rng.integers(0, len(keys), size=(1000, len(keys)))
        worse += int((sb[ix].sum(1) >= sa[ix].sum(1)).sum())
    return worse / n


def main():
    os.makedirs(CK, exist_ok=True)
    raw, full, lab, lock, signatures, fds = p3.prepare()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    lab["is_f47"] = (lab.farm == "F47").astype(float)
    wv = dc4.weather_vectors(full)
    FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    BS = [c for c in core.BASE if c != "day"] + ["season"] + dp1.NEW
    folds = [x for x in fds if x[0] == "DIAG10"]
    if STAGE == "2":
        folds = []
        for f in ("F13", "F47"):
            for d in sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique()):
                folds.append(("P2LOO", 10 + len(folds), {(f, int(d))}))
        for f in ("F13", "F47"):
            ds = sorted(lab[(lab.farm == f) & (lab.day >= 179)].day.unique())
            for k in range(0, len(ds), 5):
                folds.append(("EL1", 10 + len(folds), {(f, int(d)) for d in ds[k:k + 5]}))
    for name, i, vd in folds:
        path = os.path.join(CK, "%s_%d.csv" % (name, i))
        if os.path.exists(path):
            continue
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        frame = va[["row_id", "farm", "day", "hour", "sub_ec"]].copy()
        frame["validator"], frame["validation_fold"] = name, i
        for s in SEEDS:
            # S: same-farm members
            for nm in ("et", "lgb", "mlp"):
                frame["S_%s_%d" % (nm, s)] = np.nan
            for f in ("F13", "F47"):
                vm = (va.farm == f).values
                if not vm.any():
                    continue
                trf, vaf = tr[tr.farm == f], va[vm]
                e, l, m = wt0.members(trf, vaf, s, FS, BS)
                for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                    frame.loc[vm, "S_%s_%d" % (nm, s)] = core.shrink(v, vaf)
            # I: pooled + farm flag
            e, l, m = wt0.members(tr, va, s, FS + ["is_f47"], BS + ["is_f47"])
            for nm, v in (("et", e), ("lgb", l), ("mlp", m)):
                frame["I_%s_%d" % (nm, s)] = core.shrink(v, va)
            print("%s/%d seed %d" % (name, i, s), flush=True)
        frame.to_csv(path, index=False)
        print("%s/%d done" % (name, i), flush=True)
    summarize()


def summarize():
    P = pd.read_csv(os.path.join(env.LOCAL, "ec3_WT0_all.csv"))
    files = sorted(os.listdir(CK))
    if not files:
        return
    X = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in files], ignore_index=True)
    X["vkey"] = X.validator.map({"DIAG10": "DIAG10", "P2LOO": "P2LOO", "EL1": "EL1"})
    # WT0 fold numbering: DIAG10 0..9, then P2LOO / EL1 appended in the same order as here (offset 10)
    G = X.merge(P, on=["row_id", "validator", "validation_fold"], suffixes=("", "_p"), how="left")
    assert G.et_47.notna().all(), "WT0 rows missing for some FX1 rows"
    r = lambda e: float(np.sqrt(np.mean(np.square(e))))
    for v in ("DIAG10", "P2LOO", "EL1"):
        for rows_name, extra in (("all", None), ("pass2", 179)):
            if v != "DIAG10" and rows_name == "all":
                continue
            for f in ("F13", "F47"):
                g = G[(G.validator == v) & (G.farm == f)]
                if extra is not None:
                    g = g[g.day >= extra]
                if g.empty:
                    continue
                y = g.sub_ec.values
                line = "%-6s %-5s %s n=%5d |" % (v, rows_name, f, len(g))
                seedP, means = {}, {}
                for c in ("P", "S", "I"):
                    preds = []
                    for s in SEEDS:
                        if c == "P":
                            p = r3(g["et_%d" % s], g["lgb_%d" % s], g["mlp_%d" % s], g.lo, g.hi)
                        else:
                            p = r3(g["%s_et_%d" % (c, s)], g["%s_lgb_%d" % (c, s)], g["%s_mlp_%d" % (c, s)], g.lo, g.hi)
                        preds.append(np.asarray(p, float))
                    seedP[c] = [r(p - y) for p in preds]; means[c] = np.mean(preds, axis=0)
                    line += " %s %.4f [%s]" % (c, r(means[c] - y), " ".join("%.4f" % x for x in seedP[c]))
                print(line)
                for c in ("S", "I"):
                    better = sum(a < b for a, b in zip(seedP[c], seedP["P"])); worse = sum(a > b for a, b in zip(seedP[c], seedP["P"]))
                    eP, eC = (means["P"] - y) ** 2, (means[c] - y) ** 2
                    pw = boot_p_worse(g, eP, eC)          # P(candidate not better than P)
                    pb = boot_p_worse(g, eC, eP)          # P(P not better than candidate)
                    d = r(means[c] - y) / r(means["P"] - y) - 1
                    verdict = "판별 불가"
                    if c == "S":
                        if worse == 3 and pb < .0125: verdict = "다른 온실 학습이 도움"
                        if better == 3 and pw < .0125: verdict = "다른 온실 학습이 오차를 만듦"
                    else:
                        if better == 3 and pw < .0125: verdict = "온실 표시가 도움"
                        if worse == 3 and pb < .0125: verdict = "온실 표시가 해로움"
                    print("    %s vs P: %+.2f%%  seeds better %d/3  P(%s not better)=%.4f  P(P not better)=%.4f  -> %s"
                          % (c, 100 * d, better, c, pw, pb, verdict))


if __name__ == "__main__":
    if STAGE == "sum":
        summarize()
    else:
        main()
