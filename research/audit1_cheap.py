# -*- coding: utf-8 -*-
"""Audit 1 (methodology), cheap checks on saved artefacts.

1. Independent re-implementation of the paired greenhouse-day block bootstrap
   and comparison with feat_lib.paired_block_boot; alternative resampling
   units (calendar date cluster, 5/10-day super-blocks).
2. Fold geometry: overlap inside A, Jaccard A/B, how the pooled OOF handles
   overlapping folds, same-date twins left in training.
3. Q4 "noisy day" classifier: reproduce, test days above the Q4 threshold,
   stability of Q4 membership across seeds / fold assignments.
4. 6b.3: date share of day-level offset variance vs a permutation null, ICC.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import common
from common import split_mask
from harness import load, folds, SHIFTS_A, SHIFTS_B
from feat_lib import paired_block_boot
from cleanw_v6 import weights
from anal_q3d_adversarial import shape_features
from anal_q3e_matched import SHAPE_ONLY
from anal_q1_errors import diag_folds


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def boot_units(y, a, b, unit, n=4000, seed=1):
    """Paired cluster bootstrap on per-unit sufficient statistics."""
    u, inv = np.unique(unit, return_inverse=True)
    sa = np.bincount(inv, (y - a) ** 2)
    sb = np.bincount(inv, (y - b) ** 2)
    nn = np.bincount(inv).astype(float)
    rng = np.random.RandomState(seed)
    k = len(u)
    cnt = np.stack([np.bincount(rng.randint(0, k, k), minlength=k) for _ in range(n)])
    d = np.sqrt(cnt @ sb / (cnt @ nn)) - np.sqrt(cnt @ sa / (cnt @ nn))
    pt = np.sqrt(sb.sum() / nn.sum()) - np.sqrt(sa.sum() / nn.sum())
    return pt, np.percentile(d, 2.5), np.percentile(d, 97.5), float((d > 0).mean()), k


def main():
    panel, lab, _ = load()
    y = lab.sub_temp.values
    z = np.load(env.LOCAL + "/oof_temp_r3.npz", allow_pickle=True)
    assert (z["row_id"] == lab.row_id.values).all()
    cal = pd.read_csv(env.LOCAL + "/deep_cal_11_days.csv")[["farm", "day", "cal", "is_test"]]
    lk = lab[["farm", "day"]].merge(cal, on=["farm", "day"], how="left")
    print("rows without calendar cluster: %d" % lk.cal.isna().sum())

    print("\n== 1. bootstrap ==")
    for s in ("EXT10", "geomA", "geomB"):
        a, b = z[s], z[s + "_w02"]
        g = ~np.isnan(a) & ~np.isnan(b)
        print("%s: RMSE plain %.4f w02 %.4f (n=%d)" % (s, rmse(a[g], y[g]), rmse(b[g], y[g]), g.sum()))
        sub = lab[g].reset_index(drop=True)
        r = paired_block_boot(sub, "sub_temp", a[g], b[g], n_boot=2000, seed=0)
        print("   feat_lib   : %+.4f [%+.4f,%+.4f] Pw %.3f" % r)
        fd = (lab.farm + "_" + lab.day.astype(str)).values[g]
        units = {
            "farm-day": fd,
            "calendar date (both farms)": np.where(lk.cal.isna(), -1 - np.arange(len(lk)), lk.cal).astype(float)[g],
            "5-day superblock": (lab.farm + "_" + (lab.day // 5).astype(str)).values[g],
            "10-day superblock": (lab.farm + "_" + (lab.day // 10).astype(str)).values[g],
            "15-day both farms": (lab.day // 15).values[g],
        }
        for nm, u in units.items():
            pt, lo, hi, pw, k = boot_units(y[g], a[g], b[g], u)
            print("   %-27s k=%4d %+.4f [%+.4f,%+.4f] width %.4f Pw %.3f" % (nm, k, pt, lo, hi, hi - lo, pw))

    print("\n== 2. fold geometry ==")
    for kind, sh in (("A", SHIFTS_A), ("B", SHIFTS_B)):
        fds = folds(kind)
        cnt = {}
        for fd in fds:
            _, vam = split_mask(lab, fd)
            for k in set(zip(lab.farm.values[vam], lab.day.values[vam])):
                cnt[k] = cnt.get(k, 0) + 1
        c = pd.Series(cnt)
        print("%s: labelled days held out %d, held out >1 times %d (%.0f%%), max %d; pooled OOF keeps the LAST fold"
              % (kind, len(c), int((c > 1).sum()), 100 * (c > 1).mean(), c.max()))
        # train/val relation per fold: held-out days whose calendar twin stays in training
        tw = []
        for fd in fds:
            trm, vam = split_mask(lab, fd)
            trcal = set(lk.cal[trm].dropna())
            vd = lk[vam].drop_duplicates(["farm", "day"])
            tw.append(vd.cal.isin(trcal).mean())
        print("   held-out days with a same-calendar-date day left in training: %.0f%%" % (100 * np.mean(tw)))
    sa = set()
    sb = set()
    for kind, S in (("A", sa), ("B", sb)):
        for fd in folds(kind):
            _, vam = split_mask(lab, fd)
            S |= set(zip(lab.farm.values[vam], lab.day.values[vam]))
    print("Jaccard(A union, B union) = %.3f" % (len(sa & sb) / len(sa | sb)))
    # test days: calendar twins among labelled days
    ct = cal[cal.is_test]
    labcal = set(lk.cal.dropna())
    print("test days with a same-calendar-date labelled day: %.0f%% (of %d)" % (100 * ct.cal.isin(labcal).mean(), len(ct)))
    dmin = pd.concat([lab[["farm", "day"]]], axis=1)

    print("\n== 3. Q4 noisy-day classifier ==")
    tX, ty, sX = common.load_raw()
    a = pd.concat([tX.assign(split="train"), sX.assign(split="test")], ignore_index=True)
    a = a.query("farm in ['F13','F47']").reset_index(drop=True)
    F = shape_features(a)
    a = a.loc[F.index].reset_index(drop=True)
    F = F.reset_index(drop=True)[SHAPE_ONLY]
    yy = (a.split == "train").astype(int).values
    groups = (a.farm + "_" + a.day.astype(str)).values
    bins = np.floor(a.in_temp.clip(-2, 32)).values
    te_h = pd.Series(bins[yy == 0]).value_counts(normalize=True)
    tr_h = pd.Series(bins[yy == 1]).value_counts(normalize=True)
    sw = np.where(yy == 1, pd.Series(bins).map(te_h / tr_h).fillna(0).values, 1.0)
    sw = np.where(yy == 1, sw * (yy == 0).sum() / sw[yy == 1].sum(), sw)
    ug = np.unique(groups)
    ds = pd.read_csv(env.LOCAL + "/anal_q3f_dayscore.csv")
    ref_q4 = set(map(tuple, ds[ds.q == "Q4 train-like"][["farm", "day"]].values))
    from sklearn.model_selection import GroupKFold
    runs = []
    for rep in range(6):
        p = np.zeros(len(a))
        if rep == 0:
            splits = list(GroupKFold(5).split(F, yy, groups))
        else:
            rs = np.random.RandomState(rep)
            fa = dict(zip(ug, rs.randint(0, 5, len(ug))))
            fid = np.array([fa[g] for g in groups])
            splits = [(np.where(fid != k)[0], np.where(fid == k)[0]) for k in range(5)]
        for tr, te in splits:
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                   n_jobs=2, random_state=rep)
            p[te] = m.fit(F.iloc[tr], yy[tr], sample_weight=sw[tr]).predict_proba(F.iloc[te])[:, 1]
        a["p"] = p
        dd = a.groupby(["farm", "day", "split"]).p.agg(["mean", "size"]).reset_index()
        tr_d = dd[(dd.split == "train")].merge(ds[["farm", "day"]], on=["farm", "day"])
        thr = tr_d["mean"].quantile(0.75)
        q4 = set(map(tuple, tr_d[tr_d["mean"] > thr][["farm", "day"]].values))
        te_d = dd[dd.split == "test"]
        auc = roc_auc_score(yy, p, sample_weight=sw)
        jac = len(q4 & ref_q4) / len(q4 | ref_q4)
        runs.append(q4)
        print("rep %d: AUC %.3f | test days above train-Q4 threshold: %d/%d (%.0f%%) | test-day mean p %.3f vs train %.3f"
              " | Jaccard with saved Q4 %.2f"
              % (rep, auc, int((te_d["mean"] > thr).sum()), len(te_d), 100 * (te_d["mean"] > thr).mean(),
                 te_d["mean"].mean(), tr_d["mean"].mean(), jac))
    allq = set.union(*runs)
    core = set.intersection(*runs)
    print("Q4 across 6 runs: union %d days, in all 6: %d, saved Q4 size %d" % (len(allq), len(core), len(ref_q4)))

    print("\n== 4. 6b.3 date share of day-level offset ==")
    zd = np.load(env.LOCAL + "/oof_temp_diag.npz", allow_pickle=True)
    clean = weights(lab, 3, 0.0) >= 1
    e = zd["r3"] - y
    g = ~np.isnan(e) & clean
    dres = (pd.DataFrame({"farm": lab.farm.values[g], "day": lab.day.values[g], "e": e[g]})
              .groupby(["farm", "day"]).e.agg(res="mean", n="size").reset_index())
    dres = dres[dres.n >= 12].merge(cal[["farm", "day", "cal"]], on=["farm", "day"], how="left")
    D = dres.dropna(subset=["cal"])
    sz = D.groupby("cal").res.transform("size")
    V = D[sz >= 2].copy()

    def share(df, col):
        m = df.groupby(col).res.transform("mean")
        return 1 - ((df.res - m) ** 2).sum() / ((df.res - df.res.mean()) ** 2).sum()
    s0 = share(V, "cal")
    rng = np.random.RandomState(0)
    null = []
    for _ in range(2000):
        W = V.copy()
        W["perm"] = rng.permutation(W.cal.values)
        null.append(share(W, "perm"))
    null = np.array(null)
    G = V.cal.nunique()
    N = len(V)
    print("days %d in %d dates (mean size %.2f): share %.3f | permutation null mean %.3f, 95%% %.3f | p=%.3f | (G-1)/(N-1)=%.3f"
          % (N, G, N / G, s0, null.mean(), np.percentile(null, 95), (null >= s0).mean(), (G - 1) / (N - 1)))
    # one-way ANOVA ICC
    grp = V.groupby("cal").res
    k = grp.size()
    msb = (k * (grp.mean() - V.res.mean()) ** 2).sum() / (G - 1)
    msw = ((V.res - grp.transform("mean")) ** 2).sum() / (N - G)
    k0 = (N - (k ** 2).sum() / N) / (G - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    print("ANOVA ICC(1) of day offset within calendar date: %.3f (MSB %.3f, MSW %.3f, k0 %.2f)" % (icc, msb, msw, k0))
    # pairwise same-date different-farm
    pairs = []
    for c, gg in V.groupby("cal"):
        f13 = gg[gg.farm == "F13"].res.values
        f47 = gg[gg.farm == "F47"].res.values
        for x in f13:
            for w in f47:
                pairs.append((x, w))
    pairs = np.array(pairs)
    print("same date, F13 x F47 pairs: n=%d, r=%.3f" % (len(pairs), np.corrcoef(pairs.T)[0, 1]))
    pairs = []
    for c, gg in V.groupby("cal"):
        for f, h in gg.groupby("farm"):
            r = h.res.values
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    pairs.append((r[i], r[j]))
    pairs = np.array(pairs)
    print("same date, same farm pairs: n=%d, r=%.3f" % (len(pairs), np.corrcoef(pairs.T)[0, 1]))


if __name__ == "__main__":
    main()
