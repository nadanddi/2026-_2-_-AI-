# -*- coding: utf-8 -*-
"""Analysis Q3d: are there restored training rows beyond the 167 rule-flagged ones?

Findings so far: the 167 input-flagged rows (restored indoor temperature, too
low for their label) distort the tree/Ridge members in the cold input region
where the test lives (anal_q3c_members.py).  Rules catch only blatant cases.

Test inputs carry no restoration (problem statement, section 4), so a row
SHAPE that occurs in training but never in test is a restoration candidate.
Adversarial validation: a classifier separates F13/F47 training rows from
test rows using only local shape descriptors -- no levels, no time position,
so the colder season of the test cannot be used:
  first / second differences and their absolute size, constant-run length,
  hour-to-hour humidity vs temperature consistency (absolute humidity change),
  night indoor-outdoor gap, CO2 change vs CO2 supply / ventilation.
Grouped 5-fold by greenhouse-day.  AUC near 0.5 = nothing left to find.

Then: do the most train-like rows behave like the flagged ones (slab - air
gap shifted, larger residual in the diagnostic OOF)?

Analysis only.

Run:  cd research && PYTHONPATH="" <python> anal_q3d_adversarial.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import common
from harness import load
from cleanw_v6 import weights


def abs_hum(t, rh):
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    return 216.7 * (rh / 100.0 * es) / (273.15 + t)


def shape_features(a):
    a = a.sort_values(["farm", "t"]).copy()
    g = a.groupby("farm")
    f = pd.DataFrame(index=a.index)
    for c in ("in_temp", "in_hum", "in_co2"):
        d1 = g[c].diff()
        f[c + "_d1"] = d1
        f[c + "_ad1"] = d1.abs()
        f[c + "_d2"] = g[c].diff().groupby(a.farm).diff()
        f[c + "_ad1_next"] = (g[c].shift(-1) - a[c]).abs()
        same = (d1 == 0).astype(int)
        f[c + "_run"] = same.groupby((same == 0).cumsum()).cumsum()
    ah = abs_hum(a.in_temp, a.in_hum)
    f["abshum_d1"] = ah.groupby(a.farm).diff()
    f["rh_vs_temp"] = f["in_hum_d1"] * np.sign(f["in_temp_d1"].replace(0, np.nan))
    night = ~a.hour.between(7, 18)
    f["night_in_out_gap"] = (a.in_temp - a.out_temp).where(night)
    f["co2_d1_no_supply"] = f["in_co2_d1"].where((a.act_co2.fillna(0) == 0) & (a.act_vent.fillna(0) == 0))
    f["is_night"] = night.astype(float)
    return f


def main():
    tX, ty, sX = common.load_raw()
    tX = tX.assign(split="train")
    sX = sX.assign(split="test")
    a = pd.concat([tX, sX], ignore_index=True).query("farm in ['F13','F47']").reset_index(drop=True)
    F = shape_features(a)
    a = a.loc[F.index]
    y = (a.split == "train").astype(int).values
    groups = (a.farm + "_" + a.day.astype(str)).values
    p = np.zeros(len(a))
    for tr, te in GroupKFold(5).split(F, y, groups):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               min_child_samples=50, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, verbose=-1, n_jobs=2, random_state=0,
                               is_unbalance=True)
        p[te] = m.fit(F.iloc[tr], y[tr]).predict_proba(F.iloc[te])[:, 1]
    auc = roc_auc_score(y, p)
    print("train-vs-test from shape only: AUC %.3f  (0.5 = indistinguishable)" % auc)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                           verbose=-1, n_jobs=2, random_state=0, is_unbalance=True).fit(F, y)
    imp = pd.Series(m.feature_importances_, index=F.columns).sort_values(ascending=False)
    print("top shape features:", ", ".join("%s %d" % (k, v) for k, v in imp.head(8).items()))

    # compare suspects with the rule flags and the diagnostic residual
    panel, lab, _ = load()
    z = np.load(env.LOCAL + "/oof_temp_diag.npz", allow_pickle=True)
    res = pd.Series(z["r3"] - lab.sub_temp.values, index=lab.row_id.values)
    flags = pd.read_csv(env.LOCAL + "/eda_forensic_11_flags.csv").set_index("row_id").Vany
    tr_rows = a[a.split == "train"].assign(p_train=p[y == 1])
    test_p = p[y == 0]
    thr = np.quantile(test_p, 0.995)
    tr_rows["suspect"] = tr_rows.p_train > thr
    tr_rows["flag"] = tr_rows.row_id.map(flags).fillna(False).astype(bool)
    tr_rows["res"] = tr_rows.row_id.map(res)
    tr_rows = tr_rows.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
    lab_ewm = lab.set_index("row_id")
    print("\nthreshold = 99.5th percentile of test rows' train-likeness")
    print("suspect training rows: %d (%.1f%%) | of which rule-flagged: %d | rule-flagged total: %d"
          % (int(tr_rows.suspect.sum()), 100 * tr_rows.suspect.mean(),
             int((tr_rows.suspect & tr_rows.flag).sum()), int(tr_rows.flag.sum())))
    for nm, m_ in (("suspect only (new)", tr_rows.suspect & ~tr_rows.flag),
                   ("rule-flagged", tr_rows.flag),
                   ("neither", ~tr_rows.suspect & ~tr_rows.flag)):
        s = tr_rows[m_]
        gap = (s.sub_temp - s.in_temp)
        print("  %-20s n=%5d | slab - air %.2f | |diag residual| %.3f | in_temp mean %.2f"
              % (nm, len(s), gap.mean(), s.res.abs().mean(), s.in_temp.mean()))
    tr_rows[["row_id", "p_train", "suspect", "flag"]].to_csv(env.LOCAL + "/anal_q3d_suspects.csv", index=False)
    print("saved local/anal_q3d_suspects.csv")


if __name__ == "__main__":
    main()
