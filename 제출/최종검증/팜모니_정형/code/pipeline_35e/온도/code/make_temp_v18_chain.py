# -*- coding: utf-8 -*-
"""submission_18 temperature (2026-10-11 집 클로드, user: "온도는 사슬 보정 놓고 C18이랑 합쳐서 제출해보자").
pred = P9 (9th-round W40G-S, temp_candidate_v13.csv, LB .5127) + chain-aware early-hour correction (TC1c recipe, catalog
6.540 / 6.551: all 4 cells of the fresh layout L better, P .0297 -> near-miss; LEADERBOARD HYPOTHESIS TEST):
  For each record, among the previous 1-3 records of the same farm pick the one whose 23 h state best continues this record's
  0 h state (in_temp, in_hum, in_co2, act_heating, act_thermal, act_vent, act_circfan; mean |z-diff|).  Features: hour,
  in_temp, p, p - in_temp, lit1_in23, lit2_in23 (literal previous / second-previous existing record 23 h in_temp), ch_in23,
  ch_eve (chosen predecessor's 23 h / 18-23 h mean in_temp), ch_cost, jump = in_temp - ch_in23.
  Corrector: LightGBM (100 trees, lr .03, 15 leaves, min_child 100, subsample .8, colsample .8, seed 1) fitted on hours 0-7 of
  the stored DIAG10 out-of-fold W40G-S residuals (y - p, all 400 labelled days, members of tt1: BASE seeds 7 / 101 x TabPFN
  families A / B -> 4 correctors, corrections averaged).  Applied to evaluation rows of hours 0-7 with shrink 0.5.
Causality: training-row chain features in the MASK world (train_X inputs only); evaluation-row features from the same farm's
  inputs up to hour h of the row and earlier records (train_X + test_X records with smaller record number).  Only hour-0 values
  of the current record are used for the chain choice (h >= 0 always available).
Run:  python make_temp_v18_chain.py <data_dir> <oof_dir> <temp_candidate_v13.csv> <out.csv>
"""
import os, sys
import numpy as np, pandas as pd
import lightgbm as lgb

V = ["in_temp", "in_hum", "in_co2", "act_heating", "act_thermal", "act_vent", "act_circfan"]
FEAT = ["hour", "in_temp", "p", "p_air", "lit1_in23", "lit2_in23", "ch_in23", "ch_eve", "ch_cost", "jump"]
FARMS = ("F13", "F47")


def ident(df):
    df = df[df.row_id.str[:3].isin(FARMS)].copy()
    df["farm"], df["day"], df["hour"] = df.row_id.str[:3], df.row_id.str[4:7].astype(int), df.row_id.str[8:10].astype(int)
    return df


def chain_table(X, records, sd):
    """X: inputs available (MASK: train only; deploy: train + test).  records: all record ids (farm, day) in record order."""
    h0 = X[X.hour == 0].set_index(["farm", "day"])[V]; h23 = X[X.hour == 23].set_index(["farm", "day"])[V]
    eve = X[X.hour >= 18].groupby(["farm", "day"]).in_temp.mean()
    rows = []
    for f in FARMS:
        days = [d for ff, d in records if ff == f]
        for i, d in enumerate(days):
            prev = days[max(0, i - 3):i][::-1]
            lit = [h23.in_temp.get((f, p), np.nan) for p in prev[:2]] + [np.nan, np.nan]
            best, bc = None, np.inf
            if (f, d) in h0.index:
                for p in prev:
                    if (f, p) in h23.index:
                        c = np.nanmean(np.abs((h0.loc[(f, d)] - h23.loc[(f, p)]) / sd).values)
                        if np.isfinite(c) and c < bc:
                            best, bc = p, c
            rows.append(dict(farm=f, day=d, lit1_in23=lit[0], lit2_in23=lit[1],
                             ch_in23=h23.in_temp.get((f, best), np.nan) if best is not None else np.nan,
                             ch_eve=eve.get((f, best), np.nan) if best is not None else np.nan, ch_cost=bc if best is not None else np.nan))
    return pd.DataFrame(rows)


def w40(M, b, fam):
    g = np.where(M.in_temp.isna(), 1, np.clip((M.in_temp - 8) / 2, 0, 1))
    return (.4 * M[b] + (.2 + .4 * (1 - g)) * M.codex_REF + .4 * g * M["pfn_" + fam]).values


def main():
    data, oofdir, p9path, out = sys.argv[1:5]
    assert not os.path.exists(out), "refusing to overwrite " + out
    tX = ident(pd.read_csv(os.path.join(data, "train_X.csv"))); sX = ident(pd.read_csv(os.path.join(data, "test_X.csv")))
    records = sorted(set(zip(tX.farm, tX.day)) | set(zip(sX.farm, sX.day)))
    sd = tX[V].std()                                   # scale from training inputs only
    CT_mask = chain_table(tX, records, sd)             # training rows: MASK world
    CT_full = chain_table(pd.concat([tX, sX]), records, sd)   # evaluation rows: all earlier same-farm inputs
    D = pd.concat([pd.read_csv(os.path.join(oofdir, f)) for f in sorted(os.listdir(oofdir)) if f.startswith("DIAG10_")], ignore_index=True)
    D = D.merge(CT_mask, on=["farm", "day"], how="left"); D["jump"] = D.in_temp - D.ch_in23
    D = D[D.hour <= 7].reset_index(drop=True)
    P9 = pd.read_csv(p9path)[["row_id", "sub_temp"]].rename(columns={"sub_temp": "p"})
    Q = ident(sX[["row_id"] + V]).merge(P9, on="row_id").merge(CT_full, on=["farm", "day"], how="left")
    Q["jump"] = Q.in_temp - Q.ch_in23; Q["p_air"] = Q.p - Q.in_temp
    early = (Q.hour <= 7).values; corr = []
    for b in ("base_REF_7", "base_REF_101"):
        for fam in "AB":
            p = w40(D, b, fam); Dk = D.assign(p=p, p_air=p - D.in_temp.values)
            m = lgb.LGBMRegressor(n_estimators=100, learning_rate=.03, num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1,
                                  colsample_bytree=.8, random_state=1, verbose=-1, deterministic=True, force_col_wise=True, n_jobs=2)
            m.fit(Dk[FEAT], Dk.sub_temp.values - p); corr.append(m.predict(Q.loc[early, FEAT]))
    c = np.mean(corr, 0)
    Q["pred"] = Q.p.values; Q.loc[early, "pred"] = Q.p.values[early] + .5 * c
    S = P9[["row_id"]].merge(Q[["row_id", "pred"]], on="row_id", how="left")
    assert S.pred.notna().all() and len(S) == 1440
    S = S.rename(columns={"pred": "sub_temp"}); S.to_csv(out, index=False, float_format="%.6f")
    d = S.sub_temp.values - P9.p.values
    print("wrote %s | change vs P9: rows changed %d, RMS %.4f, mean %+.4f, min %+.3f, max %+.3f | chain predecessor found %.0f%% of eval records"
          % (out, int((np.abs(d) > 1e-9).sum()), np.sqrt((d ** 2).mean()), d.mean(), d.min(), d.max(), 100 * CT_full[CT_full.day.isin(sX.day)].ch_in23.notna().mean()))


if __name__ == "__main__":
    main()
