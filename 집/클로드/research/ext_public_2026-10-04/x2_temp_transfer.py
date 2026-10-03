# -*- coding: utf-8 -*-
"""X2 (diagnostic, fixed before running; 2026-10-04 집 클로드).
Transfer of substrate temperature from public 이레 data (18 strawberry farms, hourly
substrate temp + inside temp/hum) to F13/F47, using only the overlapping inputs.
Model: ExtraTrees on gap = substrate - inside temp; features: in_temp, in_hum, hour,
in_temp lags 1..6 h and 6/12/24-h trailing means WITHIN the same record/day
(causal; competition rule: same greenhouse current/previous inputs only).
Applied to F13/F47 DIAG10 rows -> T_x = in_temp + gap_hat.
Report: RMSE of T_x vs label; Spearman(T_x - W40G, label - W40G) on DIAG10/EXT10/EXT12;
fixed blend 0.9*W40G + 0.1*T_x (weight set now) per validator.
Reading: model test (full rule) only if the fixed blend improves all three validators."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import spearmanr
OUT = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\2ea435a4-3dc2-434b-ad01-aff27f988a60\scratchpad"
def feats(df, key, ti, hi, hour):
    df = df.sort_values(key + ["_t"]).copy(); g = df.groupby(key)
    F = pd.DataFrame({"ti": df[ti], "hi": df[hi], "hour": df[hour]}, index=df.index)
    for k in range(1, 7):
        F["ti_l%d" % k] = g[ti].shift(k)
    for w in (6, 12, 24):
        F["ti_m%d" % w] = g[ti].transform(lambda s: s.rolling(w, min_periods=1).mean())
    F["dti"] = F.ti - F.ti_l1
    return df, F
I = pd.read_csv(os.path.join(OUT, "x0_ire_hourly.csv"), parse_dates=["t"]).rename(columns={"내부온도": "ti", "내부습도": "hi", "배지온도": "st"})
I = I.dropna(subset=["ti", "hi", "st"]); I["_t"] = I.t; I["hour"] = I.t.dt.hour; I["day"] = I.t.dt.date
I, FI = feats(I, ["farm"], "ti", "hi", "hour")
m = ExtraTreesRegressor(400, min_samples_leaf=20, max_features=.7, random_state=0, n_jobs=4).fit(FI, I.st - I.ti)
T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv"))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_hum"])
T = T.merge(X, on="row_id", how="left")
T["_t"] = T.hour
T, FT = feats(T, ["validator", "fold", "farm", "day"], "in_temp", "in_hum", "hour")
FT.columns = FI.columns
ok = FT.ti.notna()
T["tx"] = np.nan; T.loc[ok, "tx"] = T.loc[ok, "in_temp"] + m.predict(FT[ok].fillna(FT[ok].median()))
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
T["w40"] = 0.4 * (T.mask_base_7 + T.mask_base_101) / 2 + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn
T["bl"] = np.where(T.tx.notna(), 0.9 * T.w40 + 0.1 * T.tx, T.w40)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
allok = True
for v, G in T.groupby("validator"):
    G = G.dropna(subset=["sub_temp", "w40"])
    H = G.dropna(subset=["tx"])
    rho = spearmanr(H.tx - H.w40, H.sub_temp - H.w40).correlation
    a, b = r(G.w40 - G.sub_temp), r(G.bl - G.sub_temp)
    allok &= b < a
    print("%-6s n %5d | T_x RMSE %.3f (in_temp only %.3f) | W40G %.4f -> blend .1 %.4f (%+.2f%%) | rho(T_x-W40G, resid) %+.2f" % (
        v, len(G), r(H.tx - H.sub_temp), r(H.in_temp - H.sub_temp), a, b, 100 * (b / a - 1), rho))
print("\nX2 worth a full model test:", allok)
