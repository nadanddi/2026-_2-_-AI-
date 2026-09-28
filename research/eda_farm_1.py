# -*- coding: utf-8 -*-
"""Per-greenhouse coverage of cold regime + first-order response fit sub_temp ~ a + b*ewm(in_temp, hl)."""
import env  # noqa
import numpy as np, pandas as pd
import common

tX, ty, sX = common.load_raw()
lab = tX.merge(ty[["row_id", "sub_temp", "sub_ec"]], on="row_id")
allx = pd.concat([tX.assign(is_test=False), sX.assign(is_test=True)], ignore_index=True)
allx = allx.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")

HLS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48]
rows = []
panels = {}
for f, g in allx.groupby("farm"):
    g = g.sort_values("t").set_index("t")
    full = np.arange(g.index.min(), g.index.max() + 1)
    p = g.reindex(full)
    p["farm"] = f
    for hl in HLS:
        p["ew%d" % hl] = p.in_temp.ewm(halflife=hl, ignore_na=True).mean()
    panels[f] = p
    L = p[p.sub_temp.notna() & (p.is_test != True)]
    y = L.sub_temp.values
    best = None
    for hl in HLS:
        x = L["ew%d" % hl].values
        m = np.isfinite(x)
        A = np.c_[np.ones(m.sum()), x[m]]
        c, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        r = y[m] - A @ c
        s = np.sqrt(np.mean(r ** 2))
        if best is None or s < best[0]:
            best = (s, hl, c[0], c[1])
    # 2-kernel fit: in_temp ew-fast + ew-slow
    x2 = L[["ew2", "ew24"]].values
    m = np.isfinite(x2).all(1)
    A = np.c_[np.ones(m.sum()), x2[m]]
    c2, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    s2 = np.sqrt(np.mean((y[m] - A @ c2) ** 2))
    frac_int = np.mean(np.isclose(y, np.round(y)))
    rows.append(dict(farm=f, n_lab=len(L), n_test=int((p.is_test == True).sum()),
                     days=int(L.day.nunique()), day_min=int(L.day.min()), day_max=int(L.day.max()),
                     int_frac=round(frac_int, 3),
                     in_mean=L.in_temp.mean(), sub_mean=L.sub_temp.mean(), sub_std=L.sub_temp.std(),
                     n_cold_ew6=int((L.ew6 < 9.72).sum()), n_cold_in=int((L.in_temp < 7.2).sum()),
                     n_cold_ew6_12=int((L.ew6 < 12).sum()),
                     best_hl=best[1], a=best[2], b=best[3], rmse1=best[0],
                     c2_fast=c2[1], c2_slow=c2[2], c2_a=c2[0], rmse2=s2,
                     has_rad=int(p.in_rad.notna().any()) if "in_rad" in p else 0))
R = pd.DataFrame(rows).round(3)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100); pd.set_option("display.max_columns", 40)
print(R.to_string())
R.to_csv(env.LOCAL + "/eda_farm_1_perfarm.csv", index=False)
pd.to_pickle(panels, env.LOCAL + "/eda_farm_panels.pkl")
# test rows stats
T = pd.concat([panels[f][panels[f].is_test == True] for f in ["F13", "F47"]])
print("test ew6<9.72: %.3f  in<7.2: %.3f  ew6 q01,q05,q50: %s" % ((T.ew6 < 9.72).mean(), (T.in_temp < 7.2).mean(), np.nanpercentile(T.ew6, [1, 5, 50]).round(2)))
for f in ["F13", "F47"]:
    Tf = panels[f][panels[f].is_test == True]
    print(f, "test ew6 pct:", np.nanpercentile(Tf.ew6, [1, 5, 25, 50]).round(2), "min in_temp", Tf.in_temp.min())
