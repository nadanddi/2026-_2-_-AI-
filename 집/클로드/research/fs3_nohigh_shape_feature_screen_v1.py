# -*- coding: utf-8 -*-
"""FS3: within-day SHAPE features in the NO-HIGH world (26 high-EC days removed entirely), fast screen.
(2026-10-09 집 클로드, user: "하루 안 모양도 예측하는 특징도 찾아봐. 변수나 생육 특성 잘 활용해서").  Fixed before running.
Hypotheses are GENERAL KNOWLEDGE (not from read sources): measured EC follows substrate temperature, which lags air
temperature by hours; daytime transpiration concentrates the substrate.  Exploratory check (sh1 log): the model
explains 15 % of within-day log-EC shape; slow in_temp EWM change since 0 h correlates +.46-.48 with the shape.
All features causal: same farm, same day, hours 0..h.
Families added to REF (= current EC features + season + DP1):
  THERM   in_temp EWM (half-life 2, 4, 8 h) minus in_temp at 0 h; EWM level (half-life 4, 8 h)
  TRANSP  out_rad cumulative since 0 h, VPD x out_rad cumulative, hours with out_rad > 0 so far, VPD now
  SHAPE   THERM + TRANSP
Screen model: LightGBM tweedie (core.lg, seed 3131), house DIAG10 folds minus high days, exclusions as FS1.
Metrics: all-rows RMSE and within-day SHAPE RMSE (residual minus its day mean, log scale).
Screen rule (vs REF): all-rows RMSE better by >= 1 % AND >= 7/10 folds better -> stage 2; SHAPE RMSE reported.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u fs3_nohigh_shape_feature_screen_v1.py
"""
import env  # noqa: F401
import importlib.util, os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("wt0", os.path.join(HERE, "ec3_WT0_r3_member_weights_v1.py"))
wt0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt0)
dp1, dc4, p3, core = wt0.dp1, wt0.dc4, wt0.p3, wt0.core
OTHER = {"F13": "F47", "F47": "F13"}
SEED = 3131
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def shape_features(full):
    a = full.sort_values(["farm", "day", "hour"]).reset_index(drop=True).copy()
    g = a.groupby(["farm", "day"], sort=False)
    F = pd.DataFrame({"row_id": a.row_id}); fam = {"THERM": [], "TRANSP": []}

    def add(f, n, v):
        F[n] = np.asarray(v, float); fam[f].append(n)
    t0 = g.in_temp.transform("first")
    for hl in (2, 4, 8):
        ew = g.in_temp.transform(lambda s, hl=hl: s.ewm(halflife=hl, ignore_na=True).mean())
        add("THERM", "Tewm%d_d_h0" % hl, ew - t0)
        if hl in (4, 8):
            add("THERM", "Tewm%d" % hl, ew)
    rad = a.out_rad.fillna(0)
    es = 0.6108 * np.exp(17.27 * a.in_temp / (a.in_temp + 237.3)); vpd = es * (1 - a.in_hum / 100)
    a["_rad"] = rad; a["_tr"] = (vpd * rad).fillna(0); a["_lit"] = (rad > 0).astype(float)
    g = a.groupby(["farm", "day"], sort=False)
    add("TRANSP", "rad_cum", g["_rad"].cumsum()); add("TRANSP", "transp_cum", g["_tr"].cumsum())
    add("TRANSP", "lit_hours", g["_lit"].cumsum()); add("TRANSP", "vpd_now2", vpd)
    fam["SHAPE"] = fam["THERM"] + fam["TRANSP"]
    return F, fam


def main():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    high = {(f, int(d)) for f, d in json.load(open(os.path.join(env.LOCAL, "hx1_day_sets.json"), encoding="utf-8"))["HIGH"]}
    NF, fam = shape_features(full)
    lab = lab[[(f, int(d)) not in high for f, d in zip(lab.farm, lab.day)]].copy()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    lab = lab.merge(NF, on="row_id", how="left", validate="one_to_one").set_index(lab.index)
    wv = dc4.weather_vectors(full)
    REF = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    configs = {"REF": REF, **{k: REF + v for k, v in fam.items()}}
    rows = []
    for name, i, vd in [x for x in fds if x[0] == "DIAG10"]:
        vd = {(f, d) for f, d in vd if (f, d) not in high}
        va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
        forb = ({(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)}
                | {(f, d + j) for f, d in lock for j in (-1, 0, 1)})
        tr_m = np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])
        tr, va = lab[tr_m].copy(), lab[va_m].copy()
        tdays = tr[["farm", "day"]].drop_duplicates(); vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
        season, vq = dc4.season_index(tdays, vdays, wv)
        tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]; vdays["season"] = vq
        va = va.merge(vdays, on=["farm", "day"], how="left").set_index(va.index)
        out = va[["farm", "day", "hour", "sub_ec"]].copy(); out["fold"] = i
        for c, cols in configs.items():
            out[c] = np.clip(core.shrink(core.predict_model(core.lg(SEED, "tweedie"), tr, va, cols), va), tr.sub_ec.min(), tr.sub_ec.max())
        rows.append(out); print("DIAG10/%d done" % i, flush=True)
    O = pd.concat(rows, ignore_index=True); O.to_csv(os.path.join(env.LOCAL, "fs3_screen_oof.csv"), index=False)
    y = O.sub_ec.to_numpy(float)

    def shape_rmse(col, m=slice(None)):
        d = O.loc[m].copy(); d["e"] = np.log(d[col]) - np.log(d.sub_ec)
        return r(d.e - d.groupby(["farm", "day"]).e.transform("mean"))
    ref = r(O.REF - y); p2 = (O.day >= 179).values
    print("\nREF all %.4f | pass-2 %.4f | SHAPE %.4f (pass-2 %.4f)" % (ref, r(O.REF[p2] - y[p2]), shape_rmse("REF"), shape_rmse("REF", p2)))
    passed = []
    for c in fam:
        d = r(O[c] - y) / ref - 1
        folds = sum(r(g[c] - g.sub_ec) < r(g.REF - g.sub_ec) for _, g in O.groupby("fold"))
        ok = d <= -.01 and folds >= 7
        passed += [c] if ok else []
        print("%-6s all %+.2f%% folds %d/10 | pass-2 %+.2f%% | SHAPE %+.2f%% (pass-2 %+.2f%%) -> %s" % (
            c, 100 * d, folds, 100 * (r(O[c][p2] - y[p2]) / r(O.REF[p2] - y[p2]) - 1),
            100 * (shape_rmse(c) / shape_rmse("REF") - 1), 100 * (shape_rmse(c, p2) / shape_rmse("REF", p2) - 1),
            "PASS to stage 2" if ok else "drop"))
    json.dump({"passed": passed, "families": fam}, open(os.path.join(env.LOCAL, "fs3_screen_result.json"), "w", encoding="utf-8"))
    print("passed:", passed)


if __name__ == "__main__":
    main()
