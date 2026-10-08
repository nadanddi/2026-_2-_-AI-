# -*- coding: utf-8 -*-
"""FS1: feature discovery in the NO-HIGH world (the 26 high-EC days, day-mean EC >= 1.2, removed from the data
entirely, as HX1b).  Stage 1 = fast screen.  (2026-10-09 집 클로드, user: "고EC날을 빼고 데이터를 새로 만들어 낸 다음,
모델을 새로 만들어봐. 특징도 새로 발견해보고").  Fixed before running.

All new features are causal: same farm, same day, hours 0..h of that day (previous-day family: same farm, day d-1,
all 24 hours, which are past inputs).  Computed from train_X inputs only.
Reference feature set REF = current EC features (FULL without day) + season (DC4) + DP1.
Families (each added alone to REF):
  OUT   outside weather: out_temp/out_hum/out_rad/out_wspd current, day-to-date mean, value at 0 h; radiation sum to h
  INDYN indoor dynamics: in_temp/in_hum/in_co2 day-to-date min/max/mean, change since 0 h, change over 1 h and 3 h
  PHYS  physics combos: VPD (from in_temp/in_hum) current and day-to-date mean, in-out temperature difference current
        and day-to-date mean, closed-hours-to-date (vent==0 and circfan==0), closed-hours x radiation-to-date
  CO2D  CO2 dynamics: drop from the day-to-date max, day-to-date |hour-to-hour change| mean, change over 1 h
  PREV  previous day (d-1, same farm) 24 h means of in_temp/in_hum/in_co2/out_temp/out_rad and act means (control)
Screen model: LightGBM tweedie (core.lg, seed 3131) on REF (+ family); folds = house DIAG10 minus high days;
exclusions same farm +-1, lock-40 +-1, other farm d-3..d+3.
Screen rule (per family, vs REF, all rows of the 334 days): RMSE improvement >= 1 % AND >= 7/10 folds improved
-> family goes to stage 2.  Pass-2 rows reported (descriptive).  No adoption here.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u fs1_nohigh_feature_screen_v1.py
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
OUTV = ["out_temp", "out_hum", "out_rad", "out_wspd"]; INV = ["in_temp", "in_hum", "in_co2"]
ACTV = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
r = lambda e: float(np.sqrt(np.mean(np.square(e))))


def new_features(full):
    a = full.sort_values(["farm", "day", "hour"]).reset_index(drop=True).copy()
    g = a.groupby(["farm", "day"], sort=False)
    F = pd.DataFrame({"row_id": a.row_id})
    fam = {k: [] for k in ("OUT", "INDYN", "PHYS", "CO2D", "PREV")}

    def add(f, name, val):
        F[name] = np.asarray(val, float); fam[f].append(name)
    for v in OUTV:
        add("OUT", v + "_now", a[v]); add("OUT", v + "_tdmean", g[v].transform(lambda s: s.expanding().mean()))
        add("OUT", v + "_h0", g[v].transform("first"))
    add("OUT", "out_rad_tdsum", g["out_rad"].transform(lambda s: s.fillna(0).cumsum()))
    for v in INV:
        add("INDYN", v + "_tdmin", g[v].transform(lambda s: s.expanding().min()))
        add("INDYN", v + "_tdmax", g[v].transform(lambda s: s.expanding().max()))
        add("INDYN", v + "_tdmean", g[v].transform(lambda s: s.expanding().mean()))
        add("INDYN", v + "_d_h0", a[v] - g[v].transform("first"))
        add("INDYN", v + "_d1", g[v].diff(1)); add("INDYN", v + "_d3", g[v].diff(3))
    es = 0.6108 * np.exp(17.27 * a.in_temp / (a.in_temp + 237.3)); vpd = es * (1 - a.in_hum / 100)
    a["_vpd"] = vpd; a["_dT"] = a.in_temp - a.out_temp
    a["_closed"] = ((a.act_vent.fillna(0) == 0) & (a.act_circfan.fillna(0) == 0)).astype(float)
    g = a.groupby(["farm", "day"], sort=False)
    add("PHYS", "vpd_now", vpd); add("PHYS", "vpd_tdmean", g["_vpd"].transform(lambda s: s.expanding().mean()))
    add("PHYS", "dT_now", a._dT); add("PHYS", "dT_tdmean", g["_dT"].transform(lambda s: s.expanding().mean()))
    ch = g["_closed"].cumsum(); add("PHYS", "closed_hours_td", ch)
    add("PHYS", "closed_x_rad_td", ch * g["out_rad"].transform(lambda s: s.fillna(0).cumsum()))
    add("CO2D", "co2_drop_from_max", g["in_co2"].transform(lambda s: s.expanding().max()) - a.in_co2)
    a["_absd"] = g["in_co2"].diff(1).abs(); g = a.groupby(["farm", "day"], sort=False)
    add("CO2D", "co2_absd_tdmean", g["_absd"].transform(lambda s: s.expanding().mean()))
    add("CO2D", "co2_d1", g["in_co2"].diff(1))
    D = a.groupby(["farm", "day"])[INV + ["out_temp", "out_rad"] + ACTV].mean()
    prev = D.copy(); prev.index = pd.MultiIndex.from_arrays([prev.index.get_level_values(0), prev.index.get_level_values(1) + 1])
    P = prev.reindex(pd.MultiIndex.from_arrays([a.farm, a.day]))
    for c in P.columns:
        add("PREV", "prev_" + c, P[c].values)
    return F, fam


def main():
    raw, full, lab, lock, signatures, fds = p3.prepare()
    high = {(f, int(d)) for f, d in json.load(open(os.path.join(env.LOCAL, "hx1_day_sets.json"), encoding="utf-8"))["HIGH"]}
    assert len(high) == 26
    NF, fam = new_features(full)
    lab = lab[[(f, int(d)) not in high for f, d in zip(lab.farm, lab.day)]].copy()
    lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
    lab = lab.merge(NF, on="row_id", how="left", validate="one_to_one").set_index(lab.index)
    assert lab[["farm", "day"]].drop_duplicates().shape[0] == 334
    wv = dc4.weather_vectors(full)
    REF = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
    configs = {"REF": REF, **{k: REF + v for k, v in fam.items()}}
    print({k: len(v) for k, v in configs.items()}, flush=True)
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
        rows.append(out)
        print("DIAG10/%d done" % i, flush=True)
    O = pd.concat(rows, ignore_index=True); O.to_csv(os.path.join(env.LOCAL, "fs1_screen_oof.csv"), index=False)
    y = O.sub_ec.to_numpy(float); ref = r(O.REF - y)
    print("\nREF all-rows RMSE %.4f | pass-2 rows %.4f" % (ref, r(O.REF[O.day >= 179] - y[O.day >= 179])))
    passed = []
    for c in fam:
        d = r(O[c] - y) / ref - 1
        folds = sum(r(g[c] - g.sub_ec) < r(g.REF - g.sub_ec) for _, g in O.groupby("fold"))
        p2 = O.day >= 179; d2 = r(O[c][p2] - y[p2]) / r(O.REF[p2] - y[p2]) - 1
        ok = d <= -.01 and folds >= 7
        passed += [c] if ok else []
        print("%-6s all %+.2f%%  folds better %d/10  | pass-2 %+.2f%%  -> %s" % (c, 100 * d, folds, 100 * d2, "PASS to stage 2" if ok else "drop"))
    json.dump({"passed": passed, "families": fam}, open(os.path.join(env.LOCAL, "fs1_screen_result.json"), "w", encoding="utf-8"))
    print("passed:", passed)


if __name__ == "__main__":
    main()
