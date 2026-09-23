# -*- coding: utf-8 -*-
"""Shared data loading, causal feature engineering and CV splitting.

Design constraints derived from the problem statement (PDF) and the EDA:
  * test_X contains only F13/F47 -> the evaluation model is farm-specific.
  * in_rad / act_side / act_valve / act_cool / act_pump are 100% missing in
    test -> dropped everywhere.
  * Section 5: a row may only use inputs of the SAME greenhouse at the current
    or an EARLIER interval.  Every rolling/lag feature here is causal.
  * Past LABELS are never available at test time (>=25h gap), so no target lags.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), u"정형데이터")

TARGET_FARMS = ["F13", "F47"]

OUT_COLS = ["out_temp", "out_hum", "out_rad", "out_wspd"]
IN_COLS = ["in_temp", "in_hum", "in_co2"]
ACT_COLS = ["act_vent", "act_shade", "act_thermal", "act_heating",
            "act_circfan", "act_co2", "act_fog"]
USABLE = OUT_COLS + IN_COLS + ACT_COLS          # 14 columns present in test
UNUSABLE = ["in_rad", "act_side", "act_valve", "act_cool", "act_pump"]


def load_raw():
    tX = pd.read_csv(os.path.join(DATA, "train_X.csv"))
    ty = pd.read_csv(os.path.join(DATA, "train_y.csv"))
    sX = pd.read_csv(os.path.join(DATA, "test_X.csv"))
    for df in (tX, ty, sX):
        df["farm"] = df.row_id.str[:3]
        df["day"] = df.row_id.str[4:7].astype(int)
        df["hour"] = df.row_id.str[8:10].astype(int)
        df["t"] = df.day * 24 + df.hour
    return tX, ty, sX


def _vpd(temp, rh):
    """Vapour pressure deficit (kPa) from air temperature and relative humidity."""
    es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    return es * (1.0 - rh / 100.0)


# Substrate-temperature and substrate-EC domain knowledge encoded below.
# References that motivated each block are named in the comments.
LATENT_HEAT = 2.45e6          # J/kg, vaporisation
BAILLE_A = 0.4                # dimensionless, Baille et al. 1994 range .12-.67
BAILLE_B = 25e-3              # kg/m2/h/kPa, Baille et al. 1994 range 14-37e-3
ROOT_OPT = 18.0               # degC, lower bound of the strawberry root optimum


def add_derived(p):
    """Physically motivated combinations, all from same-interval inputs."""
    p["vpd_in"] = _vpd(p["in_temp"], p["in_hum"])
    p["vpd_out"] = _vpd(p["out_temp"], p["out_hum"])
    p["dt_in_out"] = p["in_temp"] - p["out_temp"]
    p["dh_in_out"] = p["in_hum"] - p["out_hum"]
    p["transp"] = p["out_rad"] * p["vpd_in"]
    p["vent_loss"] = p["act_vent"] * p["dt_in_out"]

    # ---- radiation actually reaching the slab -----------------------------
    # in_rad is absent for F13/F47, so reconstruct it from the outside sensor
    # attenuated by the shading and thermal screens.
    # curtain values are openness (100 = retracted); see features_v2._derive
    shade = p["act_shade"].fillna(100) / 100.0
    therm = p["act_thermal"].fillna(100) / 100.0
    p["rad_eff"] = p["out_rad"] * shade * therm
    p["rad_shade_only"] = p["out_rad"] * shade

    # ---- Baille simplified Penman-Monteith transpiration (kg/m2/h) --------
    # T = A*Rs/lambda + B*VPD ; drives water removal, hence EC concentration.
    p["transp_pm"] = (BAILLE_A * p["rad_eff"] * 3600.0 / LATENT_HEAT
                      + BAILLE_B * p["vpd_in"])

    # ---- root-zone thermal terms -----------------------------------------
    p["root_dh"] = (p["in_temp"] - ROOT_OPT).clip(lower=0)   # degree-hours
    p["heat_input"] = p["act_heating"] * (ROOT_OPT - p["in_temp"]).clip(lower=0)
    # night-time radiative loss is suppressed while the thermal screen is shut
    p["screen_keep"] = (1.0 - therm) * p["dt_in_out"]
    return p


DERIVED = ["vpd_in", "vpd_out", "dt_in_out", "dh_in_out", "transp",
           "vent_loss", "rad_eff", "rad_shade_only", "transp_pm", "root_dh",
           "heat_input", "screen_keep"]

# columns that get lag / rolling treatment
DYN = ["in_temp", "in_hum", "in_co2", "out_temp", "out_hum", "out_rad",
       "out_wspd", "act_vent", "act_shade", "act_thermal", "act_heating",
       "act_circfan", "vpd_in", "transp", "dt_in_out", "rad_eff", "transp_pm"]

# thermal inertia: a substrate slab is a first-order low-pass filter on air
# temperature and radiation, so exponential kernels fit it better than boxcars.
EWM_COLS = ["in_temp", "out_temp", "rad_eff", "act_heating", "act_thermal",
            "transp_pm", "vpd_in", "root_dh"]
EWM_HL = [2, 4, 8, 16, 48]

LAGS = [1, 2, 3, 6, 12, 24]
ROLLS = [3, 6, 12, 24, 72, 168]


def build_panel(tX, sX):
    """Hourly panel for F13/F47 over train+test X with causal features.

    Using test_X rows as feature history is legitimate: they are *inputs*, and
    section 5 only forbids looking at later intervals.
    """
    cols = ["row_id", "farm", "day", "hour", "t"] + USABLE
    allx = pd.concat([tX[cols], sX[cols]], ignore_index=True)
    allx = allx[allx.farm.isin(TARGET_FARMS)]

    out = []
    for farm, g in allx.groupby("farm"):
        g = g.sort_values("t")
        full = np.arange(g.t.min(), g.t.max() + 1)
        p = g.set_index("t").reindex(full)
        p["farm"] = farm
        p["day"] = p.index // 24
        p["hour"] = p.index % 24
        p = add_derived(p)

        feats = {}
        for c in DYN:
            s = p[c]
            for L in LAGS:
                feats["%s_lag%d" % (c, L)] = s.shift(L)
            for W in ROLLS:
                # window closes on the current interval -> causal
                r = s.rolling(W, min_periods=max(2, W // 4))
                feats["%s_r%dm" % (c, W)] = r.mean()
                if W in (24, 72):
                    feats["%s_r%ds" % (c, W)] = r.std()
            feats["%s_d1" % c] = s - s.shift(1)
            feats["%s_d24" % c] = s - s.shift(24)
            feats["%s_dev24" % c] = s - feats["%s_r24m" % c]

        # ---- exponential (thermal-inertia) kernels ---------------------------
        for c in EWM_COLS:
            s = p[c]
            for hl in EWM_HL:
                e = s.ewm(halflife=hl, ignore_na=True).mean()
                feats["%s_ewm%d" % (c, hl)] = e
                if hl in (4, 16):
                    feats["%s_ewmdev%d" % (c, hl)] = s - e

        # ---- season-long accumulation (salt build-up in the substrate) -------
        elapsed = (p.index - p.index.min() + 1).astype(float)
        for c in ["transp_pm", "rad_eff", "root_dh"]:
            cs = p[c].fillna(0).cumsum()
            feats["%s_cum" % c] = cs
            feats["%s_cumrate" % c] = cs / elapsed
        p = pd.concat([p, pd.DataFrame(feats, index=p.index)], axis=1)

        # ---- previous whole-day aggregates (causal for every hour of day d) --
        dm = p.groupby("day")[DYN].mean()
        dmx = p.groupby("day")[["in_temp", "out_temp", "out_rad"]].max()
        dmn = p.groupby("day")[["in_temp", "out_temp"]].min()
        dagg = pd.concat([dm.add_suffix("_dmean"),
                          dmx.add_suffix("_dmax"),
                          dmn.add_suffix("_dmin")], axis=1)
        dfull = dagg.reindex(np.arange(dagg.index.min(), dagg.index.max() + 1))
        prev = {}
        for k in (1, 2, 3):
            for c in dfull.columns:
                prev["%s_pd%d" % (c, k)] = dfull[c].shift(k)
        for c in dfull.columns:
            prev["%s_pd1_7m" % c] = dfull[c].shift(1).rolling(7, min_periods=2).mean()
        prevdf = pd.DataFrame(prev, index=dfull.index)
        p = p.join(prevdf, on="day")

        # ---- expanding aggregates inside the current day (causal) ------------
        for c in ["in_temp", "out_rad", "transp", "act_vent", "in_co2",
                  "rad_eff", "transp_pm", "act_heating", "root_dh"]:
            gb = p.groupby("day")[c]
            p["%s_todaymean" % c] = gb.transform(lambda s: s.expanding().mean())
            p["%s_todaysum" % c] = gb.transform(lambda s: s.expanding().sum())

        # ---- RTR: 24h mean temperature over daily radiation sum --------------
        # Dutch growers steer assimilate balance with this ratio, so it tracks
        # the management regime that in turn sets irrigation and hence EC.
        p["rtr_today"] = p["in_temp_todaymean"] / (p["rad_eff_todaysum"] / 24.0 + 1.0)
        p["rtr_pd1"] = (p["in_temp_dmean_pd1"]
                        / (p["rad_eff_dmean_pd1"].abs() + 1.0))
        p["rtr_pd1_7m"] = (p["in_temp_dmean_pd1_7m"]
                           / (p["rad_eff_dmean_pd1_7m"].abs() + 1.0))
        # water balance proxy: transpiration demand relative to radiation, i.e.
        # how much drying happened per unit of irrigation trigger
        p["transp_per_rad_pd1"] = (p["transp_pm_dmean_pd1"]
                                   / (p["rad_eff_dmean_pd1"].abs() + 1.0))

        out.append(p.reset_index().rename(columns={"index": "t"}))

    panel = pd.concat(out, ignore_index=True)
    panel["hr_sin"] = np.sin(2 * np.pi * panel.hour / 24)
    panel["hr_cos"] = np.cos(2 * np.pi * panel.hour / 24)
    panel["day_par"] = panel.day % 2
    panel["day_mod7"] = panel.day % 7
    panel["farm_id"] = (panel.farm == "F47").astype(int)
    return panel.dropna(subset=["row_id"]).reset_index(drop=True)


def feature_columns(panel, extra_drop=()):
    drop = set(["row_id", "farm", "t", "sub_temp", "sub_ec"]) | set(UNUSABLE)
    drop |= set(extra_drop)
    return [c for c in panel.columns if c not in drop]


# --------------------------------------------------------------------------
# Cross-validation that reproduces the test geometry: contiguous blocks of
# 5/10/10/5 days, with a 1-day buffer excluded from training on each side, so
# validation rows sit >=25h away from any training label -- exactly like test.
# --------------------------------------------------------------------------
BLOCK_PATTERN = [5, 10, 10, 5]


def make_folds(days_by_farm, n_folds=6, seed=0):
    """Return a list of folds; each fold is {farm: set(validation days)}."""
    folds = [dict() for _ in range(n_folds)]
    for fi, (farm, days) in enumerate(sorted(days_by_farm.items())):
        days = np.sort(np.asarray(days))
        blocks, i, k = [], 0, 0
        while i < len(days):
            L = BLOCK_PATTERN[k % len(BLOCK_PATTERN)]
            blocks.append(days[i:i + L])
            i += L
            k += 1
        rng = np.random.RandomState(seed + 17 * fi)
        order = rng.permutation(len(blocks))
        for j, b in enumerate(order):
            folds[j % n_folds].setdefault(farm, []).extend(blocks[b].tolist())
    return [{f: set(v) for f, v in fd.items()} for fd in folds]


def split_mask(df, val_days, buffer_days=1):
    """Boolean masks (train, valid) honouring the 1-day buffer."""
    val = np.zeros(len(df), bool)
    buf = np.zeros(len(df), bool)
    d = df.day.values
    for farm, ds in val_days.items():
        isf = (df.farm == farm).values
        v = isf & np.isin(d, list(ds))
        val |= v
        near = set()
        for x in ds:
            for o in range(-buffer_days, buffer_days + 1):
                near.add(x + o)
        buf |= isf & np.isin(d, list(near))
    return (~buf), val


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))
