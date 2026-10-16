# -*- coding: utf-8 -*-
"""Curated, physics-grouped feature set (replaces the blanket lag/roll grid).

Why this rewrite
----------------
v1 applied the same 17 transforms to every one of 17 columns, producing ~500
features.  For sub_ec the effective sample size is not 9,600 rows but ~400
greenhouse-days (95% of EC variance is the daily level), so v1 had more
features than samples.  Here every transform is chosen for a physical reason
and each target gets only the block of features that acts on its timescale:

  sub_temp -- a slab is a first-order thermal filter on air temperature,
              radiation reaching it, and pipe heating.  Exponential kernels at
              a few time constants + short lags; no season-long accumulators.
  sub_ec   -- a daily water/salt balance.  Previous-day aggregates, causal
              same-day sums, and season-long accumulation (salt build-up);
              no 1-3 hour dynamics, which only add noise at that timescale.

References behind the physical terms are named at each block.
"""
import numpy as np
import pandas as pd

from common import USABLE, TARGET_FARMS, _vpd

LATENT_HEAT = 2.45e6      # J/kg
BAILLE_A = 0.4            # Baille et al. 1994: 0.12-0.67
BAILLE_B = 25e-3          # Baille et al. 1994: 14-37e-3 kg/m2/h/kPa
ROOT_OPT = 18.0           # degC, strawberry root-zone optimum lower bound

EWM_HL = [2, 6, 24]
EWM_COLS = ["in_temp", "out_temp", "rad_eff", "act_heating", "transp_pm"]
DAILY_COLS = ["in_temp", "in_hum", "in_co2", "out_temp", "out_rad", "out_wspd",
              "rad_eff", "transp_pm", "vpd_in", "act_vent", "act_heating",
              "act_shade", "act_thermal"]
TODAY_COLS = ["in_temp", "rad_eff", "transp_pm", "act_vent", "act_heating",
              "root_dh"]


def _derive(p):
    p["vpd_in"] = _vpd(p["in_temp"], p["in_hum"])
    p["vpd_out"] = _vpd(p["out_temp"], p["out_hum"])
    p["dt_in_out"] = p["in_temp"] - p["out_temp"]
    # Curtain values are OPENNESS (PDF: "개폐율은 ... 열림 정도").  Verified on
    # the data: act_thermal ~8 at night / ~83 at midday, and at night a value
    # near 0 goes with a +7.6 degC in-out gap (screen shut, insulating) while
    # 90+ goes with -2 degC.  So 100 = retracted, 0 = deployed.
    shade_open = p["act_shade"].fillna(100) / 100.0
    therm_open = p["act_thermal"].fillna(100) / 100.0
    # in_rad is absent for F13/F47: radiation reaching the slab is what gets
    # past both screens, i.e. proportional to the OPEN fractions.
    p["rad_eff"] = p["out_rad"] * shade_open * therm_open
    # insulation term: the shut fraction of the thermal screen times the heat
    # it is holding in (Baille-style screen models use exactly this product)
    p["screen_ins"] = (1.0 - therm_open) * p["dt_in_out"]
    p["heat_screen"] = p["act_heating"] * (1.0 - therm_open)
    # Baille simplified Penman-Monteith transpiration, kg/m2/h
    p["transp_pm"] = (BAILLE_A * p["rad_eff"] * 3600.0 / LATENT_HEAT
                      + BAILLE_B * p["vpd_in"])
    p["root_dh"] = (p["in_temp"] - ROOT_OPT).clip(lower=0)
    p["heat_input"] = p["act_heating"] * (ROOT_OPT - p["in_temp"]).clip(lower=0)
    return p


INSTANT = USABLE + ["vpd_in", "vpd_out", "dt_in_out", "rad_eff", "transp_pm",
                    "root_dh", "heat_input", "screen_ins", "heat_screen"]


def build(tX, sX, farms=None):
    """farms=None -> the two evaluation greenhouses; pass a list to pool more."""
    cols = ["row_id", "farm", "day", "hour", "t"] + USABLE
    allx = pd.concat([tX[cols], sX[cols]], ignore_index=True)
    allx = allx[allx.farm.isin(farms or TARGET_FARMS)]

    out = []
    for farm, g in allx.groupby("farm"):
        g = g.sort_values("t")
        full = np.arange(g.t.min(), g.t.max() + 1)
        p = g.set_index("t").reindex(full)
        p["farm"] = farm
        p["day"] = p.index // 24
        p["hour"] = p.index % 24
        p = _derive(p)
        f = {}

        # -- thermal memory: first-order kernels at a few time constants ------
        for c in EWM_COLS:
            s = p[c]
            for hl in EWM_HL:
                f["%s_ewm%d" % (c, hl)] = s.ewm(halflife=hl, ignore_na=True).mean()
        f["in_temp_dev6"] = p.in_temp - f["in_temp_ewm6"]
        f["in_temp_dev24"] = p.in_temp - f["in_temp_ewm24"]
        f["rad_eff_dev6"] = p.rad_eff - f["rad_eff_ewm6"]

        # -- short dynamics (hourly response of the slab) ---------------------
        for L in (1, 2, 3):
            f["in_temp_lag%d" % L] = p.in_temp.shift(L)
        f["in_temp_d1"] = p.in_temp - p.in_temp.shift(1)
        f["in_temp_d3"] = p.in_temp - p.in_temp.shift(3)
        f["rad_eff_lag1"] = p.rad_eff.shift(1)
        f["out_temp_lag1"] = p.out_temp.shift(1)

        # -- 24h / multi-day context ------------------------------------------
        for c in ["in_temp", "rad_eff", "transp_pm", "vpd_in", "out_temp",
                  "act_heating"]:
            f["%s_r24m" % c] = p[c].rolling(24, min_periods=6).mean()
        for c in ["in_temp", "rad_eff"]:
            f["%s_r24s" % c] = p[c].rolling(24, min_periods=6).std()
        for c in ["in_temp", "rad_eff", "transp_pm"]:
            f["%s_r72m" % c] = p[c].rolling(72, min_periods=18).mean()
            f["%s_r168m" % c] = p[c].rolling(168, min_periods=42).mean()

        # -- season-long accumulation: salts left behind by transpiration ------
        # (nutrient salts not taken up accumulate and raise substrate EC)
        elapsed = (p.index - p.index.min() + 1).astype(float)
        for c in ["transp_pm", "rad_eff", "root_dh"]:
            cs = p[c].fillna(0).cumsum()
            f["%s_cum" % c] = cs
            f["%s_cumrate" % c] = cs / elapsed

        p = pd.concat([p, pd.DataFrame(f, index=p.index)], axis=1)

        # -- previous whole days (causal for every hour of day d) -------------
        dm = p.groupby("day")[DAILY_COLS].mean().add_suffix("_dm")
        dm["in_temp_dmax"] = p.groupby("day").in_temp.max()
        dm["in_temp_dmin"] = p.groupby("day").in_temp.min()
        dm["out_rad_dmax"] = p.groupby("day").out_rad.max()
        dfull = dm.reindex(np.arange(dm.index.min(), dm.index.max() + 1))
        prev = {}
        for c in dfull.columns:
            prev["%s_pd1" % c] = dfull[c].shift(1)
            prev["%s_pd7m" % c] = dfull[c].shift(1).rolling(7, min_periods=2).mean()
        for c in ["transp_pm_dm", "rad_eff_dm", "in_temp_dm"]:
            prev["%s_pd2" % c] = dfull[c].shift(2)
        p = p.join(pd.DataFrame(prev, index=dfull.index), on="day")

        # -- causal same-day accumulation --------------------------------------
        for c in TODAY_COLS:
            gb = p.groupby("day")[c]
            p["%s_tdmean" % c] = gb.transform(lambda s: s.expanding().mean())
            p["%s_tdsum" % c] = gb.transform(lambda s: s.expanding().sum())

        # -- RTR: growers steer irrigation by temperature-to-light ratio -------
        p["rtr_today"] = p.in_temp_tdmean / (p.rad_eff_tdsum / 24.0 + 1.0)
        p["rtr_pd1"] = p.in_temp_dm_pd1 / (p.rad_eff_dm_pd1.abs() + 1.0)
        p["rtr_pd7m"] = p.in_temp_dm_pd7m / (p.rad_eff_dm_pd7m.abs() + 1.0)
        p["transp_per_rad_pd1"] = (p.transp_pm_dm_pd1
                                   / (p.rad_eff_dm_pd1.abs() + 1.0))

        out.append(p.reset_index().rename(columns={"index": "t"}))

    panel = pd.concat(out, ignore_index=True)
    panel["hr_sin"] = np.sin(2 * np.pi * panel.hour / 24)
    panel["hr_cos"] = np.cos(2 * np.pi * panel.hour / 24)
    panel["farm_id"] = (panel.farm == "F47").astype(int)
    panel["farm_code"] = panel.farm.str[1:].astype(int)   # for pooled models
    panel["day_par"] = panel.day % 2
    return panel.dropna(subset=["row_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Target-specific views.  Each target only sees the timescales that physically
# act on it; this is the main overfitting control, applied before any tuning.
# --------------------------------------------------------------------------
META = ["row_id", "farm", "t", "sub_temp", "sub_ec", "is_test", "day", "hour"]


def _cols(panel, *preds):
    return [c for c in panel.columns
            if c not in META and any(p(c) for p in preds)]


def view(panel, target):
    base = ["day", "hour", "hr_sin", "hr_cos", "farm_id"]
    if target == "sub_temp":
        sel = _cols(
            panel,
            lambda c: c in INSTANT,
            lambda c: "_ewm" in c or "_dev" in c,
            lambda c: "_lag" in c or c.endswith("_d1") or c.endswith("_d3"),
            lambda c: "_r24" in c or "_r72m" in c or "_r168m" in c,
            lambda c: c.endswith("_tdmean") or c.endswith("_tdsum"),
            lambda c: c.endswith("_pd1"),
            lambda c: c == "rtr_today",
        )
    elif target == "sub_ec":
        sel = _cols(
            panel,
            lambda c: c in INSTANT,
            lambda c: c.endswith("_ewm24") or c == "in_temp_dev24",
            lambda c: "_r24m" in c or "_r72m" in c or "_r168m" in c,
            lambda c: c.endswith("_cum") or c.endswith("_cumrate"),
            lambda c: "_pd1" in c or "_pd2" in c or "_pd7m" in c,
            lambda c: c.endswith("_tdmean") or c.endswith("_tdsum"),
            lambda c: c.startswith("rtr_") or c.startswith("transp_per_rad"),
            lambda c: c == "day_par",
        )
    else:
        raise ValueError(target)
    return sorted(set(sel) | set(base))


# Day-level view: every column here is known at hour 00 of the day, so a model
# built on it can emit one constant value for the whole day without ever
# looking at a later interval.
def day_view(panel):
    return sorted(set(
        [c for c in panel.columns
         if c not in META and ("_pd1" in c or "_pd2" in c or "_pd7m" in c
                               or c.endswith("_cum") or c.endswith("_cumrate")
                               or c in ("rtr_pd1", "rtr_pd7m",
                                        "transp_per_rad_pd1"))]
        + ["day", "farm_id", "day_par"]))
