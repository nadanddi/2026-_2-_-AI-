# -*- coding: utf-8 -*-
"""Causal calendar / true-yesterday features for F13 and F47.

For each greenhouse-day d (row_id relative day), using ONLY
  * inputs of the same record (F13 or F47) on days with a SMALLER day index, and
  * inputs of day d itself at hour 00,
we compute:
  twin      earlier same-record day whose hour-00 weather matches d (same calendar date)
  pred      "same-source calendar yesterday": earlier same-record day e minimising
            midnight continuity of outdoor weather (e 22-23h -> d 0h) + indoor sensors
            + actuator states, with a mild preference for small gaps
  cday      calendar-equivalent day: cday(twin) if a twin exists, else
            cday(pred) + (d - pred) if gap<=3, else cday(pred) + 2, else d
Nothing uses labels, the other record, or any later day.
"""
import numpy as np
import pandas as pd

import common

TF = ["F13", "F47"]
W = ["out_temp", "out_hum", "out_rad", "out_wspd"]
IN = ["in_temp", "in_hum", "in_co2"]
WSC = np.array([1.186, 4.83, 50.0, 0.538])
ISC = {"in_temp": 1.0, "in_hum": 5.0, "in_co2": 40.0}
TWIN_THR = 0.02


def _days():
    tX, ty, sX = common.load_raw()
    a = pd.concat([tX, sX], ignore_index=True)
    a = a[a.farm.isin(TF)].sort_values(["farm", "day", "hour"])
    out = {}
    for f, g in a.groupby("farm"):
        dd = {}
        for d, x in g.groupby("day"):
            dd[int(d)] = x.set_index("hour").reindex(range(24))
        out[f] = dd
    return out


def day_summary(x):
    it = x.in_temp
    night = it[list(range(0, 7)) + [22, 23]]
    return dict(y_in_temp_m=it.mean(), y_in_temp_min=night.min(), y_in_temp_max=it.max(),
                y_in_hum_m=x.in_hum.mean(), y_in_co2_m=x.in_co2.mean(),
                y_heat_m=x.act_heating.mean(), y_heat_hrs=float((x.act_heating > 5).sum()),
                y_therm_m=x.act_thermal.mean(), y_vent_m=x.act_vent.mean(),
                y_circ_m=x.act_circfan.mean(), y_in_temp_23=it[23],
                y_dt_io=(x.in_temp - x.out_temp).mean())


def cont_cost(e, d, gap):
    """continuity e(22,23h) -> d(0h).  Only hour 0 of d is used."""
    c = 0.0
    for v, s, w in [("out_temp", WSC[0], 1.0), ("out_hum", WSC[1], 1.0), ("out_wspd", WSC[3], 0.3)]:
        g = (d[v][0] - (e[v][23] + 0.5 * (e[v][23] - e[v][22]))) / s
        c += 0.0 if np.isnan(g) else w * g * g
    for v in IN:
        g = (d[v][0] - (e[v][23] + 0.5 * (e[v][23] - e[v][22]))) / ISC[v]
        c += 0.0 if np.isnan(g) else g * g
    for v in ["act_heating", "act_thermal", "act_circfan", "act_vent"]:
        g = abs(d[v][0] - e[v][23]) / 50
        c += 0.0 if np.isnan(g) else g
    return c + 0.15 * np.log1p(max(gap - 1, 0))


def build():
    D = _days()
    rows = []
    for f, dd in D.items():
        days = sorted(dd)
        w0 = {d: dd[d][W].iloc[0].to_numpy(float) for d in days}
        cday = {}
        for i, d in enumerate(days):
            x = dd[d]
            prev = days[:i]
            twin, tdist = np.nan, np.nan
            if prev:
                dist = np.array([np.nanmean(np.abs(w0[e] - w0[d]) / WSC) for e in prev])
                if np.isfinite(dist).any():
                    j = int(np.nanargmin(dist))
                    tdist = dist[j]
                    if dist[j] < TWIN_THR:
                        twin = prev[j]
            pred, pcost, alt = np.nan, np.nan, np.nan
            if prev:
                cs = np.array([cont_cost(dd[e], x, d - e) for e in prev])
                o = np.argsort(cs)
                pred, pcost = prev[o[0]], cs[o[0]]
                alt = cs[o[1]] if len(o) > 1 else np.nan
            if not np.isnan(twin):
                cday[d] = cday[twin]
            elif not np.isnan(pred):
                g = d - pred
                cday[d] = cday[pred] + (g if g <= 3 else 2)
            else:
                cday[d] = d
            r = dict(farm=f, day=d, twin=twin, twin_dist=tdist, has_twin=float(not np.isnan(twin)),
                     pred=pred, pred_gap=(d - pred) if not np.isnan(pred) else np.nan,
                     pred_cost=pcost, pred_margin=(alt - pcost) if not np.isnan(alt) else np.nan,
                     cday=cday[d], cday_shift=d - cday[d])
            if not np.isnan(pred):
                r.update(day_summary(dd[int(pred)]))
            for lag in (1, 2):
                if d - lag in dd:
                    for kk, v in day_summary(dd[d - lag]).items():
                        r[kk.replace("y_", "l%d_" % lag)] = v
            rows.append(r)
    return pd.DataFrame(rows)


def to_rows(dayfeat, frame, cols):
    m = frame[["row_id", "farm", "day"]].merge(dayfeat[["farm", "day"] + cols],
                                               on=["farm", "day"], how="left")
    return m[["row_id"] + cols]


# ---------------------------------------------------------------- version 2
def wcost(e, d):
    """outdoor-weather-only midnight continuity e(22,23h) -> d(0h)."""
    c = 0.0
    for v, s, w in [("out_temp", WSC[0], 1.0), ("out_hum", WSC[1], 1.0), ("out_wspd", WSC[3], 0.3)]:
        g = (d[v][0] - (e[v][23] + 0.5 * (e[v][23] - e[v][22]))) / s
        c += 0.0 if np.isnan(g) else w * g * g
    return c


def icost(e, d):
    """indoor + actuator midnight continuity (source identity)."""
    c = 0.0
    for v in IN:
        g = (d[v][0] - (e[v][23] + 0.5 * (e[v][23] - e[v][22]))) / ISC[v]
        c += 0.0 if np.isnan(g) else g * g
    for v in ["act_heating", "act_thermal", "act_circfan", "act_vent"]:
        g = abs(d[v][0] - e[v][23]) / 50
        c += 0.0 if np.isnan(g) else g
    return c


W_ACCEPT = 3.0
SEARCH_LAMBDA = 2.0


def build2():
    """Two-stage causal rule.

    1. date group: days earlier than d whose hour-0 weather equals d's (twins).
    2. calendar-previous group P(d):
         if d has a twin t:  P(d) = P(t) (+ any earlier day twinned with those)
         else: e = latest earlier day not in d's date; if weather continuity e->d
               < W_ACCEPT, P(d) = date group of e; else search every earlier day
               for the best weather continuity and take its date group.
    3. same-source yesterday = argmin indoor/actuator continuity over P(d).
    cal2 = calendar step count: cal2(twin) if twin, else cal2(P-rep) + 1.
    """
    D = _days()
    rows = []
    for f, dd in D.items():
        days = sorted(dd)
        w0 = {d: dd[d][W].iloc[0].to_numpy(float) for d in days}
        grp = {}   # day -> canonical date id (first day seen)
        P = {}     # day -> list of calendar-previous days (earlier than day)
        cal2 = {}
        for i, d in enumerate(days):
            x = dd[d]
            prev = days[:i]
            twin = None
            if prev:
                dist = np.array([np.nanmean(np.abs(w0[e] - w0[d]) / WSC) for e in prev])
                if np.isfinite(dist).any() and np.nanmin(dist) < TWIN_THR:
                    twin = prev[int(np.nanargmin(dist))]
            grp[d] = grp[twin] if twin is not None else d
            members = lambda g, lim: [e for e in days if e < lim and grp.get(e) == g]
            how = "none"
            if twin is not None:
                pg = P.get(grp[twin], [])
                pgid = grp[pg[0]] if pg else None
                cand = members(pgid, d) if pgid is not None else []
                how = "twin"
            else:
                cand = []
                older = [e for e in prev if grp[e] != grp[d]]
                if older:
                    e = older[-1]
                    if wcost(dd[e], x) < W_ACCEPT:
                        cand = members(grp[e], d); how = "adjacent"
                    else:
                        cs = [wcost(dd[q], x) + SEARCH_LAMBDA * np.log1p(d - q) for q in older]
                        q = older[int(np.argmin(cs))]
                        cand = members(grp[q], d); how = "search"
            P[d] = cand
            if grp[d] == d:
                P[d] = cand
            else:
                P[grp[d]] = P.get(grp[d]) or cand
            if twin is not None:
                cal2[d] = cal2[twin]
            elif cand:
                cal2[d] = max(cal2[c] for c in cand) + 1
            else:
                cal2[d] = len(set(grp[e] for e in prev))
            pred, pc, pm = np.nan, np.nan, np.nan
            if cand:
                cs = np.array([icost(dd[c], x) for c in cand])
                o = np.argsort(cs)
                pred, pc = cand[o[0]], cs[o[0]]
                pm = cs[o[1]] - cs[o[0]] if len(o) > 1 else np.nan
            r = dict(farm=f, day=d, twin=np.nan if twin is None else twin, has_twin=float(twin is not None),
                     n_twins=float(len(members(grp[d], d))), how=how, n_cand=len(cand),
                     pred=pred, pred_gap=(d - pred) if not np.isnan(pred) else np.nan,
                     pred_cost=pc, pred_margin=pm, cal2=cal2[d],
                     second_pass=float(twin is not None and d - twin > 20))
            if not np.isnan(pred):
                r.update(day_summary(dd[int(pred)]))
            for lag in (1, 2):
                if d - lag in dd:
                    for kk, v in day_summary(dd[d - lag]).items():
                        r[kk.replace("y_", "l%d_" % lag)] = v
            rows.append(r)
    return pd.DataFrame(rows)
