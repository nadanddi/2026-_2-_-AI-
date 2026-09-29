# -*- coding: utf-8 -*-
"""Generation structure: midnight jumps, cross-farm day copies; F32 actuator event responses; label rounding facts."""
import env  # noqa
import numpy as np, pandas as pd
P = pd.read_pickle(env.LOCAL + "/eda_farm_panels.pkl")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
# ---------- Q5a midnight jumps
rows = []
for f, p in P.items():
    ds = p.sub_temp.diff(); di = p.in_temp.diff(); dh = p.in_hum.diff()
    h0 = p.hour == 0; oth = (p.hour != 0) & (p.hour != 12)
    rows.append(dict(farm=f, sub_abs_h0=ds[h0].abs().mean(), sub_abs_oth=ds[oth].abs().mean(),
                     in_abs_h0=di[h0].abs().mean(), in_abs_oth=di[oth].abs().mean(),
                     hum_h0=dh[h0].abs().mean(), hum_oth=dh[oth].abs().mean(),
                     # residual jump: sub diff not explained by in diff
                     sub_min_in_h0=(ds - di * 0.35)[h0].abs().mean(), sub_min_in_oth=(ds - di * 0.35)[oth].abs().mean()))
J = pd.DataFrame(rows).set_index("farm")
J["sub_ratio"] = J.sub_abs_h0 / J.sub_abs_oth; J["in_ratio"] = J.in_abs_h0 / J.in_abs_oth; J["hum_ratio"] = J.hum_h0 / J.hum_oth
print("midnight |diff| ratios (hour0 / other hours)"); print(J[["sub_ratio", "in_ratio", "hum_ratio"]].round(2).T.to_string())
print("farms with sub_ratio>1.5:", list(J.index[J.sub_ratio > 1.5]), " in_ratio>1.5:", list(J.index[J.in_ratio > 1.5]))
# jump by hour profile for F13, F47
for f in ["F13", "F47"]:
    p = P[f]; L = p[p.sub_temp.notna()]
    print(f, "mean|dsub| by hour:", p.sub_temp.diff().abs().groupby(p.hour).mean().round(2).values)
    print(f, "mean|din| by hour:", p.in_temp.diff().abs().groupby(p.hour).mean().round(2).values)
# ---------- Q5b cross-farm daily copies of indoor 24h vectors
keys = {}
for f, p in P.items():
    q = p.dropna(subset=["in_temp", "in_hum"])
    for d, g in q.groupby("day"):
        if len(g) != 24: continue
        k = tuple(np.round(g.in_temp.values, 1)) + tuple(np.round(g.in_hum.values, 0))
        keys.setdefault(k, []).append((f, d))
dup = [v for v in keys.values() if len(set(x[0] for x in v)) > 1]
pairs = {}
for v in dup:
    fs = sorted(set(x[0] for x in v))
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            a = [x[1] for x in v if x[0] == fs[i]][0]; b = [x[1] for x in v if x[0] == fs[j]][0]
            pairs.setdefault((fs[i], fs[j]), []).append(b - a)
print("cross-farm identical indoor days (pair: n, day offsets):")
for k, v in sorted(pairs.items(), key=lambda kv: -len(kv[1])): print(" ", k, len(v), sorted(set(v))[:5])
within = sum(len(v) - len(set(x[0] for x in v)) for v in keys.values())
print("within-farm repeated indoor days:", within)
# hour-level partial match (3 indoor vars equal) between any pair, via hash on (in_temp,in_hum,in_co2) hourly triples in runs of 6
# ---------- Q4 F32 actuators
p = P["F32"].copy()
print("\nF32 columns non-null share:", p[["act_pump", "act_valve", "act_cool", "act_side", "act_vent", "act_circfan", "act_co2", "in_rad", "out_temp"]].notna().mean().round(2).to_dict())
for c in ["act_pump", "act_valve", "act_cool", "act_side"]:
    print(c, "value counts top:", p[c].round(1).value_counts().head(6).to_dict())
p["dsub"] = p.sub_temp.diff(); p["din"] = p.in_temp.diff()
L = p[p.sub_temp.notna()]
import numpy.linalg as la
X = L[["din", "act_pump", "act_valve", "act_cool", "act_side", "in_rad"]].copy()
X["hour"] = L.hour
m = X.notna().all(1) & L.dsub.notna()
A = np.c_[np.ones(m.sum()), X.loc[m, ["din", "act_pump", "act_valve", "act_cool", "act_side", "in_rad"]].values]
c, *_ = la.lstsq(A, L.dsub[m].values, rcond=None)
print("F32 dsub ~ din + pump + valve + cool + side + in_rad coef:", np.round(c, 4))
# event-triggered: pump onset
on = (p.act_pump > 0) & (p.act_pump.shift(1).fillna(0) <= 0)
idx = np.where(on.values)[0]
print("pump onsets:", len(idx), " hour dist:", pd.Series(p.hour.values[idx]).value_counts().sort_index().to_dict())
resp = []
for i in idx:
    if i - 3 < 0 or i + 4 >= len(p): continue
    s = p.sub_temp.values[i - 3:i + 4] - p.sub_temp.values[i - 1]; a = p.in_temp.values[i - 3:i + 4] - p.in_temp.values[i - 1]
    resp.append(np.r_[s, a])
R = np.nanmean(np.array(resp), 0)
print("pump onset response t-3..t+3 sub:", np.round(R[:7], 2), " air:", np.round(R[7:], 2))
# matched-hour comparison: dsub - k*din for pump on vs off at same hour
p["res"] = p.dsub - 0.5 * p.din
g = p[p.sub_temp.notna()].groupby([p.hour, p.act_pump > 0]).res.mean().unstack()
print("F32 mean (dsub-0.5 din) by hour x pump>0:"); print(g.round(3).T.to_string())
# F32 daily offset: sub - in daily mean; relation to daily pump sum
d = p.groupby("day").agg(off=("sub_temp", "mean"), air=("in_temp", "mean"), pump=("act_pump", "sum"), valve=("act_valve", "sum"), cool=("act_cool", "sum"))
d["off"] = d.off - d.air; d = d.dropna()
print("F32 daily corr(off, pump/valve/cool):", d.corr().loc["off", ["pump", "valve", "cool", "air"]].round(3).to_dict())
# F32 midnight jump
print("F32 mean|dsub| by hour:", p.dsub.abs().groupby(p.hour).mean().round(2).values)
