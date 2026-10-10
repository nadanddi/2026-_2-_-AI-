# -*- coding: utf-8 -*-
"""TD23: do day-level CONTROL SETPOINTS (inferred from actuator switching, as F32's thermostat-like control suggests)
explain F13/F47 temperature phenomena that inputs could not?  (2026-10-10 집 클로드, user: "제어의 움직임 기준이 있을 것
아니야? 그걸 F13/F47의 설명하기 힘든 현상들과 연관 지을 수 있냐")  Diagnostic; criteria fixed before running.
Per (farm, day) setpoint features (inputs only):
  heat_on_T   median in_temp at hours where act_heating switches 0 -> >0 (NaN if none)
  heat_hold_T mean in_temp over night hours (0-6, 19-23) with act_heating >= 50   (controlled night air temperature)
  vent_on_T   in_temp at the first hour (6-14 h) with act_vent > 0
  vent_frac   share of 9-16 h with act_vent > 0
  th_close_h  first hour >= 14 with act_thermal >= 50 (curtain closing); th_open_h first hour 5-12 with act_thermal < 50
  fan_night   mean act_circfan 0-6 h;  co2_start_h first hour with act_co2 > 0
Targets: OFF = daily mean W40G-S DIAG10 residual (seed 7, PFN A; + = overpredicted); DEF (TD13: afternoon warming deficit);
  GAP = daily mean (sub - in_temp)  [label-only].  Rough strict days (6.437) excluded.
Tests per farm and pooled:
  (1) Spearman of each setpoint with OFF / DEF / GAP, partial on day in_temp mean; 'related' iff same sign in F13 and F47,
      min |rho| >= .2 and both p < .05/9.
  (2) non-causal upper bound: leave-one-day-out Ridge (alpha 10) of OFF on all 9 setpoints (+ day in_temp mean, out_temp mean)
      vs the same with only the 2 weather means; 'signal' iff R^2 gain > .05 with day-bootstrap 95% CI of the gain > 0.
  (3) F47 cold days (in_temp < 8 any hour): setpoints of the worst 6 OFF days vs the rest (descriptive).
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td23_control_setpoints_offsets_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
import common
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
CK = os.path.join(env.LOCAL, "tt1_ckpt")
RD = set(map(tuple, pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "rw1_rough_days_v1.csv"))[["farm", "day"]].values))
SP = ["heat_on_T", "heat_hold_T", "vent_on_T", "vent_frac", "th_close_h", "th_open_h", "fan_night", "co2_start_h"]


def first(h, m):
    h = h[m]; return float(h.iloc[0]) if len(h) else np.nan


def main():
    tX, ty, _ = common.load_raw()
    a = tX[tX.farm.isin(["F13", "F47"])].merge(ty[["row_id", "sub_temp"]], on="row_id").sort_values(["farm", "t"])
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True); G = G[G.validator == "DIAG10"]
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    G["e"] = .4 * G.base_REF_7 + (.2 + .4 * (1 - g)) * G.codex_REF + .4 * g * G.pfn_A - G.sub_temp
    a = a.merge(G[["row_id", "e"]], on="row_id")
    rows = []
    for (f, d), x in a.groupby(["farm", "day"]):
        if (f, d) in RD:
            continue
        x = x.sort_values("hour"); h = x.hour
        sw = (x.act_heating > 0) & (x.act_heating.shift(1).fillna(0) == 0)
        night = (h <= 6) | (h >= 19)
        hh = x[night & (x.act_heating >= 50)].in_temp
        mid = x[(h >= 13) & (h <= 16)]; m8 = x[h == 8]
        rows.append(dict(farm=f, day=d, OFF=x.e.mean(), GAP=(x.sub_temp - x.in_temp).mean(),
                         DEF=((mid.e.mean() - m8.e.mean()) if len(m8) and len(mid) else np.nan),
                         tin=x.in_temp.mean(), tout=x.out_temp.mean(), cold=bool((x.in_temp < 8).any()),
                         heat_on_T=x[sw].in_temp.median() if sw.any() else np.nan,
                         heat_hold_T=hh.mean() if len(hh) >= 2 else np.nan,
                         vent_on_T=first(x.in_temp, (h >= 6) & (h <= 14) & (x.act_vent > 0)),
                         vent_frac=(x[(h >= 9) & (h <= 16)].act_vent > 0).mean(),
                         th_close_h=first(h, (h >= 14) & (x.act_thermal >= 50)),
                         th_open_h=first(h, (h >= 5) & (h <= 12) & (x.act_thermal < 50)),
                         fan_night=x[h <= 6].act_circfan.mean(), co2_start_h=first(h, x.act_co2 > 0)))
    D = pd.DataFrame(rows)
    print("날 %d (잡음 낀 날 제외) | 설정값 결측률: %s" % (len(D), {c: round(D[c].isna().mean(), 2) for c in SP}))
    print("\n(1) 하루 실내 평균을 통제한 Spearman (F13 / F47)  [기준: 같은 부호, |ρ|≥.2, p<%.4f]" % (.05 / 9))
    for t in ("OFF", "DEF", "GAP"):
        for c in SP:
            res = []
            for f in ("F13", "F47"):
                z = D[D.farm == f][[c, t, "tin"]].dropna()
                if len(z) < 20:
                    res.append((np.nan, np.nan)); continue
                rx = z[c] - np.poly1d(np.polyfit(z.tin, z[c], 1))(z.tin); ry = z[t] - np.poly1d(np.polyfit(z.tin, z[t], 1))(z.tin)
                res.append(spearmanr(rx, ry))
            ok = all(not np.isnan(r[0]) for r in res) and np.sign(res[0][0]) == np.sign(res[1][0]) and min(abs(res[0][0]), abs(res[1][0])) >= .2 and max(res[0][1], res[1][1]) < .05 / 9
            if ok or max(abs(r[0]) if not np.isnan(r[0]) else 0 for r in res) >= .2:
                print("  %-4s %-12s F13 %+.2f (p %.3f) / F47 %+.2f (p %.3f) %s" % (t, c, res[0][0], res[0][1], res[1][0], res[1][1], "← 연관" if ok else ""))
    print("\n(2) 비인과 상한: 하루 오프셋 OFF를 날씨 평균만 vs 날씨+설정값 9개로 (날 하나 빼기 Ridge)")
    for f in ("F13", "F47", "both"):
        z = (D if f == "both" else D[D.farm == f]).reset_index(drop=True)
        y = z.OFF.values
        def lodo(cols):
            p = np.zeros(len(z)); X = z[cols].values
            for i in range(len(z)):
                m = np.arange(len(z)) != i
                p[i] = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)).fit(X[m], y[m]).predict(X[i:i + 1])[0]
            return p
        p0, p1 = lodo(["tin", "tout"]), lodo(["tin", "tout"] + SP)
        R = lambda yy, pp: 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            j = rng.integers(0, len(z), len(z)); bs.append(R(y[j], p1[j]) - R(y[j], p0[j]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print("  %-4s 날 %d | R² 날씨만 %+.3f, +설정값 %+.3f, 이득 %+.3f [%+.3f, %+.3f] → %s" % (f, len(z), R(y, p0), R(y, p1), R(y, p1) - R(y, p0), lo, hi, "신호" if R(y, p1) - R(y, p0) > .05 and lo > 0 else "정보 부족"))
    print("\n(3) F47 추운 날: 하루 오프셋 최악 6일 vs 나머지 (설정값 평균)")
    c47 = D[(D.farm == "F47") & D.cold].sort_values("OFF", ascending=False)
    print(pd.DataFrame({"최악6": c47.head(6)[SP + ["OFF"]].mean(), "나머지": c47.iloc[6:][SP + ["OFF"]].mean()}).round(2).T.to_string())
    print("  최악 6일:", list(zip(c47.head(6).day, c47.head(6).OFF.round(2))))


if __name__ == "__main__":
    main()
