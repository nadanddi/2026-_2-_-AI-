# -*- coding: utf-8 -*-
"""TD14: growth-related (생육) fingerprints for substrate temperature, from the expert answers (10-09) + literature survey
(10-10 agent: 급액량 ∝ 식물 크기 x 일사, 물탱크 수온, 캐노피 가림, 센서 아래쪽).  (2026-10-10 집 클로드, user: "답변 내용으로
파생되거나 연계될 수 있는 생육 정보 쪽을 조사해봐")  Diagnostic, F13/F47 training days, inputs + labels.  Fixed before running:
Per day: RISE = sub(13-16 h mean) - sub(8 h); PRED_RISE = same on W40G-S DIAG10 OOF (seed 7, PFN A);
  DEF = PRED_RISE - RISE (+ = substrate warmed less than the model expected);  RATIO = RISE / (in_temp 13-16 mean - in 8 h).
 H1 night/tank: DEF vs same-record night (0-6 h) in_temp mean and act_heating mean.  Predicted: colder night -> DEF up
    (rho(night in_temp, DEF) < 0; rho(heating, DEF) > 0).   [prior: 6b.39 multi-day tank memory rejected; this uses the
    SAME record's night only]
 H2 growth progression: RATIO vs day index, partial on daily radiation sum and day in_temp mean; predicted rho < 0.
 H3 canopy proxies (closed hours: act_vent == 0 and act_co2 == 0, 7-11 h): CO2 drawdown per radiation, absolute-humidity
    rise per radiation; 7-record rolling mean; RATIO partial on day index + radiation; predicted rho < 0.
 H4 canopy shading: per 14-day window, OLS slope of hourly sub change (8-15 h) on out_rad (controlling in_temp change);
    predicted monotone decline: Spearman(window index, slope) <= -.5.
 H5 irrigation pulses (sensor near slab bottom): skewness of hourly residual change (de) at 8-15 h vs 20-5 h;
    predicted day-hours skew < night skew - .3.
PASS per hypothesis: predicted sign in BOTH farms, |rho| >= .20, p < .01 (k = 5, .05/5) [H4/H5: their own thresholds,
  both farms].  Reported also on cold days (in_temp < 8 any hour).  Passing = 'data supports', NOT a model gain.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u td14_growth_fingerprints_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
import common
from scipy.stats import spearmanr, skew
CK = os.path.join(env.LOCAL, "tt1_ckpt")


def abs_hum(t, rh):
    return 6.112 * np.exp(17.67 * t / (t + 243.5)) * rh * 2.1674 / (273.15 + t)


def partial(df, x, y, ctrl):
    d = df[[x, y] + ctrl].dropna()
    if len(d) < 15:
        return np.nan, np.nan, len(d)
    A = np.c_[np.ones(len(d)), d[ctrl].values]
    rx = d[x] - A @ np.linalg.lstsq(A, d[x], rcond=None)[0]; ry = d[y] - A @ np.linalg.lstsq(A, d[y], rcond=None)[0]
    rho, p = spearmanr(rx, ry)
    return rho, p, len(d)


def main():
    tX, ty, _ = common.load_raw()
    a = tX[tX.farm.isin(["F13", "F47"])].merge(ty[["row_id", "sub_temp"]], on="row_id").sort_values(["farm", "t"]).reset_index(drop=True)
    G = pd.concat([pd.read_csv(os.path.join(CK, f)) for f in sorted(os.listdir(CK))], ignore_index=True)
    G = G[G.validator == "DIAG10"]
    g = np.where(G.in_temp.isna(), 1, np.clip((G.in_temp - 8) / 2, 0, 1))
    G["p"] = .4 * G.base_REF_7 + (.2 + .4 * (1 - g)) * G.codex_REF + .4 * g * G.pfn_A
    a = a.merge(G[["row_id", "p"]], on="row_id")
    a["e"] = a.p - a.sub_temp
    a["ah"] = abs_hum(a.in_temp, a.in_hum)
    rows = []
    for (f, d), x in a.groupby(["farm", "day"]):
        x = x.set_index("hour")
        if not {8, 13, 14, 15, 16}.issubset(x.index):
            continue
        mid = x.loc[13:16]
        rise, prise, air = mid.sub_temp.mean() - x.sub_temp[8], mid.p.mean() - x.p[8], mid.in_temp.mean() - x.in_temp[8]
        cl = x.loc[7:11]; cl = cl[(cl.act_vent.fillna(0) == 0) & (cl.act_co2.fillna(0) == 0)]
        radc = cl.out_rad.sum()
        co2d = (-(cl.in_co2.diff()).sum() / radc) if len(cl) >= 3 and radc > 50 else np.nan
        ahd = ((cl.ah.diff()).sum() / radc) if len(cl) >= 3 and radc > 50 else np.nan
        rows.append(dict(farm=f, day=d, DEF=prise - rise, RATIO=rise / air if air > 1 else np.nan,
                         night_in=x.loc[0:6].in_temp.mean(), night_heat=x.loc[0:6].act_heating.mean(),
                         rad=x.out_rad.sum(), tin=x.in_temp.mean(), co2d=co2d, ahd=ahd, cold=bool((x.in_temp < 8).any())))
    D = pd.DataFrame(rows)
    for c in ("co2d", "ahd"):
        D[c + "7"] = D.groupby("farm")[c].transform(lambda s: s.rolling(7, min_periods=3).mean())
    tests = [("H1", "night_in", "DEF", ["tin", "rad"], -1), ("H1", "night_heat", "DEF", ["tin", "rad"], +1),
             ("H2", "day", "RATIO", ["rad", "tin"], -1),
             ("H3", "co2d7", "RATIO", ["day", "rad", "tin"], -1), ("H3", "ahd7", "RATIO", ["day", "rad", "tin"], -1)]
    for subset in ("전체", "추운 날"):
        S = D if subset == "전체" else D[D.cold]
        print("\n######## %s (%d일)" % (subset, len(S)))
        for h, x, y, ctrl, sgn in tests:
            res = [partial(S[S.farm == f], x, y, ctrl) for f in ("F13", "F47")]
            ok = all(not np.isnan(rr[0]) and np.sign(rr[0]) == sgn and abs(rr[0]) >= .2 and rr[1] < .01 for rr in res)
            print("  %s %-10s → %-5s (통제 %s) 예상 부호 %+d | F13 rho %+.2f p %.3f n%d | F47 rho %+.2f p %.3f n%d | %s" % (
                h, x, y, ",".join(ctrl), sgn, *res[0], *res[1], "지지" if ok else "불지지"))
    print("\n== H4 14일 구간별 '배지 시간 변화 ~ 일사' 기울기 (8~15시, 실내 변화 통제)")
    for f in ("F13", "F47"):
        x = a[(a.farm == f) & a.hour.between(8, 15)].copy()
        x["ds"] = x.groupby("day").sub_temp.diff(); x["di"] = x.groupby("day").in_temp.diff(); x = x.dropna(subset=["ds", "di", "out_rad"])
        x["w"] = x.day // 14; sl = []
        for w, z in x.groupby("w"):
            if len(z) > 40:
                A = np.c_[np.ones(len(z)), z.out_rad, z.di]; sl.append((w, np.linalg.lstsq(A, z.ds, rcond=None)[0][1] * 100))
        sl = np.array(sl); rho = spearmanr(sl[:, 0], sl[:, 1])[0]
        print("  %s 구간 %d개, 일사 100당 ℃/h: %s | Spearman(구간, 기울기) %+.2f → %s" % (f, len(sl), np.round(sl[:, 1], 3), rho, "지지" if rho <= -.5 else "불지지"))
    print("\n== H5 시간별 잔차 변화의 비대칭(왜도): 낮 8~15시 vs 밤 20~5시")
    a["de"] = a.groupby(["farm", "day"]).e.diff()
    for f in ("F13", "F47"):
        x = a[a.farm == f]
        dsk = skew(x[x.hour.between(8, 15)].de.dropna()); nsk = skew(x[(x.hour >= 20) | (x.hour <= 5)].de.dropna())
        hs = {h: round(float(skew(x[x.hour == h].de.dropna())), 2) for h in range(8, 16)}
        print("  %s 낮 왜도 %+.2f, 밤 %+.2f → %s | 시각별 %s" % (f, dsk, nsk, "지지" if dsk < nsk - .3 else "불지지", hs))
    print("\n참고: 잔차 변화 de = 예측-정답의 시간 변화. 찬 관수로 배지가 갑자기 내려가면 de가 양(+)으로 튐 → 'de 왜도 > 0'이 관수 흔적.")
    for f in ("F13", "F47"):
        x = a[a.farm == f]
        print("  %s 낮 de 왜도(부호 반전 확인) %+.2f / 밤 %+.2f" % (f, skew(x[x.hour.between(8, 15)].de.dropna()), skew(x[(x.hour >= 20) | (x.hour <= 5)].de.dropna())))


if __name__ == "__main__":
    main()
