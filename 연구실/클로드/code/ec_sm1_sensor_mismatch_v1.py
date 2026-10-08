# -*- coding: utf-8 -*-
"""SM1 — '실내 입력이 정답과 다른 센서(출처)에서 온 날' 가설 검사 (사용자 가설 10-08, 실행 전 고정) — 2026-10-08 연구실 클로드
근거: 배지온도는 실내온도를 약 3시간 늦게 따라감(5.3, 하루 안 상관 .9). 실내 입력이 정답과 다른 센서라면 그날 이 결합이 깨진다.
날 지표(정답 있는 F13·F47 400일, 입력은 학습 입력):
 K1 결합 = 하루 안 corr(sub_temp(t), EWMA_t(in_temp)) — EWMA 반감기 4h, 그날 0시 값에서 시작(잡음에 강하도록 평활), t=3..23
 K2 결합 잔차 = 그날 sub_temp ~ a + b·EWMA 회귀의 잔차 RMSE (℃)
 K3 수준 차 = 하루 평균(sub_temp − in_temp)의 1차 분포 대비 |z|
 K4 습도-온도 내부 일관성 = 하루 안 corr(in_temp, in_hum) (보통 강한 음)
 K5 정답끼리 = 하루 안 log EC ~ sub_temp 기울기 (6.27, 정상 약 .016~.02)
사전 예측(가설이 맞다면): CO₂ 거친 날(6.437 정의, 학습 1차 90분위 14.0 초과 & in_co2 차분 자기상관 < 0)이 나머지보다
 K1 낮음 · K2 큼 (단측 Mann-Whitney, 둘 다 p < .01). K5(정답끼리)는 차이 없어야 함(입력만 바뀐 것이므로).
 그리고 '불일치 날'(K1 < 1차 날 2.5분위 또는 K2 > 97.5분위)이 거친 날에 몰려야 함.
대조: 같은 수의 '거칠지 않은 F47 날'(같은 1·2차 비율)과 비교도 보고(F47 효과 분리).
추가 서술: 불일치 날의 DIAG10 하루 오차·편향(EC), 평가 입력에서 K4 분포(정답 없이 계산 가능한 유일한 지표) — 학습 불일치 날과 비교.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
R = os.path.dirname(os.path.abspath(__file__)); DC = os.path.join(R, "..", "local", "drive_copy")
def load(fn):
    d = pd.read_csv(os.path.join(env.DATA, fn)); d["farm"] = d.row_id.str[:3]; d["day"] = d.row_id.str[4:7].astype(int); d["hour"] = d.row_id.str[8:10].astype(int); return d
TR = load("train_X.csv").merge(pd.read_csv(os.path.join(env.DATA, "train_y.csv")), on="row_id", how="left")
TR = TR[TR.farm.isin(["F13", "F47"]) & TR.sub_ec.notna()].sort_values(["farm", "day", "hour"])
TE = load("test_X.csv").sort_values(["farm", "day", "hour"])
AL = 1 - 0.5 ** (1 / 4)
def ewm(x):
    x = pd.Series(x).ffill().bfill().values; o = np.empty_like(x, dtype=float); o[0] = x[0]
    for i in range(1, len(x)):
        o[i] = o[i - 1] + AL * (x[i] - o[i - 1])
    return o
rows = []
for (f, d), g in TR.groupby(["farm", "day"]):
    st, it, ih, ec = g.sub_temp.values, g.in_temp.values, g.in_hum.values, g.sub_ec.values
    e = ewm(it); m = np.arange(len(g)) >= 3; ok = m & ~np.isnan(st) & ~np.isnan(e)
    k1 = np.corrcoef(st[ok], e[ok])[0, 1] if ok.sum() > 8 and np.std(st[ok]) > 0 else np.nan
    if ok.sum() > 8:
        b = np.polyfit(e[ok], st[ok], 1); k2 = np.sqrt(np.mean((st[ok] - np.polyval(b, e[ok])) ** 2))
    else:
        k2 = np.nan
    k3 = np.nanmean(st - it)
    o2 = ~np.isnan(it) & ~np.isnan(ih); k4 = np.corrcoef(it[o2], ih[o2])[0, 1] if o2.sum() > 8 else np.nan
    o3 = ~np.isnan(st) & (ec > 0); k5 = np.polyfit(st[o3], np.log(ec[o3]), 1)[0] if o3.sum() > 8 and np.std(st[o3]) > .3 else np.nan
    rows.append((f, d, k1, k2, k3, k4, k5, g.sub_ec.mean()))
K = pd.DataFrame(rows, columns=["farm", "day", "K1", "K2", "K3raw", "K4", "K5", "ec"]); K["p2"] = K.day >= 179
mu, sd = K.loc[~K.p2, "K3raw"].mean(), K.loc[~K.p2, "K3raw"].std(); K["K3"] = ((K.K3raw - mu) / sd).abs()
C = pd.read_csv(os.path.join(R, "..", "results", "ec_co2r_days_v1.csv"))
H = pd.read_csv(os.path.join(R, "..", "results", "ec_hc1_day_change_v1.csv"))[["farm", "day", "in_co2_ac"]]
K = K.merge(C[["farm", "day", "med"]], on=["farm", "day"]).merge(H, on=["farm", "day"])
thr = C.loc[C.grp == "TR1", "med"].quantile(.9); K["rough"] = (K.med > thr) & (K.in_co2_ac < 0)
print(f"정답 날 {len(K)}, 거친 날(문턱 {thr:.1f} & 자기상관<0) {int(K.rough.sum())} — F13 {int(K[(K.farm=='F13')].rough.sum())}, F47 {int(K[(K.farm=='F47')].rough.sum())}")
print("\n지표 중앙값 (거친 / 나머지 전체 / 나머지 F47만):")
for k, alt in [("K1", "less"), ("K2", "greater"), ("K3", "greater"), ("K4", "greater"), ("K5", "two-sided")]:
    a, b, c = K.loc[K.rough, k].dropna(), K.loc[~K.rough, k].dropna(), K.loc[~K.rough & (K.farm == "F47"), k].dropna()
    p = mannwhitneyu(a, b, alternative=alt).pvalue; pc = mannwhitneyu(a, c, alternative=alt).pvalue
    print(f"  {k}: {a.median():+.3f} / {b.median():+.3f} / {c.median():+.3f} | 단측({alt}) p 전체 {p:.2e}, F47 대조 {pc:.2e}")
lo1, hi2 = K.loc[~K.p2, "K1"].quantile(.025), K.loc[~K.p2, "K2"].quantile(.975)
K["mis"] = (K.K1 < lo1) | (K.K2 > hi2)
print(f"\n불일치 날(K1<{lo1:.3f} 또는 K2>{hi2:.3f}): {int(K.mis.sum())}일 — 거친 날 중 {int((K.mis & K.rough).sum())}/{int(K.rough.sum())} ({(K.mis & K.rough).sum()/K.rough.sum():.2f}), 나머지 중 {int((K.mis & ~K.rough).sum())}/{int((~K.rough).sum())} ({(K.mis & ~K.rough).sum()/(~K.rough).sum():.3f})")
print("불일치 날 목록:", [(f"{r.farm}_{r.day}", round(r.K1, 2), round(r.K2, 2), 'R' if r.rough else '', round(r.ec, 2)) for r in K[K.mis].sort_values(["farm", "day"]).itertuples()])
print("\n거친 날 K1 하위 8:", [(f"{r.farm}_{r.day}", round(r.K1, 2), round(r.K2, 2)) for r in K[K.rough].sort_values("K1").head(8).itertuples()])
WT = pd.read_csv(os.path.join(DC, "ec3_WT1_all.csv")); WT = WT[WT.validator == "DIAG10"]
cur = np.mean([np.clip(.8 * (.6 * WT["et_%d" % s] + .3 * WT["lgb_%d" % s] + .1 * WT["mlp_%d" % s]) + .2 * WT.pfn, WT.lo, WT.hi) for s in (47, 1414, 6464)], axis=0)
de = WT.assign(e2=(cur - WT.sub_ec) ** 2, r=cur - WT.sub_ec).groupby(["farm", "day"]).agg(mse=("e2", "mean"), bias=("r", "mean")).reset_index()
T = K.merge(de, on=["farm", "day"])
print(f"\nDIAG10 EC: 불일치 {int(T.mis.sum())}일 MSE {T[T.mis].mse.mean():.3f} 편향 {T[T.mis].bias.mean():+.3f} | 나머지 MSE {T[~T.mis].mse.mean():.3f} 편향 {T[~T.mis].bias.mean():+.3f}")
print(f"ρ(K1, EC 하루 MSE) {spearmanr(T.K1, T.mse, nan_policy='omit').correlation:+.3f}, ρ(K2, MSE) {spearmanr(T.K2, T.mse, nan_policy='omit').correlation:+.3f}")
k4te = TE.groupby(["farm", "day"]).apply(lambda g: np.corrcoef(g.in_temp, g.in_hum)[0, 1], include_groups=False)
print(f"\nK4(실내 온도-습도 상관) 중앙값: 평가 {k4te.median():+.3f} (10분위 {k4te.quantile(.1):+.3f}) | 학습 거친 {K[K.rough].K4.median():+.3f} | 학습 나머지 {K[~K.rough].K4.median():+.3f}")
K.to_csv(os.path.join(R, "..", "results", "ec_sm1_days_v1.csv"), index=False)
