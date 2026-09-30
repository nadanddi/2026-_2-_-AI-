# -*- coding: utf-8 -*-
"""재분석 21 (2026-10-01, 집 클로드): 분리 날 생성 가설 H1~H5 검증 — re21_forensics_hypotheses.md(실행 전 고정)대로.
결과 local/re21_forensics_test.txt
"""
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

out = open(env.LOCAL + "/re21_forensics_test.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")

G = pd.read_csv("re17_day_offsets.csv")
X = pd.read_csv(env.DATA + "/train_X.csv"); Y = pd.read_csv(env.DATA + "/train_y.csv")
X = X[X.row_id.str[:3].isin(["F13", "F47"])].merge(Y[["row_id", "sub_temp"]], on="row_id")
k_ = X.row_id.str.extract(r"^(F\d+)_(\d+)_(\d+)$")
X["farm"], X["day"], X["hour"] = k_[0], k_[1].astype(int), k_[2].astype(int)
X = X.sort_values(["farm", "day", "hour"])
days = {(f, d): g.set_index("hour") for (f, d), g in X.groupby(["farm", "day"])}

def fit1(sub, air):
    s0, a0, ds = sub[:-1], air[:-1], np.diff(sub)
    m = ~(np.isnan(s0) | np.isnan(a0) | np.isnan(ds))
    if m.sum() < 10: return np.nan, np.nan, np.nan
    A = np.c_[a0[m] - s0[m], np.ones(m.sum())]
    coef, *_ = np.linalg.lstsq(A, ds[m], rcond=None)
    pred = A @ coef
    r2 = 1 - ((ds[m] - pred) ** 2).sum() / ((ds[m] - ds[m].mean()) ** 2).sum()
    return coef[0], coef[1], r2

rows = []
for (f, d), q in days.items():
    q = q.reindex(range(24))
    sub, air = q.sub_temp.values, q.in_temp.values
    k, c, r2 = fit1(sub, air)
    A = (np.nanmax(sub) - np.nanmin(sub)) / max(np.nanmax(air) - np.nanmin(air), 0.1)
    n = slice(0, 7)
    night_gap = np.nanmean(sub[n] - air[n])
    sl_s, sl_a = np.polyfit(range(7), sub[n], 1)[0] if not np.isnan(sub[n]).any() else np.nan, np.polyfit(range(7), air[n], 1)[0] if not np.isnan(air[n]).any() else np.nan
    dh = slice(8, 18)
    dsub, dair = np.diff(sub)[7:17], np.diff(air)[7:17]
    drops = int(np.nansum((dair > 0) & (dsub <= -0.3)))
    ds_, da_ = np.diff(sub), np.diff(air)
    lags = [np.corrcoef(ds_[L:], da_[:len(da_) - L])[0, 1] if L else np.corrcoef(ds_, da_)[0, 1] for L in range(0, 7)]
    best_lag = int(np.nanargmax(lags)) if not np.all(np.isnan(lags)) else np.nan
    rows.append(dict(farm=f, day=d, k=k, c=c, ss=c / k if k and k > 0.02 else np.nan, r2_own=r2, A=A,
                     night_gap=night_gap, night_slope_ratio=sl_s / sl_a if sl_a and abs(sl_a) > 0.05 else np.nan,
                     drops=drops, lag=best_lag))
R = pd.DataFrame(rows).merge(G[["farm", "day", "temp_offset", "group"]], on=["farm", "day"])

# H4: 같은 날짜 다른 기록(외기 24h 완전 일치)의 공기로 자기 배지 설명
sig = {k: tuple(np.round(q.reindex(range(24))[["out_temp", "out_hum", "out_rad", "out_wspd"]].values.ravel(), 1)) for k, q in days.items() if len(q) == 24}
by = {}
for k, s in sig.items(): by.setdefault(s, []).append(k)
best_other = []
for r in R.itertuples():
    key = (r.farm, r.day); others = [o for o in by.get(sig.get(key), []) if o != key]
    sub = days[key].reindex(range(24)).sub_temp.values
    r2o = [fit1(sub, days[o].reindex(range(24)).in_temp.values)[2] for o in others]
    best_other.append(np.nanmax(r2o) if r2o and not np.all(np.isnan(r2o)) else np.nan)
R["r2_other"] = best_other
R["other_better"] = R.r2_other > R.r2_own

p(f"날 {len(R)}  그룹 {R.group.value_counts().to_dict()}")
def cmp(col, label):
    out_ = []
    for gname in ("과대", "과소"):
        gg = {"과대": "over", "과소": "under"}[gname]
        a, b = R.loc[R.group == gg, col].dropna(), R.loc[R.group == "normal", col].dropna()
        pv = mannwhitneyu(a, b).pvalue if len(a) > 2 else np.nan
        out_.append(f"{gname} 중앙 {a.median():+.3f} (n {len(a)}) vs 정상 {b.median():+.3f} (n {len(b)}) p {pv:.4f}")
    p(f"  {label:34s} | " + " | ".join(out_))
p("\nH1 열용량/반응속도")
cmp("k", "k (공기 추종 속도, 1/h)")
cmp("A", "진폭비 A")
cmp("r2_own", "1차 모델 R² (자기 공기)")
p("H2 추가 열원")
cmp("ss", "정상상태 오프셋 c/k (℃)")
cmp("night_gap", "밤 0~6시 배지−공기 (℃)")
cmp("night_slope_ratio", "밤 배지 기울기/공기 기울기")
p("H3 찬 관수")
cmp("drops", "낮 '공기↑ 배지↓≥0.3' 시간 수")
p("H4 짝 어긋남")
for gg in ("over", "under", "normal"):
    q = R[(R.group == gg) & R.r2_other.notna()]
    p(f"  {gg:6s} 남의 공기가 더 잘 맞는 날 {int(q.other_better.sum())}/{len(q)} = {q.other_better.mean():.2f} | R² 자기 중앙 {q.r2_own.median():.3f} 남 최대 중앙 {q.r2_other.median():.3f}")
p("H5 시간 어긋남")
cmp("lag", "최적 지연 (h)")

# 전체 400일 연속 관계 (서술)
p("\n전체 400일 오프셋과 날별 파라미터 스피어만 (온실별)")
for c in ("k", "A", "ss", "night_gap", "drops", "lag", "r2_own"):
    rr = [spearmanr(R.loc[R.farm == f, c], R.loc[R.farm == f, "temp_offset"], nan_policy="omit")[0] for f in ("F13", "F47")]
    p(f"  {c:10s} F13 {rr[0]:+.3f}  F47 {rr[1]:+.3f}")
R.to_csv(env.LOCAL + "/re21_day_params.csv", index=False)
