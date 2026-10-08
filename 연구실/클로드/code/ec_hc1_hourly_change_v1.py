# -*- coding: utf-8 -*-
"""HC1 — 구동기·실내·외부 변수의 시간당 변화폭 분석 (서술, 사용자 요청 10-08) — 2026-10-08 연구실 클로드
DV(데이터·검증 의심)의 하나. 새 특징 탐색이 아니라 '학습 입력이 평가 입력과 같은 방식으로 기록·가공되었나'를 본다.
단위: 같은 날 안의 1시간 차분 |x_h − x_{h−1}| (h=1..23), 자정 차분(전 기록 23시 → 이 기록 0시)은 따로.
집단: TR1 = F13·F47 학습 1차(day<179), TR2 = 학습 2차(day≥179), TE = 평가(test_X, 전부 2차), OTH = 다른 온실 학습.
날 단위 통계: 하루 평균 |Δ|, 하루 0변화 비율, 차분 1차 자기상관. 집단 비교는 날 단위 Mann-Whitney(양측), 변수 14 × 비교 2 = 28 → 본페로니 .05/28.
추가: (1) 시각대(밤 1~6, 낮 7~18, 저녁 19~23) 평균 |Δ|, (2) 자정 차분/다른 시각 비, (3) 구동기 0 아닌 변화의 값 해상도(가장 흔한 크기 5개, 정수 비율),
(4) F13·F47 학습 날의 하루 거칠기와 EC(일평균)·DIAG10 하루 오차(CUR, 시드 평균)의 스피어만 — 서술만.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

R = os.path.dirname(os.path.abspath(__file__)); DC = os.path.join(R, "..", "local", "drive_copy")
OUT = ["out_temp", "out_hum", "out_rad", "out_wspd"]; IN = ["in_temp", "in_hum", "in_co2"]
ACT = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
V = OUT + IN + ACT

def load(fn):
    d = pd.read_csv(os.path.join(env.DATA, fn)); d["farm"] = d.row_id.str[:3]; d["day"] = d.row_id.str[4:7].astype(int); d["hour"] = d.row_id.str[8:10].astype(int)
    return d.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
TR = load("train_X.csv"); TE = load("test_X.csv")
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); TR = TR.merge(Y, on="row_id", how="left")
TR["grp"] = np.where(TR.farm.isin(["F13", "F47"]), np.where(TR.day >= 179, "TR2", "TR1"), "OTH"); TE["grp"] = "TE"
# 학습 F13·F47 은 정답 있는 날만(평가 날은 test_X 쪽)
TR = TR[(TR.grp == "OTH") | TR.sub_ec.notna()]
D = pd.concat([TR, TE], ignore_index=True)
D = D.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
for v in V:
    D["d_" + v] = D.groupby(["farm", "day"])[v].diff()
# 자정 차분: 같은 온실 직전 기록(day−1) 23시 → 이 날 0시 (직전 기록이 test·train 어디든 같은 온실 연속 번호면 사용)
prev = D[D.hour == 23][["farm", "day"] + V].copy(); prev["day"] += 1
mid = D[D.hour == 0][["farm", "day", "grp"] + V].merge(prev, on=["farm", "day"], suffixes=("", "_p"))
for v in V:
    mid["m_" + v] = mid[v] - mid[v + "_p"]

def dayagg(g):
    r = {}
    for v in V:
        x = g["d_" + v].values[1:]; ok = ~np.isnan(x)
        if ok.sum() < 12:
            r[v + "_m"] = r[v + "_z"] = r[v + "_ac"] = np.nan; continue
        a = np.abs(x[ok]); r[v + "_m"] = a.mean(); r[v + "_z"] = (a == 0).mean()
        xx = x[ok]; r[v + "_ac"] = np.corrcoef(xx[:-1], xx[1:])[0, 1] if xx.std() > 0 else np.nan
    return pd.Series(r)
DA = D.groupby(["farm", "day", "grp"]).apply(dayagg, include_groups=False).reset_index()
print("날 수:", DA.grp.value_counts().to_dict())
ff = DA.farm.isin(["F13", "F47"])
GR = ["TR1", "TR2", "TE", "OTH"]; NT = 28; ALPHA = .05 / NT

print("\n=== 1. 하루 평균 |Δ| (날 단위 중앙값) / 0변화 비율 / 차분 자기상관 ===")
print(f"{'변수':12s} " + " ".join(f"{g:>22s}" for g in GR) + " | TE÷TR2  p(TE vs TR2)  p(TE vs TR1)")
for v in V:
    cells = []
    for g in GR:
        s = DA[DA.grp == g]
        cells.append(f"{s[v+'_m'].median():8.3f} z{s[v+'_z'].median():.2f} ac{s[v+'_ac'].median():+.2f}" if s[v + "_m"].notna().any() else f"{'결측':>22s}")
    a, b, c = DA.loc[DA.grp == "TE", v + "_m"].dropna(), DA.loc[DA.grp == "TR2", v + "_m"].dropna(), DA.loc[DA.grp == "TR1", v + "_m"].dropna()
    p2 = mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan; p1 = mannwhitneyu(a, c).pvalue if len(a) and len(c) else np.nan
    st = lambda p: "**" if p < ALPHA else ("*" if p < .05 else "")
    print(f"{v:12s} " + " ".join(f"{c_:>22s}" for c_ in cells) + f" | {a.median()/max(b.median(),1e-9):6.2f}  {p2:9.2e}{st(p2):2s} {p1:9.2e}{st(p1):2s}")

print("\n=== 2. 시각대별 평균 |Δ| (F13·F47) — 밤 1~6 / 낮 7~18 / 저녁 19~23 ===")
D["band"] = pd.cut(D.hour, [0, 6, 18, 23], labels=["밤", "낮", "저녁"])
F = D[D.farm.isin(["F13", "F47"]) & (D.hour >= 1)]
for v in V:
    t = F.assign(a=F["d_" + v].abs()).groupby(["grp", "band"], observed=True).a.mean().unstack()
    print(f"{v:12s} " + " | ".join(f"{g}: " + "/".join(f"{t.loc[g, b]:.3f}" for b in ["밤", "낮", "저녁"]) for g in ["TR1", "TR2", "TE"] if g in t.index))

print("\n=== 3. 자정 |Δ| ÷ 다른 시각 |Δ| (같은 집단, F13·F47 vs 다른 온실) ===")
for v in V:
    cells = []
    for g in GR:
        m = mid.loc[mid.grp == g, "m_" + v].abs().mean(); o = D.loc[(D.grp == g) & (D.hour >= 1), "d_" + v].abs().mean()
        cells.append(f"{g} {m/o:5.2f}" if o and o > 0 and not np.isnan(m) else f"{g}   -  ")
    print(f"{v:12s} " + "  ".join(cells))

print("\n=== 4. 구동기 0 아닌 변화의 값 해상도 ===")
for v in ACT:
    for g in ["TR1", "TR2", "TE", "OTH"]:
        x = D.loc[(D.grp == g) & (D.hour >= 1), "d_" + v].abs(); x = x[(x > 0) & x.notna()]
        if len(x) < 20:
            continue
        vc = x.round(4).value_counts(); frac_int = (np.abs(x - x.round()) < 1e-6).mean()
        print(f"{v:12s} {g:4s} n {len(x):6d} 정수 {frac_int:.2f} 상위 {[(float(k), round(c/len(x),2)) for k, c in vc.iloc[:5].items()]}")

print("\n=== 5. 값 범위·고착(같은 값 6시간 이상 연속) 비율 ===")
def stuck(s):
    r = (s != s.shift()).cumsum(); ln = s.groupby(r).transform("size"); return ((ln >= 6) & s.notna()).mean()
for v in V:
    cells = []
    for g in GR:
        x = D.loc[D.grp == g]
        if x[v].notna().sum() == 0:
            cells.append(f"{g} 결측"); continue
        st_ = x.groupby(["farm", "day"])[v].apply(stuck).mean()
        cells.append(f"{g} 고착{st_:.2f} 결측{x[v].isna().mean():.2f}")
    print(f"{v:12s} " + "  ".join(cells))

print("\n=== 6. F13·F47 학습 날: 하루 거칠기 ~ EC 일평균 / DIAG10 하루 오차 (서술) ===")
WT = pd.read_csv(os.path.join(DC, "ec3_WT1_all.csv")); WT = WT[WT.validator == "DIAG10"]
cur = np.mean([np.clip(.8 * (.6 * WT["et_%d" % s] + .3 * WT["lgb_%d" % s] + .1 * WT["mlp_%d" % s]) + .2 * WT.pfn, WT.lo, WT.hi) for s in (47, 1414, 6464)], axis=0)
WT = WT.assign(cur=cur, e2=(cur - WT.sub_ec) ** 2); de = WT.groupby(["farm", "day"]).agg(ym=("sub_ec", "mean"), mse=("e2", "mean"), bias=("cur", "mean")).reset_index()
de["bias"] = de.bias - de.ym
T = DA[DA.grp.isin(["TR1", "TR2"])].merge(de, on=["farm", "day"])
print(f"날 {len(T)} (DIAG10 360일과 교집합)")
for v in V:
    c = T[v + "_m"]
    if c.notna().sum() < 50:
        continue
    r1 = spearmanr(c, T.ym, nan_policy="omit").correlation; r2 = spearmanr(c, T.mse, nan_policy="omit").correlation; r3 = spearmanr(c, T.bias, nan_policy="omit").correlation
    print(f"{v:12s} ρ(EC) {r1:+.2f}  ρ(하루 MSE) {r2:+.2f}  ρ(하루 편향) {r3:+.2f}")
DA.to_csv(os.path.join(R, "..", "results", "ec_hc1_day_change_v1.csv"), index=False)
print("저장 완료")
