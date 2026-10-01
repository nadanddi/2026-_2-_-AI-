# -*- coding: utf-8 -*-
"""재분석 23 (2026-10-01, 집 클로드): 문헌 기반 흔적 T1~T8 검증 — re23_lit_fingerprints.md(실행 전 고정)대로.
결과 local/re23_lit_fingerprints.txt"""
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, rankdata, t as Tdist
from sklearn.metrics import roc_auc_score

out = open(env.LOCAL + "/re23_lit_fingerprints.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
THR = 0.05 / 8
G = pd.read_csv("re17_day_offsets.csv")
X = pd.read_csv(env.DATA + "/train_X.csv"); Y = pd.read_csv(env.DATA + "/train_y.csv")
X = X[X.row_id.str[:3].isin(["F13", "F47"])].merge(Y[["row_id", "sub_temp", "sub_ec"]], on="row_id")
k_ = X.row_id.str.extract(r"^(F\d+)_(\d+)_(\d+)$")
X["farm"], X["day"], X["hour"] = k_[0], k_[1].astype(int), k_[2].astype(int)
X = X.sort_values(["farm", "day", "hour"])
rows = []
for (f, d), q in X.groupby(["farm", "day"]):
    q = q.set_index("hour").reindex(range(24))
    s, a = q.sub_temp.values, q.in_temp.values
    n = slice(0, 7)
    ds = np.diff(s)
    drop = s[:-1] - s[1:]                       # 시간당 강하(양수=식음)
    pred = -0.27 + 0.13 * (s[:-1] - a[:-1])
    m = np.zeros(23, bool); m[:6] = True; m[19:] = True   # 밤 전이 (0→1..5→6, 19→20..22→23)
    m &= (s[:-1] > a[:-1]) & ~np.isnan(drop) & ~np.isnan(pred)
    S_hours = [5, 8, 10, 20, 22]
    absd = np.abs(ds)
    idx_s = [h - 1 for h in S_hours if 1 <= h <= 23]
    other = [i for i in range(23) if i not in idx_s]
    day_rise_s = np.nanmax(s[9:17]) - s[8]
    day_rise_a = np.nanmax(a[9:17]) - a[8]
    rows.append(dict(farm=f, day=d,
                     n_std=np.nanstd(s[n]), n_gap=np.nanmean(s[n] - a[n]),
                     decay_res=np.nanmedian((drop - pred)[m]) if m.sum() >= 3 else np.nan,
                     step_ratio=np.nanmean(absd[idx_s]) / max(np.nanmean(absd[other]), 1e-3),
                     rise_ratio=day_rise_s / day_rise_a if day_rise_a > 0.5 else np.nan,
                     gap_mean=np.nanmean(s - a), n_heat=np.nanmean(q.act_heating.values[n]),
                     n_air=np.nanmean(a[n]), n_out=np.nanmean(q.out_temp.values[n]), out_mean=np.nanmean(q.out_temp.values),
                     ec=np.nanmean(q.sub_ec.values),
                     sig=tuple(np.round(q[["out_temp", "out_hum", "out_rad", "out_wspd"]].values.ravel(), 1))))
R = pd.DataFrame(rows).merge(G[["farm", "day", "temp_offset", "group"]], on=["farm", "day"])
grp = lambda g: R[R.group == g]

def mw(col, g1, alt):
    a, b = grp(g1)[col].dropna(), grp("normal")[col].dropna()
    return a.median(), b.median(), mannwhitneyu(a, b, alternative=alt).pvalue, len(a), len(b)

res = {}
# T1
m1, m0, p1, n1, n0 = mw("n_std", "under", "less"); g1, g0, p2, _, _ = mw("n_gap", "under", "greater")
ok = (m1 <= 0.7 * m0) and p1 < THR and (g1 >= g0 + 2) and p2 < THR
p(f"T1 근권가온(under): 밤 배지 표준편차 {m1:.3f} vs {m0:.3f} (p {p1:.4f}) | 밤 배지−공기 {g1:+.2f} vs {g0:+.2f} (p {p2:.4f}) n {n1}/{n0} → {'합격' if ok else '불합격'}")
# T2
m1, m0, pv, n1, n0 = mw("decay_res", "under", "less")
ok = (m1 <= m0 - 0.15) and pv < THR
mo, _, pvo, no, _ = mw("decay_res", "over", "two-sided")
p(f"T2 숨은 열원(under): 밤 강하 잔차 {m1:+.3f} vs normal {m0:+.3f} ℃/h (p {pv:.4f}, n {n1}/{n0}) → {'합격' if ok else '불합격'} | 참고 over {mo:+.3f} (p {pvo:.4f}); normal 잔차 크기 = 모델 이식성")
# T3
m1, m0, pv, n1, n0 = mw("step_ratio", "under", "greater")
ok = (m1 >= 1.5 * m0) and pv < THR
p(f"T3 가온 시각 계단(under): 비 {m1:.3f} vs {m0:.3f} (p {pv:.4f}) → {'합격' if ok else '불합격'}")
# T4 (입력만): 정상 날로 온실별 선형식 적합 → 잔차
for c in ("n_heat", "n_air"):
    R[c + "_res"] = np.nan
    for f in ("F13", "F47"):
        nm = (R.farm == f) & (R.group == "normal") & R[c].notna() & R.n_out.notna()
        coef = np.polyfit(R.loc[nm, "n_out"], R.loc[nm, c], 1)
        fm = (R.farm == f)
        R.loc[fm, c + "_res"] = R.loc[fm, c] - np.polyval(coef, R.loc[fm, "n_out"])
h1, h0, ph, _, _ = mw("n_heat_res", "under", "less"); a1, a0, pa, n1, n0 = mw("n_air_res", "under", "less")
sub = R[R.group.isin(["under", "normal"])].dropna(subset=["n_heat_res", "n_air_res"])
yb = (sub.group == "under").astype(int)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
Z = sub[["n_heat_res", "n_air_res"]].values
pr = cross_val_predict(LogisticRegression(), Z, yb, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
auc = roc_auc_score(yb, pr)
ok = h1 < 0 and a1 < 0 and ph < THR and pa < THR and auc >= 0.70
p(f"T4 공간난방 낮음(입력만, under): 밤 난방 잔차 {h1:+.2f} vs {h0:+.2f} (p {ph:.4f}) | 밤 기온 잔차 {a1:+.2f} vs {a0:+.2f} (p {pa:.4f}) | 교차적합 AUC {auc:.3f} (n {n1}/{n0}) → {'합격' if ok else '불합격'}")
# T5
m1, m0, pv, n1, n0 = mw("rise_ratio", "over", "less")
ok = (m1 <= 0.7 * m0) and pv < THR
p(f"T5 찬 양액(over): 낮 배지/공기 상승비 {m1:.3f} vs {m0:.3f} (p {pv:.4f}, n {n1}/{n0}) → {'합격' if ok else '불합격'}")
# T6 부분 순위상관
def pr_(x, y, z):
    mm = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    rx, ry, rz = rankdata(x[mm]), rankdata(y[mm]), rankdata(z[mm])
    Zm = np.c_[np.ones(mm.sum()), rz]
    ex = rx - Zm @ np.linalg.lstsq(Zm, rx, rcond=None)[0]; ey = ry - Zm @ np.linalg.lstsq(Zm, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]; n = mm.sum(); tt = r * np.sqrt((n - 3) / (1 - r * r))
    return r, 2 * Tdist.sf(abs(tt), n - 3)
rr = [pr_(R.loc[R.farm == f, "n_heat"].values, R.loc[R.farm == f, "gap_mean"].values, R.loc[R.farm == f, "out_mean"].values) for f in ("F13", "F47")]
ok = all(r <= -0.20 and pv < THR for r, pv in rr)
p(f"T6 난방↑ 배지<공기: 부분ρ F13 {rr[0][0]:+.3f} (p {rr[0][1]:.4f}) F47 {rr[1][0]:+.3f} (p {rr[1][1]:.4f}) → {'합격' if ok else '불합격'}")
# T7 같은 날짜 F13·F47 동시 분리
by = {}
for r in R.itertuples():
    by.setdefault(r.sig, []).append(r)
both = one = none = 0
for sgn, lst in by.items():
    f13 = [r for r in lst if r.farm == "F13"]; f47 = [r for r in lst if r.farm == "F47"]
    for a in f13:
        for b in f47:
            da, db = abs(a.temp_offset) > 0.7, abs(b.temp_offset) > 0.7
            both += da and db; one += da != db; none += (not da) and (not db)
npair = both + one + none
pa_ = R[R.farm == "F13"].temp_offset.abs().gt(0.7).mean(); pb_ = R[R.farm == "F47"].temp_offset.abs().gt(0.7).mean()
exp = npair * pa_ * pb_
odds, pf = fisher_exact([[both, one], [one, none]], alternative="greater") if npair else (np.nan, np.nan)
ok = npair > 0 and both >= 2 * exp and pf < THR
p(f"T7 동시 분리: 같은 날짜 F13×F47 짝 {npair}, 둘 다 분리 {both} (기대 {exp:.2f}), Fisher p {pf:.4f} → {'합격' if ok else '불합격'}")
# T8
o_low = (grp("over").ec < 0.15).sum(); o_n = grp("over").ec.notna().sum()
n_low = (grp("normal").ec < 0.15).sum(); n_n = grp("normal").ec.notna().sum()
_, pf8 = fisher_exact([[o_low, o_n - o_low], [n_low, n_n - n_low]], alternative="greater")
ok = (o_low / o_n) >= 3 * max(n_low / n_n, 1e-9) and pf8 < THR
p(f"T8 센서 이탈(over): EC<0.15 비율 {o_low}/{o_n} vs {n_low}/{n_n} (p {pf8:.4f}) → {'합격' if ok else '불합격'} | EC 중앙 over {grp('over').ec.median():.3f} under {grp('under').ec.median():.3f} normal {grp('normal').ec.median():.3f}")
R.drop(columns=["sig"]).to_csv(env.LOCAL + "/re23_day_fingerprints.csv", index=False)
