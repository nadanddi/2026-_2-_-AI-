# -*- coding: utf-8 -*-
"""CO2R — CO₂ 가 하루 동안 유난히 거칠게 움직이는 날 확인 (사용자 질문 10-08: 시간당 변화폭 중앙값 21.6 vs 4.8?) — 서술
하루 지표: 같은 날 1시간 |Δin_co2| 중앙값(h=1..23). 분포·집단(TR1/TR2/TE)·CO₂ 공급(act_co2>0 시간)·밀폐 날(6.11)·EC·DIAG10 오차와의 관계."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
R = os.path.dirname(os.path.abspath(__file__)); DC = os.path.join(R, "..", "local", "drive_copy")
def load(fn):
    d = pd.read_csv(os.path.join(env.DATA, fn)); d["farm"] = d.row_id.str[:3]; d["day"] = d.row_id.str[4:7].astype(int); d["hour"] = d.row_id.str[8:10].astype(int); return d
TR = load("train_X.csv").merge(pd.read_csv(os.path.join(env.DATA, "train_y.csv")), on="row_id", how="left"); TE = load("test_X.csv")
TR = TR[TR.farm.isin(["F13", "F47"]) & TR.sub_ec.notna()]; TR["grp"] = np.where(TR.day >= 179, "TR2", "TR1"); TE["grp"] = "TE"
D = pd.concat([TR, TE]).sort_values(["farm", "day", "hour"]); D["dc"] = D.groupby(["farm", "day"]).in_co2.diff().abs()
D["dose"] = (D.act_co2 > 0).astype(bool)
def agg(g):
    x = g.dc.iloc[1:]; dz = g.dose.astype(bool).values; keep = ~(dz[1:] | dz[:-1]); nd = x[keep]
    return pd.Series({"med": x.median(), "med_nodose": nd.median() if len(nd) >= 6 else np.nan, "night_med": g[(g.hour >= 1) & (g.hour <= 5)].dc.median(),
                      "dose_h": g.dose.sum(), "vent0": (g.act_vent == 0).mean(), "fan": g.act_circfan.mean(), "co2m": g.in_co2.mean(), "ec": g.sub_ec.mean()})
A = D.groupby(["farm", "day", "grp"]).apply(agg, include_groups=False).reset_index()
A["sealed"] = (A.fan < 10) & (A.vent0 > .85)
print("하루 |ΔCO₂| 중앙값 분포(집단별 분위 10/25/50/75/90/97):")
for g in ["TR1", "TR2", "TE"]:
    print(f"  {g}: n {int((A.grp==g).sum())}", np.round(A.loc[A.grp == g, "med"].quantile([.1, .25, .5, .75, .9, .97]).values, 1))
h, e = np.histogram(A.med, bins=[0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 200]); print("히스토그램:", dict(zip([f"{a}-{b}" for a, b in zip(e[:-1], e[1:])], h)))
# 거친 날 정의 후보: 학습 1차 분포 상위 10%
thr = A.loc[A.grp == "TR1", "med"].quantile(.9); A["rough"] = A.med > thr
print(f"\n거친 날(학습 1차 상위 10%, 문턱 {thr:.1f}): 집단별 비율", A.groupby("grp").rough.mean().round(3).to_dict())
print(f"거친 날 중앙값의 중앙값 {A[A.rough].med.median():.1f} vs 나머지 {A[~A.rough].med.median():.1f} (비 {A[A.rough].med.median()/A[~A.rough].med.median():.1f})")
for c in ["med_nodose", "night_med", "dose_h", "sealed", "co2m", "ec"]:
    a, b = A.loc[A.rough, c].astype(float), A.loc[~A.rough, c].astype(float)
    print(f"  {c:10s} 거친 {a.mean():8.2f} (중앙 {a.median():.2f}) | 나머지 {b.mean():8.2f} (중앙 {b.median():.2f})")
print("\n공급 없는 시각만의 |ΔCO₂| 중앙값 기준 거친 날:")
thr2 = A.loc[A.grp == "TR1", "med_nodose"].quantile(.9); A["rough2"] = A.med_nodose > thr2
print(f"  문턱 {thr2:.1f}, 집단별 비율", A.groupby("grp").rough2.mean().round(3).to_dict(), "| 둘 다 거친 날 비율(학습)", round((A.rough & A.rough2)[A.grp != "TE"].sum() / max(A.rough[A.grp != "TE"].sum(), 1), 2))
WT = pd.read_csv(os.path.join(DC, "ec3_WT1_all.csv")); WT = WT[WT.validator == "DIAG10"]
cur = np.mean([np.clip(.8 * (.6 * WT["et_%d" % s] + .3 * WT["lgb_%d" % s] + .1 * WT["mlp_%d" % s]) + .2 * WT.pfn, WT.lo, WT.hi) for s in (47, 1414, 6464)], axis=0)
de = WT.assign(c=cur, e2=(cur - WT.sub_ec) ** 2, r=cur - WT.sub_ec).groupby(["farm", "day"]).agg(mse=("e2", "mean"), bias=("r", "mean")).reset_index()
T = A.merge(de, on=["farm", "day"])
print(f"\nDIAG10 360일: 거친 날 {int(T.rough.sum())}일 하루 MSE 평균 {T[T.rough].mse.mean():.3f} vs {T[~T.rough].mse.mean():.3f} (MW p {mannwhitneyu(T[T.rough].mse, T[~T.rough].mse).pvalue:.3g}); 편향 {T[T.rough].bias.mean():+.3f} vs {T[~T.rough].bias.mean():+.3f}")
for s in [False, True]:
    t = T[T.sealed == s]
    if t.rough.sum() >= 5:
        print(f"  밀폐={s}: 거친 {int(t.rough.sum())}일 MSE {t[t.rough].mse.mean():.3f} 편향 {t[t.rough].bias.mean():+.3f} | 나머지 {int((~t.rough).sum())}일 MSE {t[~t.rough].mse.mean():.3f} 편향 {t[~t.rough].bias.mean():+.3f}")
print("ρ(하루 |ΔCO₂| 중앙값, EC)", round(spearmanr(T.med, T.ec).correlation, 3), " ρ(·, MSE)", round(spearmanr(T.med, T.mse).correlation, 3), " ρ(·, 편향)", round(spearmanr(T.med, T.bias).correlation, 3))
print("\n가장 거친 10일:", A.sort_values("med", ascending=False).head(10)[["farm", "day", "grp", "med", "med_nodose", "dose_h", "sealed", "ec"]].round(2).values.tolist())
print("평가 거친 날:", A[(A.grp == "TE") & A.rough][["farm", "day", "med", "dose_h", "sealed"]].round(1).values.tolist())
A.to_csv(os.path.join(R, "..", "results", "ec_co2r_days_v1.csv"), index=False)
