# -*- coding: utf-8 -*-
"""VW3 (verify_water) — 후속 점검 (2026-10-13 집 클로드). VW1·VW2 결과를 본 뒤 정한 점검 → '사후' 표시.
 (a) dah_closed_day: 정의에 들어간 밀폐 낮 시간 수(qualifying hours)와 얽혔는지, 영향 큰 두 기록(F47_151, F13_177)이 어느 에피소드인지.
 (b) VW2(3) 하루 안 결합(시간당 ΔEC 와 시간당 증산 대리): 정상 기록 +.48, 고EC 기록 ~0.
     - 같은 날짜 쌍둥이(실외 동일) 정상 동과 짝지어 비교 → 날씨(흐린 날이라 증산 변동 작음) 탓인지 동 상태 탓인지.
     - 현재 모델 sg 의 시간당 Δ 도 같은 식으로 → 모델이 고EC 날에 '정상형 하루 모양'을 그리는지 (진단 전용, 정답 사용).
     - 에피소드 부트스트랩 CI, 온실·차수별.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
rng = np.random.default_rng(51013)
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HX9, "water", "vars_v1.csv")).set_index("rec").loc[R.index]
R = R.join(F)
H = pd.read_csv(os.path.join(HX9, "data", "hourly.csv")).sort_values(["rec", "hour"])

def es(T): return 0.6108 * np.exp(17.27 * T / (T + 237.3))
def ahum(T, RH): return 2165.0 * (np.clip(RH, 1, 100) / 100.0) * es(T) / (T + 273.15)
HRS = np.arange(24); DAY = (HRS >= 9) & (HRS <= 16); W = (HRS >= 7) & (HRS <= 18)
V = ["out_rad", "in_temp", "in_hum", "act_vent", "act_shade", "act_thermal", "act_fog", "sub_ec", "sg"]
rows = []
for rec, g in H.groupby("rec"):
    g = g.set_index("hour").reindex(range(24))
    a = {v: g[v].interpolate(limit_direction="both").values.astype(float) for v in V}
    rad = np.clip(a["out_rad"] - 5, 0, None)
    radIn = rad * 0.65 * (1 - 0.6 * a["act_shade"] / 100) * (1 - 0.4 * a["act_thermal"] / 100)
    vpd = es(a["in_temp"]) * (1 - np.clip(a["in_hum"], 0, 100) / 100)
    E = np.where(rad > 10, 0.30 * radIn + 20 * vpd, 10 * vpd)
    mm = DAY & (a["act_vent"] < 5) & (a["act_fog"] <= 0)
    d = dict(rec=rec, q_hours=int(mm.sum()), E_sd=np.std(E[W]))
    for lab in ("sub_ec", "sg"):
        dv = np.r_[np.nan, np.diff(a[lab])]
        d["r_" + lab] = np.corrcoef(dv[W], E[W])[0, 1] if np.isfinite(dv[W]).all() and np.std(dv[W]) > 0 and np.std(E[W]) > 0 else np.nan
    rows.append(d)
Q = pd.DataFrame(rows).set_index("rec"); R = R.join(Q)

print("=== (a) dah_closed_day vs qualifying closed-day hours (hi10 with OOF) ===")
HI2 = R[(R.hi10 == 1) & (R.has_sg == 1)]
d = HI2[["dah_closed_day", "q_hours", "resid_mean", "ep_id", "farm", "pass2", "ec_mean", "sg_mean"]].dropna()
print(f"  r(dah, q_hours) {d[['dah_closed_day','q_hours']].corr().iloc[0,1]:+.3f}; r(q_hours, resid) {d[['q_hours','resid_mean']].corr().iloc[0,1]:+.3f}")
Z = np.c_[np.ones(len(d)), d.q_hours]
ex = d.dah_closed_day - Z @ np.linalg.lstsq(Z, d.dah_closed_day, rcond=None)[0]; ey = d.resid_mean - Z @ np.linalg.lstsq(Z, d.resid_mean, rcond=None)[0]
print(f"  partial r(dah, resid | q_hours) {np.corrcoef(ex, ey)[0,1]:+.3f}")
print("  most influential records:"); print(d.loc[[r for r in ["F47_151", "F13_177"] if r in d.index]].round(3).to_string())
print("  q_hours distribution hi10:", d.q_hours.value_counts().sort_index().to_dict())

print("\n=== (b) within-day coupling r(hourly dEC, hourly ET proxy), 07-18h ===")
def ep_ci(s, col, gcol="ep_id", nb=2000):
    s = s[[col, gcol]].dropna(); ug = s[gcol].unique(); idx = {u: np.where(s[gcol].values == u)[0] for u in ug}
    out = [s[col].values[np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])].mean() for _ in range(nb)]
    return np.percentile(out, 2.5), np.percentile(out, 97.5)
hi = R[R.hi10 == 1]; no = R[R.hi10 == 0]
for col in ("r_sub_ec", "r_sg"):
    lo, up = ep_ci(hi, col)
    print(f"  {col:9s}: hi10 mean {hi[col].mean():+.3f} (episode CI [{lo:+.2f},{up:+.2f}], n {hi[col].notna().sum()}) | normal mean {no[col].mean():+.3f} (n {no[col].notna().sum()})")
for f_ in ("F13", "F47"):
    print(f"  truth coupling {f_}: hi10 {hi[hi.farm==f_].r_sub_ec.mean():+.3f} (n {int((hi.farm==f_).sum())}) | normal {no[no.farm==f_].r_sub_ec.mean():+.3f}")
for p in (0, 1):
    print(f"  truth coupling pass{p+1}: hi10 {hi[hi.pass2==p].r_sub_ec.mean():+.3f} (n {int((hi.pass2==p).sum())}) | normal {no[no.pass2==p].r_sub_ec.mean():+.3f}")
print(f"  ET proxy hourly sd (07-18h): hi10 {hi.E_sd.mean():.1f} vs normal {no.E_sd.mean():.1f}")
# same-date twins: hi10 record vs its normal twins (same outdoor)
pairs = []
for r in hi.index:
    tw = R.at[r, "twin_recs"]
    if not isinstance(tw, str): continue
    for t in tw.split(";"):
        if t in R.index and R.at[t, "hi10"] == 0:
            pairs.append(dict(hi=r, tw=t, ep=R.at[r, "ep_id"], r_hi=R.at[r, "r_sub_ec"], r_tw=R.at[t, "r_sub_ec"], Esd_hi=R.at[r, "E_sd"], Esd_tw=R.at[t, "E_sd"]))
P = pd.DataFrame(pairs).dropna(subset=["r_hi", "r_tw"])
P["diff"] = P.r_hi - P.r_tw
lo, up = ep_ci(P, "diff", "ep")
print(f"  same-date twin pairs (hi10 vs normal, same outdoor): n {len(P)} pairs / {P.hi.nunique()} hi10 recs; "
      f"coupling hi10 {P.r_hi.mean():+.3f} vs twin {P.r_tw.mean():+.3f}; diff {P['diff'].mean():+.3f} (episode CI [{lo:+.2f},{up:+.2f}]); "
      f"ET sd hi10 {P.Esd_hi.mean():.1f} vs twin {P.Esd_tw.mean():.1f}")
# control for ET variability: normal records with E_sd in the hi10 range
q = no[(no.E_sd >= hi.E_sd.quantile(.1)) & (no.E_sd <= hi.E_sd.quantile(.9))]
print(f"  normal records with ET sd inside hi10 10-90% range: n {len(q)}, coupling {q.r_sub_ec.mean():+.3f}")
# does model residual coupling matter for the daily-mean target? (hourly shape vs level)
print(f"  r(truth coupling, resid_mean) hi10: {hi[['r_sub_ec','resid_mean']].dropna().corr().iloc[0,1]:+.3f}")
