# -*- coding: utf-8 -*-
"""재분석 검증 게이트 (2026-10-01, 집 클로드). analysis-verification.

harness 패널을 쓰지 않고 원 CSV + OOF npz(row_id 결합)로 핵심 수치를 독립 재계산하고,
다른 방법(스피어만, 비선형 GBM, 수동 루프)으로 교차 확인한다.
C1 SSE 하루 오프셋 비중 (DIAG10 56.6%, EXT10 60.2%)
C2 DIAG10 하루 오프셋 ~ 하루 입력 요약: 선형 R² -0.03 → 비선형 GBM, 오염 의심 날 제외로도 확인
C3 EXT10 하루 오프셋 ~ heat_mean r +0.405, in_temp_min r -0.398 → 스피어만
C4 DIAG10 하루 안 잔차 ~ dT1 r +0.012
C5 DIAG10 하루 오프셋 k=1 자기상관 +0.169 → 수동 루프
C6 다른 온실 추운 행 sub-in (F13 ~1.0, F47 ~0.7, 타 온실 대부분 >3)
결과: local/re05_verify.txt
"""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.join(env.LOCAL, "re05_verify.txt")
_f = open(OUT, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _f.write(s + "\n")


X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
D = pd.DataFrame({"row_id": z["row_id"]})
D = D.merge(X, on="row_id", how="left").merge(Y[["row_id", "sub_temp"]], on="row_id", how="left")
assert len(D) == 9600 and (D.row_id.values == z["row_id"]).all()
D["farm"] = D.row_id.str[:3]
D["day"] = D.row_id.str[4:7].astype(int)
D["hour"] = D.row_id.str[8:10].astype(int)
t = D.in_temp.values
gg = np.where(np.isnan(t), 1.0, np.clip((t - 8) / 2, 0, 1))
for s in ("DIAG10", "EXT10"):
    base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], axis=0)
    cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], axis=0)
    pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{s}.npy")).mean(0)
    D["e_" + s] = (0.6 - 0.2 * (1 - gg)) * base + (0.2 + 0.4 * (1 - gg)) * cx + 0.2 * gg * pfn - D.sub_temp

# ---------------- C1: 수동 루프로 날 평균 제거
p("C1 하루 오프셋 SSE 비중 (수동 루프)")
for s in ("DIAG10", "EXT10"):
    sse_all = sse_day = 0.0
    nday = 0
    for (fm, dy), g in D[D["e_" + s].notna()].groupby(["farm", "day"]):
        e = g["e_" + s].values
        sse_all += (e ** 2).sum()
        sse_day += len(e) * e.mean() ** 2
        nday += 1
    p(f"  {s}: 날 {nday}, 하루 오프셋 {100*sse_day/sse_all:.1f}%, 전체 RMSE {np.sqrt(sse_all/(nday*24)):.4f}")

# 하루 요약 (원 CSV에서 직접)
D["dio"] = D.in_temp - D.out_temp
agg = D.groupby(["farm", "day"]).agg(
    e_DIAG10=("e_DIAG10", "mean"), e_EXT10=("e_EXT10", "mean"),
    in_temp_mean=("in_temp", "mean"), in_temp_min=("in_temp", "min"), in_temp_max=("in_temp", "max"),
    out_temp_mean=("out_temp", "mean"), dio_mean=("dio", "mean"), in_hum_mean=("in_hum", "mean"),
    in_co2_mean=("in_co2", "mean"), rad_sum=("out_rad", "sum"), heat_mean=("act_heating", "mean"),
    vent_mean=("act_vent", "mean"), thermal_mean=("act_thermal", "mean"), circfan_mean=("act_circfan", "mean"),
    co2low=("in_co2", lambda s: (s < 300).sum()), n_nan=("in_temp", lambda s: s.isna().sum()),
).reset_index()
cols = ["in_temp_mean", "in_temp_min", "in_temp_max", "out_temp_mean", "dio_mean", "in_hum_mean",
        "in_co2_mean", "rad_sum", "heat_mean", "vent_mean", "thermal_mean", "circfan_mean", "day"]

# ---------------- C2: 비선형 GBM, 오염 의심 날 제외
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


def oof_r2(df, ycol, seed=0):
    Xm = df[cols].assign(f47=(df.farm == "F47").astype(float)).values
    y = df[ycol].values
    pr = np.zeros(len(y))
    for tr, te in KFold(10, shuffle=True, random_state=seed).split(Xm):
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05,
                                          min_samples_leaf=20, random_state=seed).fit(Xm[tr], y[tr])
        pr[te] = m.predict(Xm[te])
    return 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()


p("\nC2 DIAG10 하루 오프셋 ~ 하루 요약, 비선형 GBM 10겹 교차적합 R² (시드 3개)")
p("  전체 400일:", [round(oof_r2(agg, "e_DIAG10", s), 3) for s in (0, 1, 2)])
clean = agg[(agg.co2low == 0) & (agg.n_nan == 0)]
p(f"  CO2<300·결측 날 제외 {len(clean)}일:", [round(oof_r2(clean, "e_DIAG10", s), 3) for s in (0, 1, 2)])
p("  스피어만 |rho| 최대 (DIAG10, 온실 내):",
  max(abs(spearmanr(agg.loc[agg.farm == fm, c], agg.loc[agg.farm == fm, "e_DIAG10"], nan_policy="omit")[0])
      for c in cols for fm in ("F13", "F47")).__round__(3))
ext = agg[agg.e_EXT10.notna()]
p(f"  비교: EXT10 {len(ext)}일 GBM R²:", [round(oof_r2(ext, "e_EXT10", s), 3) for s in (0, 1, 2)])

# ---------------- C3: EXT10 스피어만
p("\nC3 EXT10 하루 오프셋 상관 (피어슨 전체 / 스피어만 전체 / 스피어만 F13, F47)")
for c in ("heat_mean", "in_temp_min", "in_temp_mean", "circfan_mean", "thermal_mean"):
    rp = np.corrcoef(ext[c], ext.e_EXT10)[0, 1]
    rs = spearmanr(ext[c], ext.e_EXT10)[0]
    rf = [spearmanr(ext.loc[ext.farm == fm, c], ext.loc[ext.farm == fm, "e_EXT10"])[0] for fm in ("F13", "F47")]
    p(f"  {c:13s} {rp:+.3f} / {rs:+.3f} / {rf[0]:+.3f}, {rf[1]:+.3f}")
# 같은 날들에서 DIAG10은?
p("  같은 85일, DIAG10 오프셋과 heat_mean 스피어만:", round(spearmanr(ext.heat_mean, ext.e_DIAG10)[0], 3))

# ---------------- C4: 하루 안 잔차 vs dT1, 수동
p("\nC4 DIAG10 하루 안 잔차 vs dT1")
D = D.sort_values(["farm", "day", "hour"])
D["e_in"] = D.e_DIAG10 - D.groupby(["farm", "day"]).e_DIAG10.transform("mean")
D["dT1"] = D.groupby(["farm", "day"]).in_temp.diff()
m = D[["e_in", "dT1"]].dropna()
p(f"  피어슨 {np.corrcoef(m.e_in, m.dT1)[0,1]:+.3f}, 스피어만 {spearmanr(m.e_in, m.dT1)[0]:+.3f}, n {len(m)}")

# ---------------- C5: 자기상관 수동 루프
p("\nC5 DIAG10 하루 오프셋 k=1 자기상관 (수동 사전 조회)")
ed = {(r.farm, r.day): r.e_DIAG10 for r in agg.itertuples()}
for k in (1, 2, 3):
    a, b = [], []
    for (fm, dy), v in ed.items():
        if (fm, dy - k) in ed:
            a.append(v)
            b.append(ed[(fm, dy - k)])
    p(f"  k={k}: n {len(a)} 피어슨 {np.corrcoef(a, b)[0,1]:+.3f} 스피어만 {spearmanr(a, b)[0]:+.3f}")

# ---------------- C6: 추운 행 sub-in 온실별 (원 CSV, 지연 없이)
p("\nC6 in_temp<8 행의 평균 sub_temp - in_temp (온실별)")
A = X[["row_id", "in_temp"]].merge(Y[["row_id", "sub_temp"]], on="row_id")
A["farm"] = A.row_id.str[:3]
A = A[(A.in_temp >= 0) & (A.sub_temp < 40)]
c = A[A.in_temp < 8].groupby("farm").apply(lambda g: pd.Series({"n": len(g), "d": (g.sub_temp - g.in_temp).mean()}))
c = c[c.n >= 30]
p(f"  F13 {c.loc['F13','d']:.2f} (n {int(c.loc['F13','n'])}), F47 {c.loc['F47','d']:.2f} (n {int(c.loc['F47','n'])})")
o = c.drop(["F13", "F47"])
p(f"  다른 온실 {len(o)}곳: 중앙 {o.d.median():.2f}, 최소 {o.d.min():.2f}, F13·F47보다 작은 온실 {int((o.d < 1.0).sum())}곳")
_f.close()
