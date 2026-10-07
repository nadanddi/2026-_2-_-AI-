# -*- coding: utf-8 -*-
"""RF1 — 시각별 인과 '하루 잔차 보정기' 전제 검산 (실행 전 고정) — 2026-10-07 연구실 클로드
RF0b: 원래 운영 지문(하루 전체) ExtraTrees 가 DIAG10 OOF(R3S+DP1) 하루 잔차를 R² .172 설명(2차 −18%) — 하루 전체 = 미래 시각 포함.
규정판: h 시 행 특징 = 0..h 시 지문(구동기 7 누적평균·0 비율, 실내 3 누적평균, 외기 4 누적평균) + h. 목표 = 그 날 하루 잔차(y_day − p_day; 학습 정답).
모델: ExtraTrees(300, min_samples_leaf 20, max_features .5, seed 0), 날짜 묶음 단위 5분할 교차적합(같은 날짜 묶음은 같은 폴드).
보정: pred_h = p_h + r̂_h. 기준: p_h = DIAG10 OOF R3S+DP1 시드평균(행 단위).
보고: 시간 행 RMSE 전체 / 2차 / 2차 일반 / 고EC, 시각대별(0–4, 5–11, 12–23), 시드 3 각각(dp_7/101/2024 를 기준으로 같은 보정기 재적합).
[관문(고정)] 시드 3 모두 전체·2차 시간 RMSE 개선, 2차 일반 날 악화 ≤ 1% → 계획 단계로. 아니면 종료.
비고: 학습 행 특징이 평가 행과 같은 정의(같은 온실 0..h 입력)이고 MASK 열 없음. 교차적합은 날짜 묶음 기준이라 같은 날짜 다른 기록 정답 누수 차단.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]; INDOOR = ["in_temp", "in_hum", "in_co2"]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TR = TR[TR.row_id.str[:3].isin(["F13", "F47"])].copy()
TR["farm"], TR["day"], TR["hour"] = TR.row_id.str[:3], TR.row_id.str[4:7].astype(int), TR.row_id.str[8:10].astype(int)
TR = TR.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
gg = TR.groupby(["farm", "day"])
feat = pd.DataFrame({"row_id": TR.row_id, "hour": TR.hour})
for v in ACTS:
    feat[v + "_tdm"] = gg[v].transform(lambda s: s.expanding().mean()).values
    feat[v + "_tdz"] = gg[v].transform(lambda s: (s == 0).astype(float).where(s.notna()).expanding().mean()).values
for v in INDOOR + W:
    feat[v + "_tdm"] = gg[v].transform(lambda s: s.expanding().mean()).values
FC = [c for c in feat.columns if c != "row_id"]
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); DP = DP[DP.validator == "DIAG10"]
D = DP.merge(feat.drop(columns="hour"), on="row_id")
D["key"] = list(zip(D.farm, D.day))
WV = TR.pivot_table(index=["farm", "day"], columns="hour", values=W); WZ = (WV - WV.mean()) / WV.std()
keys = list(WZ.index); A = WZ.values; Dm = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
par = list(range(len(keys)))
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for i in range(len(keys)):
    for j in np.where(Dm[i] <= .05)[0]:
        par[fd(i)] = fd(j)
grp = {k: fd(i) for i, k in enumerate(keys)}
ug = sorted(set(grp.values())); rng = np.random.default_rng(0); fmap = {g_: i % 5 for i, g_ in enumerate(rng.permutation(ug))}
D["fold"] = [fmap[grp[k]] for k in D.key]
D["yd"] = D.groupby(["farm", "day"]).sub_ec.transform("mean")
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
ok = True
for s in (7, 101, 2024):
    p = D["dp_%d" % s]; pdm = p.groupby([D.farm, D.day]).transform("mean"); target = D.yd - pdm
    rhat = np.zeros(len(D))
    for k in range(5):
        tr, te = D.fold != k, D.fold == k
        m = ExtraTreesRegressor(300, min_samples_leaf=20, max_features=.5, random_state=0, n_jobs=4).fit(D.loc[tr, FC].fillna(-9), target[tr])
        rhat[te.values] = m.predict(D.loc[te, FC].fillna(-9))
    pr = p + rhat
    out = []
    for nm, msk in (("전체", D.day > 0), ("2차", D.day >= 179), ("2차 일반", (D.day >= 179) & (D.yd < 1)), ("고EC", D.yd >= 1),
                    ("0-4시", D.hour <= 4), ("5-11시", (D.hour >= 5) & (D.hour <= 11)), ("12-23시", D.hour >= 12)):
        a, b = rm((p - D.sub_ec)[msk]), rm((pr - D.sub_ec)[msk]); out.append("%s %.4f→%.4f (%+.1f%%)" % (nm, a, b, 100 * (b / a - 1)))
        if nm in ("전체", "2차"):
            ok &= b < a
        if nm == "2차 일반":
            ok &= b <= 1.01 * a
    print("시드 %4d | %s" % (s, " | ".join(out)), flush=True)
print("[RF1 관문] %s" % ("통과 → 계획 단계로" if ok else "종료"))
