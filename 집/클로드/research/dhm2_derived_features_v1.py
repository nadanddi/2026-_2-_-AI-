# -*- coding: utf-8 -*-
"""DHM2 (2026-10-16 집 클로드; 사용자 'AUC 조금 높은 것들끼리 섞어서 파생변수 만들어보면?').  서술 진단, 실행 전 커밋(기준 고정).  온도.
DHM1(6.998) 자료 그대로(어제 정보 없는 기록, 19회차 구성 최종 예측, 하루 평균 오차 m).  기록 번호(day)·2차 표시(pass2)는 대회 구조라 제외.
기본 특징: 외기 최고·최저, 밤 |실내−외기|, 난방 켠 비율·평균, CO2, 실내 최고·최저·변동, 환기, 커튼, 습도, 농장(F47).
파생(미리 정함, 물리적 이유): 추위 x 난방(= (15 − 외기 최고)+ x 난방 켠 비율), 밤 단열 부담(= 밤 |실내−외기| x 커튼), 환기 대비 CO2(= CO2 / (1 + 환기)),
  난방 효율 대리(= 밤 |실내−외기| / (1 + 난방 평균)), 하루 실내 범위 대비 외기 범위, F47 x 각 주요 특징(외기 최고·밤 차·난방 켠 비율).
모형(교차 5겹, 기록 단위로 묶어 같은 기록이 학습·검증에 동시에 들어가지 않게 GroupKFold): (a) 로지스틱(기본+파생) (b) 얕은 GBM(깊이 2, 100그루, 학습률 .05).
  과제 1 크기: HIT(|m| <= .2) vs MISS(|m| >= .8) AUC.  과제 2 방향: 모든 기록에서 m 회귀(교차 R²) + MISS 기록 부호 맞힘률.
기준: 크기 AUC >= .70 이면 '크기 구별'; 방향 R² > .05 이고 MISS 부호 맞힘 >= .70 이면 '방향 예측 가능(보정 후보)' — 아니면 점수 개선 경로 없음.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dhm2_derived_features_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(HERE, "dgx5_rows_v1.csv.gz")); R = R[R.st == "none"]; R["e"] = R.fin - R.y
m = R.groupby(["layout", "rec"]).e.mean().rename("m").reset_index()
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])]; X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:10].astype(int)
nt = X[X.h <= 5]
F = X.groupby("rec").agg(out_max=("out_temp", "max"), out_min=("out_temp", "min"), heat_on=("act_heating", lambda s: (s > 0).mean()), heat=("act_heating", "mean"), co2=("in_co2", "mean"),
                         t_max=("in_temp", "max"), t_min=("in_temp", "min"), t_sd=("in_temp", "std"), vent=("act_vent", "mean"), thermal=("act_thermal", "mean"), hum=("in_hum", "mean"))
F["night_gap"] = (nt.groupby("rec").in_temp.mean() - nt.groupby("rec").out_temp.mean()).abs(); F["f47"] = (F.index.str[:3] == "F47").astype(float)
F["cold_x_heat"] = np.clip(15 - F.out_max, 0, None) * F.heat_on; F["insul_load"] = F.night_gap * F.thermal / 100; F["co2_per_vent"] = F.co2 / (1 + F.vent)
F["heat_eff"] = F.night_gap / (1 + F.heat); F["range_ratio"] = (F.t_max - F.t_min) / (1 + F.out_max - F.out_min)
for c in ("out_max", "night_gap", "heat_on"): F[f"f47_x_{c}"] = F.f47 * F[c]
FEATS = list(F.columns); D = m.merge(F, left_on="rec", right_index=True, how="left"); D[FEATS] = D[FEATS].fillna(D[FEATS].median())
pd.set_option("display.width", 200)
S = D[(D.m.abs() <= .2) | (D.m.abs() >= .8)].reset_index(drop=True); yb = (S.m.abs() >= .8).astype(int).values; grp = S.rec.values
print(f"HIT {int((yb == 0).sum())} | MISS {int(yb.sum())} (records {S[yb == 1].rec.nunique()})")
single = sorted(((max(roc_auc_score(yb, S[f]), 1 - roc_auc_score(yb, S[f])), f) for f in FEATS), reverse=True)[:8]
print("single-feature |AUC| top:", [(f, round(a, 3)) for a, f in single])
def cv_cls(model):
    p = np.zeros(len(S))
    for tr, te in GroupKFold(5).split(S, yb, grp): p[te] = model.fit(S[FEATS].values[tr], yb[tr]).predict_proba(S[FEATS].values[te])[:, 1]
    return roc_auc_score(yb, p)
a_log = cv_cls(make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=3000))); a_gbm = cv_cls(GradientBoostingClassifier(max_depth=2, n_estimators=100, learning_rate=.05, random_state=0))
print(f"[task1 magnitude] CV AUC logistic {a_log:.3f} | GBM {a_gbm:.3f}")
yr = D.m.values; g2 = D.rec.values
def cv_reg(model):
    p = np.zeros(len(D))
    for tr, te in GroupKFold(5).split(D, yr, g2): p[te] = model.fit(D[FEATS].values[tr], yr[tr]).predict(D[FEATS].values[te])
    return p
res = {}
for nm, mdl in (("ridge", make_pipeline(StandardScaler(), Ridge(alpha=10))), ("gbm", GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate=.05, random_state=0))):
    p = cv_reg(mdl); r2 = 1 - ((yr - p) ** 2).sum() / ((yr - yr.mean()) ** 2).sum(); miss = np.abs(yr) >= .8; sign = float((np.sign(p[miss]) == np.sign(yr[miss])).mean())
    gain = 1 - np.sqrt(((yr - p) ** 2).mean()) / np.sqrt((yr ** 2).mean()); res[nm] = (r2, sign)
    print(f"[task2 direction] {nm}: CV R2 {r2:+.3f} | MISS sign hit {sign:.2f} ({int(miss.sum())}) | day-mean RMS if subtracted: {100 * (-gain):+.1f}%")
mag = max(a_log, a_gbm) >= .70; dirn = any(r2 > .05 and s >= .70 for r2, s in res.values())
print(f"DHM2: magnitude {'DETECTABLE' if mag else 'not detectable'} | direction {'PREDICTABLE (correction candidate)' if dirn else 'NOT predictable'}")
