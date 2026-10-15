# -*- coding: utf-8 -*-
"""DHM1 (2026-10-16 집 클로드; 사용자 '링커가 없어도 하루 수준을 맞추는 경우와 틀리는 경우의 차이가 있나?'; 데이터 생성 역추적 스킬).  서술 진단, 실행 전 커밋(가설·기준 아래 고정).  온도.
자료: dgx5_rows_v1.csv.gz (CV8·N·CK2·EL1, FL7 거울, 19회차 구성 최종 예측 fin), 어제 정보 없는 기록(st == none)만.  기록(배치 x 기록) 하루 평균 오차 m.
  맞힌 날 HIT: |m| <= .2 ,  크게 틀린 날 MISS: |m| >= .8  (그 사이 제외).  기존 6.662 은 '부호 포함 크기 회귀'(R² 음수) 만 봄 -> 여기선 '크기 구별'(방향 없이).
가설(실행 전 고정) | 맞다면 보여야 할 것 | 보이면 안 되는 것:
  H1 난방 전환일(난방 켜짐/꺼짐 섞인 날)  | MISS 에서 '부분 난방(0 < 켠 시간 비율 < 1)' 비율 높음        | 두 집단 비율 같음
  H2 실내 센서 이상(실내가 외기·평소와 동떨어짐) | MISS 에서 |밤 실내 − 밤 외기|·실내 하루 변동이 극단           | 분포 같음
  H3 계절·2차 구간                          | MISS 가 2차(day >= 179)·특정 계절에 몰림                    | 고르게 퍼짐
  H4 농장                                   | MISS 비율이 F13/F47 에서 다름                               | 같음
  H5 극한 날씨(매우 춥거나 더운 날·일사 큼)  | MISS 가 하루 최저/최고·외기 일사 극단에 몰림                 | 같음
  H6 보온커튼·환기 운영 특이               | MISS 에서 커튼·환기 하루 평균이 다름                          | 같음
기준: 한 특징이 HIT vs MISS 를 AUC >= .70 로 가르고, 여러 특징 합친 교차(5겹, 시드 0) 로지스틱 AUC >= .70 이면 '크기 구별 가능(비인과 상한)', 아니면 '구별 안 됨'.
  (그날 전체 입력 = 비인과 상한; 구별되더라도 방향 없이는 보정 불가 — 현장 '신뢰도 알림' 용도만.)
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dhm1_hit_vs_miss_days_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(HERE, "dgx5_rows_v1.csv.gz")); R = R[R.st == "none"]
R["e"] = R.fin - R.y; m = R.groupby(["layout", "rec"]).e.mean().rename("m").reset_index()
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])]; X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:10].astype(int)
night = X[X.h <= 5]
F = X.groupby("rec").agg(t_min=("in_temp", "min"), t_max=("in_temp", "max"), t_sd=("in_temp", "std"), out_min=("out_temp", "min"), out_max=("out_temp", "max"), rad=("out_rad", "sum"),
                         heat=("act_heating", "mean"), heat_on=("act_heating", lambda s: (s > 0).mean()), thermal=("act_thermal", "mean"), vent=("act_vent", "mean"),
                         hum=("in_hum", "mean"), co2=("in_co2", "mean"))
F["partial_heat"] = ((F.heat_on > 0) & (F.heat_on < 1)).astype(float)
F["night_in_minus_out"] = (night.groupby("rec").in_temp.mean() - night.groupby("rec").out_temp.mean()).abs()
F["day"] = F.index.str[4:7].astype(int); F["pass2"] = (F.day >= 179).astype(float); F["f47"] = (F.index.str[:3] == "F47").astype(float)
D = m.merge(F, left_on="rec", right_index=True, how="left")
D["cls"] = np.where(D.m.abs() <= .2, "HIT", np.where(D.m.abs() >= .8, "MISS", "mid")); S = D[D.cls != "mid"].copy(); yb = (S.cls == "MISS").astype(int).values
print(f"records(layout x rec) none-status {len(D)} | HIT {int((D.cls == 'HIT').sum())} | MISS {int((D.cls == 'MISS').sum())} | mid {int((D.cls == 'mid').sum())}")
print("  MISS records (unique):", sorted(set(S[S.cls == "MISS"].rec)))
FEATS = ["partial_heat", "heat_on", "heat", "night_in_minus_out", "t_sd", "pass2", "day", "f47", "t_min", "t_max", "out_min", "out_max", "rad", "thermal", "vent", "hum", "co2"]
rows = []
for f in FEATS:
    v = S[f].fillna(S[f].median()).values; a = roc_auc_score(yb, v) if len(set(v)) > 1 else .5
    rows.append(dict(feat=f, HIT_med=S[S.cls == "HIT"][f].median(), MISS_med=S[S.cls == "MISS"][f].median(), AUC=a, AUC_abs=max(a, 1 - a)))
T = pd.DataFrame(rows).sort_values("AUC_abs", ascending=False); pd.set_option("display.width", 200); print(T.round(3).to_string(index=False))
Z = S[FEATS].fillna(S[FEATS].median()).values; pr = np.zeros(len(S))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, yb):
    pr[te] = make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=2000)).fit(Z[tr], yb[tr]).predict_proba(Z[te])[:, 1]
auc = roc_auc_score(yb, pr); best = T.AUC_abs.iloc[0]
print(f"DHM1: best single-feature AUC {best:.3f} ({T.feat.iloc[0]}) | multivariate CV AUC {auc:.3f} -> {'magnitude DETECTABLE (upper bound)' if best >= .70 and auc >= .70 else 'NOT detectable'}")
print("  sign of miss offsets (records):", S[S.cls == "MISS"].groupby("rec").m.mean().round(2).to_dict())
