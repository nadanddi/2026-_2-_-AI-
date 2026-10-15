# -*- coding: utf-8 -*-
"""DHM3 (2026-10-16 집 클로드; 사용자 '외기 최고 ~12 & 밤 차 작음 & 난방 비율 낮음 같은 조건 조합(구간 규칙)도 확인했나?').  서술 진단, 실행 전 커밋(검색 범위·기준 고정).  온도.
DHM2 와 같은 자료(어제 정보 없는 기록, 19회차 구성 최종 예측 하루 평균 오차 m; 기록 번호 제외 특징 12개 + 파생 9개 = DHM2 FEATS).
구간 규칙 = 특징 1~3개의 조건 AND, 각 조건 = (특징 <= 분위 q) 또는 (특징 >= 분위 q), q ∈ {.2, .4, .6, .8} (전체 자료 분위), 지지 >= 8 기록-행.
과제 A 크기: 규칙 안 MISS(|m| >= .8) 비율 / 전체 MISS 비율 = 향상도(lift).  과제 B 방향: 규칙 안 m 평균의 t 값(|t| 최대 = 한쪽으로 쏠린 구간).
검증 (1) 우연 기준: 기록 단위로 MISS 표시(또는 m)를 섞은 자료 200번에 같은 전수 검색 -> '섞은 자료 최고값' 분포의 95% 분위와 비교
     (2) 기록 단위 GroupKFold 5: 학습 기록에서 최고 규칙 찾기 -> 검증 기록에서 그 규칙의 lift / m 평균(방향 맞음 여부).
기준: 실제 최고 > 섞은 자료 95% 분위 AND 교차 검증에서 5겹 중 4겹 이상 같은 방향 효과(lift > 1.5 또는 m 평균 부호 일치·|평균| >= .3) -> '진짜 구간 조건 후보', 아니면 '우연 수준'.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dhm3_rule_subgroup_search_v1.py
"""
import os, itertools
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "dhm2_derived_features_v1.py"), encoding="utf-8").read(); exec(src[src.index("R = pd.read_csv"):src.index("pd.set_option")])
D = D.reset_index(drop=True); M_ = (D.m.abs() >= .8).values; Y = D.m.values; REC = D.rec.values
QS = (.2, .4, .6, .8); CONDS = []
for f in FEATS:
    v = D[f].values
    for q in QS:
        t = np.quantile(v, q); CONDS.append((f"{f}<={t:.2f}", v <= t)); CONDS.append((f"{f}>={t:.2f}", v >= t))
CM = np.array([c[1] for c in CONDS]); NAMES = [c[0] for c in CONDS]; NC = len(CONDS)
pairs = [(i, j) for i in range(NC) for j in range(i + 1, NC) if NAMES[i].split("<")[0].split(">")[0] != NAMES[j].split("<")[0].split(">")[0]]
rng = np.random.default_rng(0); trip_idx = [tuple(sorted(rng.choice(NC, 3, replace=False))) for _ in range(30000)]
RULES = [(i,) for i in range(NC)] + pairs + [t for t in trip_idx if len({NAMES[k].split("<")[0].split(">")[0] for k in t}) == 3]
MASKS = np.array([np.logical_and.reduce(CM[list(r)]) for r in RULES]); SUP = MASKS.sum(1); OK = SUP >= 8; MASKS, RULES, SUP = MASKS[OK], [r for r, o in zip(RULES, OK) if o], SUP[OK]
print(f"rules searched {len(RULES)} (conditions {NC}) | rows {len(D)} | MISS rows {int(M_.sum())} (base rate {M_.mean():.3f})")
def best_lift(mk):
    rate = (MASKS & mk).sum(1) / SUP; k = int(np.argmax(rate)); return rate[k] / max(mk.mean(), 1e-9), k
def best_t(y):
    mu = (MASKS * y).sum(1) / SUP; var = (MASKS * (y - mu[:, None]) ** 2).sum(1) / np.maximum(SUP - 1, 1); t = mu / np.sqrt(var / SUP + 1e-12); k = int(np.argmax(np.abs(t))); return abs(t[k]), k, mu[k]
L0, kL = best_lift(M_); T0, kT, mu0 = best_t(Y)
desc = lambda k: " AND ".join(NAMES[i] for i in RULES[k])
print(f"[A magnitude] best lift {L0:.2f} | support {SUP[kL]} | rule: {desc(kL)}")
print(f"[B direction] best |t| {T0:.2f} | mean m {mu0:+.2f} | support {SUP[kT]} | rule: {desc(kT)}")
recs = np.unique(REC); recmiss = pd.Series(M_).groupby(REC).max(); nullL, nullT = [], []
for b in range(200):
    perm = dict(zip(recs, rng.permutation(recs)))
    mk = np.array([recmiss[perm[r]] for r in REC]) & (np.abs(pd.Series(Y).groupby(REC).transform("mean").values) >= 0)   # record-level shuffle of MISS flag
    yy = pd.Series(Y, index=REC).groupby(level=0).mean().reindex([perm[r] for r in REC]).values
    nullL.append(best_lift(mk.astype(bool))[0]); nullT.append(best_t(yy)[0])
qL, qT = np.quantile(nullL, .95), np.quantile(nullT, .95)
print(f"  chance (record-shuffled, 200x) 95% quantile: lift {qL:.2f} | |t| {qT:.2f}  -> real lift {'ABOVE' if L0 > qL else 'within'} chance, real |t| {'ABOVE' if T0 > qT else 'within'} chance")
cvA, cvB = [], []
for tr, te in GroupKFold(5).split(D, groups=REC):
    mtr = np.zeros(len(D), bool); mtr[tr] = True
    rate = (MASKS & (M_ & mtr)).sum(1) / np.maximum((MASKS & mtr).sum(1), 1); rate[(MASKS & mtr).sum(1) < 6] = -1; k = int(np.argmax(rate))
    te_in = MASKS[k][te]; cvA.append((M_[te][te_in].mean() / max(M_[te].mean(), 1e-9)) if te_in.sum() else np.nan)
    sup = (MASKS & mtr).sum(1); mu = (MASKS * np.where(mtr, Y, 0)).sum(1) / np.maximum(sup, 1); var = (MASKS * np.where(mtr, (Y - mu[:, None]) ** 2, 0)).sum(1) / np.maximum(sup - 1, 1)
    t = np.where(sup >= 6, mu / np.sqrt(var / np.maximum(sup, 1) + 1e-12), 0); k = int(np.argmax(np.abs(t))); te_in = MASKS[k][te]
    cvB.append((mu[k], Y[te][te_in].mean() if te_in.sum() else np.nan, int(te_in.sum())))
print("  CV magnitude lift on held-out records:", [round(x, 2) if np.isfinite(x) else None for x in cvA])
print("  CV direction (train mean, held-out mean, n):", [(round(a, 2), round(b, 2) if np.isfinite(b) else None, n) for a, b, n in cvB])
okA = L0 > qL and sum(1 for x in cvA if np.isfinite(x) and x > 1.5) >= 4
okB = T0 > qT and sum(1 for a, b, n in cvB if np.isfinite(b) and np.sign(a) == np.sign(b) and abs(b) >= .3) >= 4
print(f"DHM3: magnitude rule {'CANDIDATE' if okA else 'chance-level'} | direction rule {'CANDIDATE' if okB else 'chance-level'}")
