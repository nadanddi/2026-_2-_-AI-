# 선형 추세 + ET 잔차 학습 · 실행 전 고정

2026-10-04 집 코덱스. EC 개선 지속목표의 단일 후속 후보, family15. 기존 공개22fold×3seed(7/101/2024), DIAG10/A/B/EXT10/EXT12, ±1buffer 및 제외행 규칙 유지. 과거 결과를 새 규칙으로 재판정하지 않는다. 이 실험도 모든15칸 개선 + DIAG10 farm층화5기록일 bootstrap20,000, p_worse<.025/15 및 동일수준 CI상한<0으로 공개 선별한다. 새 EL1/잠금/제출 판단 없음.

근거: [Regression-Enhanced Random Forests](https://arxiv.org/html/1904.10416v1) §2 전체 알고리즘 읽음. 상수 잎의 볼록 평균 대신 선형 추세를 먼저 학습하고 잔차를 forest로 예측하는 접근. 원 논문은 Lasso+RF 및 내부 튜닝이며 이번은 고정 Ridge100+ExtraTrees 적용이다. 원 논문의 수치 재현이나 성능 보장으로 부르지 않는다. [Local Linear Forests](https://arxiv.org/abs/1807.11408)는 관련 대안으로 초록 확인, 이번 구현은 아님.

사전 중복 검색: EC POWER2/가중/고EC전용/MLP/ET-PCA/공동목표/조건혼합/잔차교정은 이미 있음. 카탈로그에 EC 선형 추세를 먼저 제거한 뒤 동일 ET600로 잔차 학습하는 시험은 발견하지 못함. 온도 물리선형+트리 모델과 목표/입력이 다름.

한 가지 변경: 기존 ET구성원을 다음으로 교체한다.

1. 기존 FULL38에서 day→season 끝 열. 각 외부 학습 fold만으로 median imputer와 StandardScaler fit, Ridge(alpha=100)로 EC를 학습. 새교호항·farmID·외기·목표가중 추가 없음.
2. 동일 입력 FULL38에 ET600/leaf1/max_features1.0/기존seed 학습, 목표는 EC−Ridge 학습예측. 새 raw=Ridge(query)+ETresidual(query). 학습 내부 순차 잔차 학습이며 OOF 메타 보정이 아니다.
3. 후보=clip(기존 season_v2+.48*(인과shrink(newraw)−기존이미shrink된ET캐시),외부학습EC최소최대). PFN/LGB/MLP·비중·최종평활 고정. 전체 손실은 원래 RMSE이며 고EC만 나아도 채택하지 않음.

| 항목 | 예측시점 가용성 / 감사 |
|---|---|
| 기존 현재/0시/현재까지 누적 입력 | 같은 온실 현재이전 입력만, query 미래/다른온실 변조 불변성 |
| season | 기존 학습날씨 기준점 fit 및 query farm/day 보간, 새 batch 통계 없음 |
| imputer/scaler/Ridge/ET residual | 외부 학습행만 fit, 검증 정답은 점수 계산만 |
| 검증 정의 | 기존 공개 분할 고정, label 기반 EXT의 정의 변경 없음 |
| EC 라벨 | 공개 OOF8640행 재사용, 원시train_y/잠금/test 접근 없음 |

첫fold 원ET 캐시 재현, Ridge 정상방정식 독립 계산, 재학습/배치/각트리평균, future/다른온실 변조 검산. 전체 RMSE 및 혼합식 math.fsum 독립 검산, farm×구간×고EC·보통일·시각 분석. 원본/다른AI파일 편집 없음. 코드·로그 ownanalysis, OOF ownlocal. 반복 공개검증/소수 고EC 표본/원래 전체 상하한 제한은 한계로 남김.
