# 사용자 요청으로 일시 중지 — 2026-10-09 · 집 · 코덱스

고EC만 학습한 OneClassSVM으로 고EC와 일반날을 구분하고, 검증에는 일반날을 남겨 오탐을 확인하는 실험이다. 사용자 요청에 따라 추가 학습·검토를 중지했다. 재개 지시 전에는 실행하지 않는다.

## 저장된 상태

- 기준: 일평균 EC ≥ 1.2. 고정 계획 PLAN_v2.md, 실행 코드 run_v2.py, 원본·비교 예측 해시 registration_v1.json.
- prepare와 첫 두 묶음 DIAG10_000/001 완료. 전체 27묶음 중 2묶음 저장, 나머지 25묶음은 시작하지 않았다.
- 예측 CSV/JSON은 local/ec_highday_oneclass_validation_20261009_v1/predictions에 있다. first_v1.log와 midpoint_score_v1.json도 보존했다.
- 첫 실행 프로세스는 정상 종료했다. 중간 검토 담당 curtain_critic은 중지했다. 저장 시 critique_midpoint_v1.md는 없으므로 중간 검토가 아직 필요하다.
- report_checks_v1.py는 작성만 했고 실행하지 않았다. 최종 결과·채택·전체 데이터 모델·제출물은 만들지 않았다.

## 그대로 이어가는 순서

1. 사용자 재개 지시 후 registration_v1.json의 원본 해시와 완료 파일을 확인한다. 기존 파일을 수정하거나 prepare/first를 다시 실행하지 않는다.
2. 독립 중간 검토를 다시 요청한다. 계획에서 고친 native predict(+1) 판정은 decision_function > 0이며, >= 0으로 바꾸지 않는다. 중간 검토를 완료하고 critique_midpoint_v1.md를 저장한 후에만 나머지 실행을 시작한다.
3. 저장소 루트 PowerShell에서 다음 명령을 실행한다(아직 실행하지 않음).

```powershell
$env:PYTHONPATH=""
python -u '집/코덱스/analysis/ec_highday_oneclass_validation_20261009_v1/run_v2.py' rest *> '집/코덱스/analysis/ec_highday_oneclass_validation_20261009_v1/rest_v1.log'
```

4. 남은 25묶음 완료 후 report_checks_v1.py로 독립 산술·재현성 검산, 최종 독립 비평, 보고서와 작업 기록 갱신을 진행한다. 실행 전 코드의 rest 전제 조건을 다시 읽는다.

## 고정된 설계와 해석 제한

SimpleImputer(median, keep_empty_features=True) → StandardScaler → OneClassSVM(rbf, nu=0.1, gamma=scale, tol=0.001). 고EC 학습 행만으로 전처리와 모델을 fit한다. 기존 73개 현재·이전 입력 특징과 27개 분할/purge를 유지한다. 검증에는 양쪽 클래스를 남긴다. h15가 주지표이고 h0/8/23은 보조다. 임계값·파라미터를 결과에 맞춰 조정하지 않는다.

첫 두 묶음은 고EC 검증일이 1일뿐인 중간 결과로, 최종 성능 판단에 쓰지 않는다. 기존 ET 비교는 학습 구성과 모델 종류가 함께 달라져 원인을 분리할 수 없다. 세 검증기의 일부 평가일이 겹치므로 독립 표본으로 합산하지 않는다. 이전 EC 수치 예측 캐시 감사는 사용자가 고EC 여부 분류로 확정했을 때 중단된 별도 작업이며 재개하지 않는다.
