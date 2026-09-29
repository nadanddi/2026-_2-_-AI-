# 정형데이터 탐색 분석

목적: 센서·온실·시간 구조, 반복 및 이상 패턴을 찾아 모델 특징과 전처리 후보를 평가합니다.
원본 자료는 수정하지 않습니다. [AI 인계용 요약](local/정형데이터_탐색보고서.md)만 공유하고, `analysis/local/`의 상세 결과·그래프·백업은 Git 추적 대상에서 제외합니다.

## 진행 상태

최신 v5: `python analysis/ec_removal_followup.py`. EC ExtraTrees에서 온실 ID를 제거해 같은 5겹 CV RMSE가 0.2998→0.2654 dS/m로 개선됐습니다. 데이터 품질 표시 동시 제거는 0.2672로 덜 좋았습니다. v4 온도 모델은 유지하며 최종 모델·예측은 `local/ec_refined_v5/`에 저장됩니다.

v4 변수 기여 분석: `python analysis/feature_group_ablation.py`. 8개 변수 묶음을 하나씩 제거해 같은 5겹 CV로 재학습하고 온실-하루 부트스트랩 95% 구간을 계산합니다. 결과는 `local/feature_group_ablation_v4/결과요약.md`에 있습니다. EC의 온실 ID 제거가 RMSE 0.2998→0.2654로 가장 큰 추가 개선 후보입니다.

최신 v4: `python analysis/compare_lgbm_extratrees.py`. 같은 v3 5겹 CV에서 온도 LightGBM 혼합은 0.8464→0.8398℃, EC ExtraTrees 통합은 0.3139→0.2998 dS/m. 전체 정답 재학습·모델 저장·재예측 검증을 완료했으며 `local/lgbm_extratrees_v4/`에 별도 보관합니다. 상세는 `MODEL_CARD.md`입니다. 대회 업로드는 하지 않았습니다.

최신 v3: `python analysis/train_all_farms_cv.py`. 온도는 51개 온실 전체 정답을 사용한 통합 모델과 대상 온실 모델을 혼합하고, EC는 정답이 있는 두 온실을 통합 학습합니다. 10일 블록 5겹 교차검증·인접일/동일 날씨/입력 중복 제외로 비교한 결과, 동일 CV 기준 온도 0.8662→0.8464℃, EC 0.3522→0.3139 dS/m입니다. 선택에 사용한 CV이므로 독립 성능 추정은 아니며 v1/v2 점수와 직접 비교하지 않습니다. 산출물은 `local/all_farms_cv_v3/`, 상세는 `MODEL_CARD.md`, 테스트는 `test_all_farms_cv.py`입니다.

추가 가설 모델 v2: `python analysis/train_hypothesis_model.py`로 공식 시간축을 유지한 이틀 전 연결·날씨 접두 일치·당일 상태 특징을 비교합니다. 결과는 `local/hypothesis_model_v2/`에 별도 저장합니다. v1 대비 개선이 일관되지 않아 비교 후보로 보관합니다. `--predict-only`로 저장 모델 예측을 재현할 수 있습니다.

기본 분석 후 초기 모델 v1 학습과 로컬 예측 CSV 생성을 완료했습니다. 온실별 온도·EC 트리 모델 4개이며, 검증으로 온도 62개 특징·EC 18개 특징을 선택했습니다. `local/initial_model_v1/MODEL_CARD.md`에 점수와 한계를 기록했습니다. 대회 업로드는 하지 않았습니다. 기존 `ec_validation.py`는 일부 실행 결과만 있고 `preprocessing_validation.py`는 미실행입니다.

초기 모델 재현: `python analysis/train_initial_model.py`. 저장 모델 재예측: `python analysis/train_initial_model.py --predict-only`. 특징 검사: `python analysis/test_initial_features.py`. 라이브러리 버전·해시와 모델은 로컬 `initial_model_v1/`에 저장됩니다.

추가 구조 분석 재현 순서: `house_sequence_audit.py` → `matched_weather_audit.py` → `exception_audit.py` → `final_structure_audit.py`. 원본 대회 자료는 별도로 로컬에 준비해야 하며 저장소에는 포함하지 않습니다.

## 실행

Python 3.12 이상과 pandas, numpy가 필요합니다. 비선형 검증과 그림은 scikit-learn, matplotlib를 추가로 사용합니다.
이 작업 환경의 Python은 `C:/Users/aozks/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`입니다.
추가 분석 라이브러리는 프로젝트의 `.analysis-tools/python`에 있습니다.

```text
python analysis/profile_tabular.py
python analysis/deep_audit.py
python analysis/daily_audit.py
python analysis/test_causal_features.py
python analysis/validate_features.py
python analysis/nonlinear_validation.py
python analysis/nonlinear_validation.py --purge-weather
python analysis/pooling_audit.py
python analysis/house_sequence_audit.py
python analysis/house_sequence_verify.py
python analysis/plot_audit.py
```

`profile_tabular.py`는 파일별 SHA-256, ID 무결성, 51개 온실 통계, 결측·반복·범위·시차·분포 차이를 기록합니다.
`deep_audit.py`는 온실 간 입력 중복과 정답 일치, 센서 해상도와 정답 변화를 조사합니다.
`daily_audit.py`는 일 경계, 일별 반복, 자기상관, 날짜 잔여류, 일내 상관을 조사합니다.
`causal_features.py`는 같은 온실의 현재·과거 입력으로만 특징을 생성합니다. 전처리 통계는 검증 훈련 부분에서 적합합니다.
`test_causal_features.py`는 미래 변경 불변성, 정답 열 비참조, 시간 공백, 온실 분리, 행 순서 독립성을 검사합니다.

## 검증 해석

아래는 초기 분석 스크립트의 설정입니다. 최신 v3는 위의 전체 온실 학습·5겹 교차검증 설정을 사용하며 F13 일차 이동을 사용하지 않습니다.

- 목표 온실 F13, F47을 각각 학습합니다. 평가 정답을 사용하거나 제출하지 않습니다.
- `past_only`: 검증 블록 이전 정답만 학습합니다.
- `blocked_train_both_sides`: 검증 블록 정답을 제외한 공개 학습 정답으로 학습합니다. 특징에는 미래 입력을 쓰지 않습니다.
- F13의 상대 일차를 2일 옮겨 비슷한 자료 분할 위치에 검증 블록을 만듭니다. 실제 날짜나 날씨가 일치한다고 가정하는 것이 아닙니다.
- 랜덤 행 분할은 사용하지 않습니다. `--purge-weather`는 검증일과 동일한 외부환경 4개 변수의 하루 벡터를 가진 훈련일도 제외합니다.
- 결과는 탐색적 로컬 검증이며 대회 점수가 아닙니다. 단일 상관이나 훈련 내 패턴만으로 유효성을 확정하지 않습니다.
- `sub_temp`와 `sub_ec`는 특징 생성에 쓰지 않습니다. 공개 정답의 시차 특징을 허용한다고 해석하지 않습니다.
- 전체 하루 평균·전체 기간 보간·중심 이동평균은 예측 특징으로 사용하지 않습니다.

외부 자료와 모델 가중치는 사용하지 않았습니다. 추가 패키지는 로컬 분석·검증용입니다.

## 숨은 동·센서계열 가설

공식 정의상 같은 온실의 상대일차가 1 증가하면 하루 뒤입니다. 아래 검사는 이 공식 시간축을 바꾸지 않고, 관측값 안에 별도의 하위 계열이 있는지만 조사합니다.
`house_sequence_audit.py`는 외부 환경 4개×24시간의 완전 일치와 블록 순서를 검사합니다.
`house_sequence_verify.py`는 같은 순번·다른 순번 비교와 평가 입력의 반복 여부를 로컬 집계합니다.
둘 다 사후 진단입니다. 전체 하루 서명이나 정답 연속성으로 얻은 계열을 평가 특징에 직접 넣지 않습니다.
