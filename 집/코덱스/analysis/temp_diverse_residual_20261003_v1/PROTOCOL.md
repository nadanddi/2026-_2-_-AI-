# 온도 W30G 새 잔차 모델 T-D1/T-D2 · 실행 전 고정

- 2026-10-03 집 코덱스. 사용자 '온도 모델 어떻게 개선할까, 뭐라도 해봐' 직접 지시. 실험만 수행, 제출 파일 생성 없음.
- 이전 모델계열조사에는 원시온도 ET가 있었으나 이 실험은 Codex MASK reset 피처+Ridge 물리식 잔차+따뜻한 행 10% 결합이다. CatBoost 온도 잔차는 검색한 카탈로그/작업일지에 없었음. 모든 과거 실험이 없다는 주장은 하지 않음.
- 타깃 sub_temp, 예측 시점 같은 온실의 현재 및 이전 입력. BASE/CODEX/PFN W30G 고정 기준선. 이전 T-S1/T-S2 기각 후 새 계열을 시험하는 순차 탐색.
- 특징 원본 CODEX FEATURE_COLUMNS 그대로(89열, day 포함). 물리식은 원본 PHYSICS_COLUMNS Ridge(alpha100), fit/imputer/scaler 학습fold만. 과거 EC 계절 지표 사용0.
- T-D1: Ridge 물리예측+CatBoost 잔차. RMSE/600trees/depth5/lr.04/l2=10/random_strength1/bootstrap Bayesian/bagging_temperature1/thread4/CPU/allow_writing_filesFalse. 시드726/727.
- T-D2: 같은 Ridge 물리예측+ExtraTrees 잔차. 300trees/minleaf20/max_features.8/n_jobs4/시드726727. 결측대체는fold학습중앙값.
- 두 안 각각 cand=W30G+0.10*g*(새멤버-BASE), g=clip((현재 in_temp-8)/2,0,1), NaN g1. 8도이하 변화0, 10도이상 BASE비중 .5->.4/새멤버.1/CODEX.2/PFN.3. 새모델을 기존CODEX 20%와 대체하지 않음. 두 후보 동시 결합/비중 탐색 없음.
- 고정 검증 DIAG10/EXT10/EXT12, 기존온도 +/-1일buffer; BASE시드7/101와 새멤버726/727 짝, PFN문맥1..8 및17..24 => 후보별12칸. 학습 MASK/원본input-only가중치 보존. CODEX cache재학습 row_id/숫자 최대차1e-8 확인.
- 이번세션의 계절2안도 포함한 보수적 다중검정 k4: 12칸 전부 RMSE개선 및 DIAG 각 p_worse<.00625,98.75% deltaMSE CI상한<0. 온실×5일블록bootstrap20k/RNG20261003 고정. 평균·fold분산·일수·후반/저온/온실/시간대 서술.
- 모델별 학습/검증RMSE와오차상관, day/shape오차 분해를 남긴다. 단독성능·상관이 좋아도 전체 판정 없이는 채택하지 않는다.
- baseline은이미존재하는W30G이며 mean baseline 재학습은생략. 기존PFN저장예측 재사용(재학습0)·전체학습일가중치rank의한계·공개검증재사용한계를명시. EC잠금파일 읽기0·test예측0. 독립수치/원시라벨/가중합검산 후 보고한다.
- 두안실패시 오차를분해하고 현재 연구에서 구별되는 입력/모델 문제를 찾는다. 실패한 파라미터를 결과에 맞춰 재탐색하지 않는다.
