# EC 최종 출력 손실 학습 · 실행 전 고정

사용자 2026-10-04 지시 「1,2번 진행해봐」에 따른 1번. 집 코덱스, EC만.

## 후보와 대조
- family22 FINAL_LOSS: 기존 계절v2의 네 구성원은 그대로 두고, 입력 BASE14를 받는 잔차 MLP δ를 원 혼합값에 더한다. 학습 손실은 실제 추론식 `clip(.5(raw+δ)+.5 prefix_mean(raw+δ), train_min, train_max)`의 MSE이다.
- family23 RAW_LOSS: 같은 입력·초기값·구조·최적화기로 평활/clip 전 `(raw+δ-y)` MSE를 학습한다. 평가는 FINAL_LOSS와 같은 후처리다. 최종 출력 목표의 효과를 구분하는 대조이며 두 후보를 외부 정답으로 섞지 않는다.
- zero-head δ=0은 기준 계절v2를 재현해야 한다. 기존 네 구성원을 재학습하지 않는다. δ 크기 제한, 시드·검증기 선별, 결과에 따른 에폭 선택을 하지 않는다.
- 이는 새 잔차 학습기와 학습 손실을 시험하는 것으로, 원 LGB의 목적함수만 바꾼 실험이라고 주장하지 않는다.

## 학습
- 기존 공개 360일 EC OOF labels만 사용한다. 원시 train_y의 EC, 잠금, EL1 재채점, test 입력·예측·제출 생성은 금지한다.
- 22개 외부 분할마다 기존 matched inner OOF의 학습 a / 평가 b를 ordered ID, ±1일 purge, 부모 외부학습행 subset, PFN 문맥 RNG/ID 및 SHA로 검증한다.
- baseline raw를 만들었던 inner 모델은 a만 학습했고 잔차 MLP는 b와 그 공개 정답만 학습한다. 외부 평가 정답은 학습·전처리·선택에 투입하지 않는다.
- inner BASE14는 `seasonal(a,b)`로, 외부 평가 BASE14는 `seasonal(outer_train,outer_query)`로 만든다. 현재·과거 입력만 쓰며 day는 빼고 season을 마지막에 붙인다. 원 MASK 제외 다섯 열은 사용하지 않는다.
- MLP 14→64→32→1, ReLU, float64 CPU, 출력 head weight/bias 0, seed 7/101/2024. Adam lr=.001, weight_decay=.01, full batch 400 steps. 입력 imputer/scaler는 잔차학습 b에서만 fit한다. early stopping/외부 score 기반 선택 없음.
- FINAL_LOSS의 clip 범위는 inner 모델 학습 a의 min/max, 외부 추론 clip은 해당 외부학습의 min/max. 24시간이 있어도 각 평가시간의 prefix에 현재 및 이전 시간만 들어간다.

## 판정 · 두 후보와 정확 잎 후보 동시 고려
- 기존 5검증기 DIAG10/A/B/EXT10/EXT12 × 3시드, 각 후보 66셀 전부 완료·최초 감사 PASS 전 점수 계산 금지.
- 두 후보와 별도 family24 정확 Tweedie 잎 대조를 합쳐 이번 신규 3안은 모두 누적 family24로 보수적으로 고정한다. alpha=.025/24, 조정 CI quantiles=[alpha,1-alpha]. 기존 이미 판정한 실험의 기준은 소급 변경하지 않는다.
- 채택은 15점수 모두 기준선 대비 엄격 개선이고 DIAG10 각 시드의 p_worse<alpha 및 조정 CI 상한<0일 때만.
- DIAG bootstrap 20,000회, seed=20261003+model_seed, 농장별 시간순 5관측일 블록. 달력 연속 5일이라는 뜻이 아니다.
- 같은 고정 모델 FINAL vs RAW의 비교는 손실 목표 효과를 보여주는 보조 비교다. 어느 한 후보가 대조보다 좋아도 계절v2 채택 조건을 대체하지 않는다.

## 감사
- 실제 fit 전 code/preparation/runtime/source/입력/캐시 SHA를 고정하고 main에 등록한다. 매 fit 전 검증한다. 기존 파일 변경 및 부분 산출물 재시작을 금지한다.
- 첫 실제 모델은 zero-head 기준선, fresh 동일시드 재학습, query 순서·배치·다른 query 독립, 현재시간 prefix, scalar 후처리와의 1e-12 대조를 통과해야 한다. 실패 산출물은 새 파일로 보존하고 점수로 채택하지 않는다.
- checkpoint는 전처리와 전체 weights, 학습 초기/최종 손실을 저장한다. 독립 whole 검산은 전체 ordered ID·공개 labels·baseline raw·출력·bounds·서명·SHA·checkpoint 및 산술을 확인한다. 재학습 없이 별도 NumPy forward로 checkpoint 출력을 재생한다.
- 같은 공개 검증셋에 누적 실험을 하므로 독립 신규 holdout의 확인이라고 주장하지 않는다. 채택·제출 파일 생성은 별도 사용자 요청 전 금지.
