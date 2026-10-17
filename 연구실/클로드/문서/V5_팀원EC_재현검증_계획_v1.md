# V5 — 팀원 EC 파이프라인(ec_v5.md) 재현 + 우리 검증기 판정 (계획 v1, 2026-10-09 연구실 클로드, 사용자 지시)

## 목적
팀원 문서 `ec_v5.md`(Drive zip, 2026-10-09)의 EC 파이프라인을 문서대로 구현하고, 우리 EC 검증기(DIAG10·A·B, 안전장치 EL1)에서 현재 기준(R3)과 비교한다. 재현 확인으로 팀원 `submission_v5.csv`의 sub_ec와 상관을 본다.

## 데이터 사실 (실행 전 확인)
- EC 정답은 F13·F47만(9600행, 잠금 40일 제외 8640행). 학습 범위 0.062~3.46 = 문서의 clip 값.
- F13·F47은 학습·평가 모두 `in_rad`, `act_pump`, `act_side`, `act_valve`, `act_cool` 100% 결측 → 문서의 evap_force·day_cum_rad·day_cum_pump는 ffill 후 0 대치로 **상수 0**. (MASK 규칙: test_X 결측 열 = 학습 결측 열이므로 학습 특징도 같은 마스크. 스크립트에서 assert)

## 구현 (문서대로, 문서에 없는 값은 아래처럼 고정)
- 입력 19열 전부 + gh 코드(F13=0/F47=1) + day + hour. 온실별 (day,hour) 정렬 후 ffill, 남은 NaN 0. 학습 묶음과 평가 묶음은 따로 ffill(문서대로).
- VPD = max(0, es − es·RH/100), es = 0.61078·exp(17.27T/(T+237.3)); evap_force = VPD·in_rad; sin/cos hour; day_cum_rad, day_cum_pump(온실·날 누적).
- Stage 1: 온실별 HuberRegressor(ε=1.35), x = day + hour/24, **폴드 안 학습 행으로만** 적합. 기울기를 [−0.04, 0.04]로 자르고, 잘렸으면 절편 = median(y − slope·x).
- Stage 2: 잔차를 ET(300그루), LGBM(600, lr .03, leaves 31, subsample .8, colsample .8), CatBoost(800, lr .05, depth 6)로 학습. 시드 42/1004/2026(문서값).
- 혼합: SLSQP(Σw=1, w≥0)로 clip(R·w + trend, .062, 3.46)의 RMSE 최소화. **가중치는 DIAG10 OOF(시드 평균 구성원)에서 한 번 정해 A·B·EL1에 그대로 적용**(DIAG10 점수는 가중치 적합에 쓰였으므로 낙관 — 표시). 민감도: 같은 가중치 1/3.
- 참고(서술): 팀원 방식 GroupKFold(day, 5) OOF RMSE — 우리 검증기와 얼마나 다른지.
- 재현 확인(서술): 학습 전체(잠금 포함 9600행)로 적합 → test 예측, `submission_v5.csv` sub_ec와 상관·RMS 차이. 제출 파일 아님(`local/`에만 저장).

## 기준
`연구실/클로드/local/ec_rw1_all_v1.csv`의 기준 열(b_*: 9회차 근사 R3 = .6ET+.3LGB+.1MLP, clip 학습 범위, 시드 47/1414/6464). 같은 폴드·같은 학습 제외(±1일, 잠금 ±1일)를 RW1과 같은 코드(p3.prepare 폴드 + EL1 5일 묶음)로 만들고, row_id·validator·fold가 기준 파일과 1:1 일치하는지 assert.

## 판정 (실행 전 고정, EC 규칙 2026-10-04)
- PASS = 시드 쌍(42↔47, 1004↔1414, 2026↔6464) 모두 × DIAG10·A·B에서 후보 RMSE < 기준, **그리고** DIAG10 시드평균 블록 부트스트랩(온실×day//5, 20000회) P(worse) < .025. 후보 1개(SLSQP 가중)만 판정, 1/3 가중은 서술.
- 안전장치(판정 아님): EL1 또는 DIAG10 2차(day ≥ 179) +2% 이상 악화 시 보류.
- 서술: 0.8·R3+0.2·PFN(WT1 PFN 열) 대비 0.8·V5+0.2·PFN, 구성원별 점수, 추세선 단독 점수.

## 예상 위험
- 문서에 하이퍼파라미터가 없어 팀원 모델과 완전히 같지는 않음 → 재현 상관으로 확인.
- 팀원이 추세선을 폴드 밖(전체)으로 적합했다면 팀원 OOF는 누수 — 우리는 폴드 안.
