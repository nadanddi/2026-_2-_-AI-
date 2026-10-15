# N4 실행 계약 보완

N4_PLAN_v1과 N4_PLAN_REVIEW_v1에 이어 실행 전 보완. 원 가설·threshold·계수·fold 변경 없음.

- 매 fold에서 동일 특징 baseline CODEX726와 CODEX9109의 raw 예측 exact equality 검사. 실패하면 해당 fold 후보와 PFN fit 전에 중단한다. IRR4의 기존 seed9109를 유지하면서 seed 차이와 특징 효과를 분리한다.
- Bootstrap은 농장×원 day//5 블록을 전체 풀에서 복원 추출하는 기존 방식이며, 농장별 층화는 하지 않는다. 농장 효과를 보장하는 검정으로 해석하지 않는다. 모든 지표는 두 농장 합산 행 MSE의 제곱근. 별도 최종 확인에서는 각 실제 stochastic seed와 각 검증기의 방향 조건을 적용해야 한다.
- 원 ZIP의 temperature package files 전부, 공식 입력3개, runtime_v1.py, 실행 source, checkpoint SHA를 recipe에 고정. 실행마다 시작 시 검사하며 캐시에도 recipe/NPZ SHA 및 학습·질의 ID 고정. 원래 공개된 source/hash 검증을 통과해야 fit 가능.
- 보류·buffer 행은 raw_fold에서 query쪽으로 옮기고 MASK world에서 입력14개를 모두 NaN으로 만든다. 학습 타깃은 train_ids만 loader에 공급한다. query frame은 sub_temp/sub_ec 열을 제거한다. 전체 BASE/CODEX/IRR/PFN feature union에 미래·다른 온실·MASK·질의 정답 독립성 검사 후에만 fit.
- 혼합 차이는 `(0.6-0.4*g)*g*(IRR-CODEX)`. g=0에서 raw exact 같음, g=1에서 0.2배 보완, NaN g=1을 합성/실제 예측에서 검사한다.
- 원3 BASE seed 평균과 PFN49 하나의 paired 결과만 다룬다. 누적 alpha6는 과거 모든 탐색의 선택 편향을 해결하지 않는다. 채택0; 실제 9회차 점수 개선 확인0.
