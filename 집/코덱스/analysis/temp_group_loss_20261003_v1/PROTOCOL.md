# 온도 그룹 손실3안 · 실행 전 고정

사용자의 추가 탐색 직접 지시. 이전 level/shape 분리 결과는 shape 개선, level 악화. 별도 0시 daylevel 회귀는 닫고, 원 CODEX 행 단위 모델을 유지하며 학습 손실만 변경한다. RC/출력평활/다른온실전이는 기존 기각 사례 확인. 그룹 손실의 이 구체식은 카탈로그와 다른 AI 소스 검색에서 확인되지 않았으며 모든 유사 연구 부재 주장은 하지 않는다.

원 물리Ridge100, MASK89피처, LGB220/.035/leaf12/minchild100/L2=15 및 학습가중치 그대로. 새 custom objective에서 가중치를 직접 gradient/hessian에 적용하므로 fit sample_weight는 넣지 않는다. weighted residual mean을 고정 intercept로 빼서 기본 L2 초기값과 맞춘다.

- L1: loss = .5 Σw e² + .5 Σday Wday mean_w(e)². 하루 수준 오차 가중치2배.
- S1: loss = .5 Σw e² − .25 Σday Wday mean_w(e)². 시간 모양 오차 유지, 하루 수준 항만 절반. 양의 정부호(수준 .5, 모양1).
- D1: loss = .5 Σw e² + .5 Σsame-day-consecutive min(w_i,w_prev)(e_i−e_prev)². 정답의 시간 증분 오차를 추가로 벌줌. 하루/온실 경계를 넘지 않는다.
- objective의 Hessian은 정확한 대각이며 그룹 손실의 비대각 결합은 LGB에서 근사한다. 예측은 현재/이전 입력만. 전체 학습일 타깃은 학습 목적함수에서만 사용하고 평가 정답/미래입력은 예측 경로에 없다.

한 안마다 원 W30G의 CODEX만 교체. BASE/PFN/게이트/비중 유지. 원 DIAG10/EXT10/EXT12 ±1buffer, 시드726/727와 BASE7/101, PFN문맥1~8/17~24. 후보별12칸 모두 RMSE감소 및 DIAG20k5일농장블록 p_worse<.025/10,99.5% ΔMSE CI상한<0. 이전7가설+이번3=10으로 보수 보정. RNG20261003. 수치미분으로 목적함수 gradient/Hessian 대각 검증과 ZERO손실 기준모델 비교 후 전체 적합.

원12칸 통과시 전체W30G 날씨GUARD 재학습검증을 자동 계속하고 정식 채택은 그 후. 실패면 기각. 기존CV를 반복 사용한 탐색이며 독립홀드아웃/리더보드개선 증거 아님. 기존TF global input-rank 유지 한계. 학습검증격차/농장·시간·전후반·level/shape 분해 검산. 새소스 main사전커밋. 제출파일/test평가예측/EC잠금열람0.
