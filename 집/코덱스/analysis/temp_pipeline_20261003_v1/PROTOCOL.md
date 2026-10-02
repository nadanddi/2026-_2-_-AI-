# 온도 학습·계산·피처·추론4안 실행 전 고정

사용자 요청: 모델 자체 학습, 계산 방식, 피처 사용, 추론 식까지 개선. 기존11안·카탈로그·온도 소스 검색 후 다음4안을 고정. 모든 유사연구 부재 주장은 하지 않는다. 고정 CODEX는 physicsRidge100의 in-sample 잔차를 LGB220/.035/leaf12/minchild100/L2=15로 학습하고 그 잔차를1배 더함. 이전 학습/검증 격차와 상대적 온도 정보의 중복을 대상으로 한다.

1. A1 학습: 외부fold 학습일만 내부3분할(같은온실 정렬일의5일묶음 순환, ±1일buffer)하여 물리식 OOF를 만든다. 학습타깃은 y−physicsOOF. 외부query는 외부train 전체 적합 physics+새잔차. 내부query정답은 내부physicsfit에서제외되며 외부query정답은어떤훈련에도없음.
2. B1 계산: 물리 Ridge가 y 자체 대신 y−in_temp_reset3를 적합하고 예측에 그 관측 온도를 다시 더한다. 원19입력/alpha100 그대로이므로 모델클래스 확장이 아니라 기울기0으로의 Ridge prior를 공기추종 prior로 옮기는 계산 변경. 결측 anchor는fold train median. 이후 같은LGB잔차.
3. C1 피처: 원89열 보존+현재in_temp 대비 h0/당일mean/reset1/3/8 차이5열, reset3−reset8 1열. 물리식/타깃/모델은원래그대로. 이력특징은평가같은온실현재·과거만,학습MASK.
4. I1 추론: 원physics/잔차모델 그대로. 같은 내부3fold에서 원physics+LGB OOF를 만들고 gamma=clip((Σw*rOOF*(y−physicsOOF)+100)/(Σw*rOOF²+100),0,1.25)를 학습. 외부예측physics+gamma*residual. 100은coefficient1에대한고정정규화. 외부점수로gamma선택0·출력평활없음.

각안 원W30G CODEX만교체,BASE/PFN/게이트/비중고정. 원DIAG10/EXT10/EXT12 ±1일buffer; CODEX726/727(BASE7/101),PFN1~8/17~24. 후보12칸 모두RMSE감소 및각DIAG20k농장5일블록 p_worse<.025/15와99.666667% ΔMSE CI상한<0. 앞11+이번4=15보수보정,RNG20261003. 원12칸통과시전체W30G 날씨GUARD추가검증 자동계속·그전채택없음.

중간diagnostic수치는train/validation격차,baselineCODEXcache재현,내부훈련일독립성·gamma분자분모·농장/시간/전후반/level/shape/fold표준편차. 새imputer/scaler/model은각외부/내부fold train만fit. 기존TF global입력rank를고정한한계,BASE/PFN캐시재사용,726/727샘플링없는시드실질독립아님,반복CV·독립홀드아웃없음 명시. test평가예측/제출파일/EC잠금0. 소스main사전커밋,원자료·기존산출물수정0.
