# N3 0시 수준·0시 대비 모양 두 머리 — 기존 기각 연결 비평

2026-10-10 연구실 코덱스 독립 계획 비평. 새 후보 코드·모델 fit/predict·정답 로딩/재채점 0. 외부자료·외부파생로그는 읽지 않았다. 아래 수치는 기존 **온도** 결과만 사용한다. N2 결과는 부모 전달(DIAG 약−.048%, p .43585; EXT10 +.216%, EXT12 +.402%; 전체혼합 미진입)이며 이번 독립 재검산은 아니다.

**판정: 확인한 공식 온도 코드 범위에서 N3와 동일한 y0 수준+Δfrom0 모양+예측모양0 차감 명세가 이미 실행됐다는 근거는 찾지 못했다. 다만 두 머리 수준·모양 구조는 6b.46 H1/H2에서 이미 실행·기각됐다. N3를 새 두 머리 발명으로 설명하면 안 된다. 새로운 부분은 하루 평균잔차 대신 실제0시 온도 anchor를 타깃으로 삼고 모양 예측을0시에서 강제0으로 맞추는 방식이다.**

## 1. 코드로 확인한 가장 가까운 선행

### 6b.46 H1/H2: 같은 두 머리 구조, 다른 anchor

`집/코덱스/analysis/temp_hierarchical_20261003_v1/run.py`는 원 physics Ridge를 절대 y에 fit하고, 그 잔차 r=y−physics의 **학습날 평균**을 level target으로 삼는다. level은 하루당 0시 한행의 DAYCOL(14개의 *_h0+farm/day/second)로 학습한다. H1은 level LGB, H2는 level Ridge이다. shape LGB220은 r−mean_day(r)를 원CF89로 예측한다. 최종은 physics(h)+level_hat(h0)+shape_hat(h)이다. 보고서에서 실행본 run_v2의 차이는 누락 day열을 보존하는 수정이라고 명시한다.

H1 W30G 기준 DIAG +.338~.363%, EXT10 +.045~.161%, EXT12 +2.018~2.232%, 0/12 개선; H2는 DIAG +1.095~1.128%, EXT10 −.557~−.245%, EXT12 +.726~.886%, 4/12 개선으로 모두 기각이다. **모양은 약 .63~.68% 좋아졌지만 수준이 악화**했고 F47/후반·초기시각 손실이 남았다. 따라서 ‘수준 병목이 있으니 분리하면 개선’은 이미 반례가 있다.

N3는 physics 전체 절대y 모델을 유지하는 H1과 달리, 별도 level 모델로 y0 자체를 예측하고 shape를 y(h)−y0에 fit한다. 그리고 shape_hat(h)−shape_hat0로0시 anchor를 강제한다. 이 차이는 모델의 target과 오차 전파가 달라지므로 같은 실행이라고 멈출 근거는 아니다. 그러나 새로운 정보원이 추가되는 것은 아니고, 기존 입력의 target/함수분해 재설계다.

### 6b.49 B1: 실제0시 정답 anchor가 아니었음

`temp_pipeline_models.py`의 B1은 impute한 **in_temp_reset3**를 anchor로 두고 physics Ridge에 y−air_anchor를 fit한 뒤 air_anchor를 복원한다. 이어 원CF89 잔차 LGB를 학습한다. 온도 정답 y0를 별도 level target으로 쓰지 않으며 shape0 차감도 없다. DIAG +.186~.200%, EXT10 −.253~−.152%, EXT12 −.406~−.390%, 8/12 개선이나 기준 미달이라는 기존 기록이다.

N3는 관측공기 anchor를 실제배지0시 정답으로 바꾸되 query에서는 이를 예측한다. 따라서 B1에서 알려진 air anchor를 되돌리는 것보다 어려운 anchor 추정 문제를 새로 추가한다. 전문가의 배지–공기 차이가 정상일 수 있다는 설명은 왜 y0와 air0를 구별해야 하는지 발상을 돕지만 **input으로 y0를 더 잘 맞힐 수 있다는 증거는 아니다**.

### 6b.47 L1/D1·N2·IRR4와의 관계

- L1은 day-level 손실2배, D1은 시간증분오차 손실추가이며 원target/구조를 유지했다. 둘 다 모든validator·통계조건 실패다. N3는 같은 손실가중 반복이 아니라 y0/Δ0 두 target 분리지만 최종절대 RMSE를 직접 최적화하는 구조에서 멀어질 위험은 공통이다.
- N2는 공식타농장 시간당Δprior를 누적 특징으로 넣었다. 사전gate 실패를 확인한 이상 N3에 N2 prior를 넣거나 N2 좋은 농장·시간만 고르는 것은 금지다. N3 자체의 y(h)−y0 target은 타농장전이가 아니며 인접Δ 누적 예측도 아니다.
- IRR4는 CODEX 구성원 DIAG 통계조건 통과/EXT10 악화로 미진입이었다. N3도 DIAG 또는 shape만 좋아진 수치로 온도 전체 방향조건을 덮지 않는다. W30G 과거기각을 W40G-S exact효과로 혼동하지 않는다.

## 2. 명세에 남은 구현·통계 위험

### A. 0시 anchor 오차가 하루 전체로 전파됨

query 최종오차는 `e(h)=e_level0 + e_delta(h)−e_delta(0)`이다. 하루 평균잔차 중심의 기존 직교 level/shape SSE분해와 달리 **y0 anchor 분해는 직교하지 않는다**. 두 항의 공분산이 최종 SSE를 바꾼다. level0/Δshape 개별 RMSE가 좋아도 합이 나빠질 수 있고, 반대로 shape가 나빠도 anchor와 상쇄될 수 있다.

원CODEX는0시 정답의 불확실성을 모든시각에 동일하게 복사하지 않는다. N3 level 모델은 농장당0시 하루 한행만 학습하므로 학습지원이 크게 줄고, 희귀 배지-state의0시 정답·측정오차를 하루모양 target 전체에 넣는다. 이미6b.13의0시 세멤버 약세,6b.46의level 약세가 반례다. **0시 RMSE/일평균편향/모양오차/level-delta 공분산**을 사전 진단표에 포함하는 것이 필요하다. 정답 y0 오라클을 연구진단으로 넣더라도 실제query 성능으로 번역하지 않는다.

### B. raw shape 학습과 anchor 차감 후 최종출력의 목적 불일치

shape를 Δ0y에 보통 Ridge+LGB loss로 fit한 뒤 inference에서 shape_hat0를 빼면, 실제 최종shape는 학습한 rawshape와 다르다. h0 target은0이지만 규제·LGB 분할·imputer 때문에 shape_hat0가0이라고 보장되지 않는다. 이 날별 값이 최종 하루수준을 이동시킨다.

따라서 코드등록 전에 **학습은 rawΔloss, 추론만0차감**이라는 현재명세를 그대로 고정할지 정해야 한다. 차감 전/후 train·OOF 진단을 하되 그 결과로 차감 강도나 level 보정계수를 재선택하면 새 사후후보다. 원명세를 분석 중 몰래 anchored loss로 바꾸지 않는다.

### C. PHYS19의 ‘현재-minus-h0’와 interceptFalse를 정확히 정의

모든 물리열의 현재값에서 같은농장·날의0시 **그 열 값**을 뺀다는 수식을 고정한다. farm_id 같은 날상수는0이고 sin/cos·reset열도 차분 정의가 필요하다. season이shape physics에 들어가는지 요청명세는PHYS19만이므로 임의 추가하지 않는다.

`StandardScaler(with_mean=True)`를 원Ridge pipeline처럼 쓰면0벡터를0으로 보존하지 않아 `fit_intercept=False`만으로 rawphysics(0)=0이 보장되지 않는다. NaN imputer도 차분0행을 바꿀 수 있다. **scale-only/centering/차분 전후 impute 순서/keep-empty열**을 사전 명시하고 합성0행 검사로 확인해야 한다. 최종 shape0 차감이 이 문제를 상쇄하더라도 rawphysics가0 anchor라는 설명은 정확하지 않을 수 있다.

### D. fold·가중치·허용 입력의 경계

- y0 table은 outer train의 완전한 농장·날 중 h0 실제라벨만 사용한다. query·purge 날의 y0는 shape target생성·level fit·feature 어느 단계에도 들어가면 안 된다. query는 label삭제 sentinel 검사까지 필요하다.
- level 학습 w는 원h0 행가중치를 쓸지 날평균 가중치를 쓸지 아직 고정할 필요가 있다. shape 원시간행 w 유지와 level 하루1행 w 정의를 구별한다. 기존41일 제거·하향을 재도입하지 않는다.
- query level 입력은0시의현재·과거허용 PHYS19+train-only season이며 뒤시각 집계로 만든0시 특징을 금지한다. 현재 원CF89의prefix 기능을 그대로 유지하고 다른날 h0 forward-fill을 금지한다.
- query의 shape0 평가도0시 입력prefix만 소비해야 한다. shape0를 전체batch로 predict하더라도 batch독립/미래·타농장변조 불변을 검사한다. h0 및h0 부재·중간시간결손 규칙은 미리 고정한다.
- level Ridge100+LGB120/.03/leaves8/minchild12/lambda15와shape Ridge100/LGB220의 모든나머지参数·thread/seed/subsample/scaler 설정을등록한다. CODEX726/727 동일예측을독립시드로 세지 않는다.

## 3. 다음 진행 판정

현재근거에서 **명세 중복을 이유로 즉시 종료할 필요는 없지만 target fit 바로 진입에 대한 무조건 승인도 아니다**. N3는H1/H2 실패에 대해①평균잔차 대신실제y0 anchor,②physics도Δ0,③shape0 subtraction이라는차이를 가진 제한된새가설이다. 세변경이동시에있어통과하더라도각각의원인기여를분리한증거는아니다.

기존근거검색결과와위명세보완을새계획/등록에고정한후1회검증이합리적이다. 사전구성원gate를둘지전체W40G-S로직접paired검증할지정하고, level만/shape만결과로새threshold를고르지않는다. 채택은DIAG10·EXT10·EXT12 전seed방향/DIAG p<.025/k·CI상단<0 및사전신선한확인계약을따른다. 실패하면‘전문가배지state라서’사후동·농장gate로덮지않는다.

## 4. 근거 provenance·검색 한계

직접읽은공식온도근거: `temp_hierarchical_20261003_v1/run.py·PROTOCOL.md·결과보고서_v1.md`, `temp_pipeline_20261003_v1/temp_pipeline_models.py`, 카탈로그온도6b.46/47/49 및선행온도기록. 관련코드의 `y0/y_h0/sub_temp*h0/anchor` 검색도 수행했다. 외부폴더/EC전용코드는검색범위에서제외했고 외부로그내용사용0이다. catalog 검색에는관련없는행도출력됐으나그수치·결론은본문근거에서사용하지않았다.

검색은동일코드발견여부를확인한것이며모든Git과ignored산출물전수동일성확인을뜻하지않는다. N3 OOF효용·전문가물리원인·9회차보다낮은평가RMSE는현재미검증이다.
