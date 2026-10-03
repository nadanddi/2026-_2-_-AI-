# 因과 중첩 볼록 수준 보정 — 실행 전 고정

2026-10-03 집 코덱스. 고EC 순위 구별(0시 AUC .983)은 강하지만 크기 과소가 남음을 현재 계절v2 공개자료로 재확인한 뒤 등록. ISO0(다른DIAG OOF/R3S 등위보정), 고값고정확대, 상태분리 실패와 구분한다. 이번은 actual season_v2의 outer fold 내부 독립 예측만으로 제한된 볼록 calibration을 학습하는 별도 레시피이며, 과거 단조보정 일반을 미시험이라고 주장하지 않는다.

- 단일 후보 NESTED_CONVEX20. 실제계절v2,동일22fold3seed15칸.
- 각outerfold에 이미 있는 innerCPU3seed+PFN4문맥 cache 재사용. innertrain/innervalid 둘 다 outer 검증±1과잠금±1 배제. 이 모형 예측 기준은 기존shrink/clip(.8R3+.2PFN)이며 현재baseline보다학습표본적은편향한계존재. 새로운PFNfit없음.
- 시각h별 학습날1행: x=inner기준예측의0..h누적평균, target=실제공개EC의하루평균−x. basis=[1,x,max(x−.6,0),max(x−1,0)]. .6/1은사전 고정하며달라질때마다새실험필요.
- L2λ10 모든계수(상수포함) 정규화, scipy.optimize.lsq_linear로 constrained residual fit. bounds=[상수자유, β1≥−1, β2≥0, β3≥0]. 전체level 함수의단조·볼록성을제약하며β들의숫자를직접사후확대하지않음.
- query는기존계절v2의0..h누적평균만위basis에넣음. candidate=clip(baseline+.2*clip(β·basis,−.3,.3),outer학습EC범위). 하루기준수준보정과시간shape는분리. 새EC정답이추론입력에들어가지않음.
- 다른온실query/외기/미래시각이querybasis에들어가지않음. daily표본·preprocessing·fit은outer허용inner자료에서만. 회귀계수positivepartbasis군외특징없음.
- family5(SOFT_RESID10/POWER2_ET/POWER2_PARTITION_ET/FARM_ET/본안),전15칸개선,DIAG 온실층화5기록일bootstrap20,000 p_worse<.025/5와99% CI상한<0. 신규채택에는EL1과재현감사필수,공개통과전EL1미사용/소모final lock금지.
- 코드/정합성/inner outer IDs 검사,독립fit 행렬과정규화KKT검사,점수fsum/bootstrap/shape유지식검산. 기준통과여부와고EC/일반/후반오차동시보고. 이후비중·knots·λ검색없음.
- ownlocal/ec_nested_convex_calibration_20261003_v1 OOF only. 제출/test예측/원시EC라벨읽기없음.
