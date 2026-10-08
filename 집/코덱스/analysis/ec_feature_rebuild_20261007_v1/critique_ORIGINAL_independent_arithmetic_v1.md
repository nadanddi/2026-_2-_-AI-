# 독립 Decimal 산술 library 및 합성 v2 비평

2026-10-07 읽기 전용 검토. 모델/추론/정답 숫자/성능채점/GPU 실행 없이 source, 저장된 합성 결과, 실행 source 현재 SHA 및 v1→v2 변경을 확인했다.

## 판단

조건부 고정 산술 재구성 library로서 새 핵심 결함은 발견하지 않았다. 실제 원66 독립 verifier/사전등록은 아직 없으며 이 합성 PASS로 전체 pipeline gate를 열 수 없다. SG2 경계 행의 후속 처리 규칙은 실제 결과 확인 전에 고정해야 한다.

## 확인된 증거

`original_independent_arithmetic_synthetic_audit_v2.json`은 96행×4stage=384 대조, invalid 14개(중복0), prediction-prefix poison 4건을 기록한다. raw mix 최대 차이 1.1e-16, 나머지 stage 최대 차이 1.5082608695652174e-16이다. auditor2/library1/kernel2 실행3SHA 모두 현재 파일과 일치한다. model_fit/heldout_truth_read/whole_pipeline_gate_passed는 모두 false다. 합성 실행을 독립 재실행한 것은 아니다.

auditor1→2 line 단위 차이는 자기 소스명·출력명 버전과 threshold fixture 한 줄뿐이다. 기존 fixture는 current값+.30으로 SG2가 실제 사용하는 prefix 평균 경계를 만들지 못했다. v2는 해당 날짜 h0..1 평균+.30으로 고친다. 이는 거부 시험 fixture의 수정이며 library의 문턱이나 허용오차를 완화한 변경이 아니다. auditor1 실패 기록은 보존한다.

## 산술 독립성

library1은 production kernel을 import하지 않고 Decimal60으로 `.6*(.6ET+.3LGB+.1MLP)+.4*mean(PFN5..8)`를 재구성한다. 같은 farm/day 0..h raw mix 평균을 이용해 shrink .5를 한 번 적용하고, 학습 bounds clip → 공급된 SG2 choice의 level와 clipped prefix 평균 차이 절반 보정 → clip 순서를 대조한다. farm/day 전환에서 두 prefix를 초기화한다.

auditor가 production kernel을 사용하는 것은 비교 대상 출력 생성에 한정된다. 독립 library 계산에는 해당 kernel 객체가 전달되지 않는다. SG2 callback은 실제 source가 아니라 고정 synthetic choice를 이용하므로 source 선택 검증으로 해석할 수 없다.

ID/정렬/완전24h/중복, exact4 PFN seed, exact4 stage, exact choice coverage와 schema, bool/nonfinite, bounds 역전 및 distinct samefarm train reference를 거부하는 분기를 확인했다. 다만 caller가 주는 train_days의 진실성과 선택 level의 학습 label 기원은 검증하지 않는다. train_days tuple 내부 농장/일차 타입까지 엄격 검증하지 않으므로 실제 caller는 원 registry의 정규화된 학습 ID로 exact set을 구성하고 독립 대조해야 한다.

## 경계와 미래 교란 한계

원 SG2는 float에서 abs(delta)<=.30으로 보정한다. library는 Decimal delta가 ±.30으로부터 1e-12 이내면 먼저 거부하므로, 나머지 영역에서 `<.30` 분기는 원 정책과 같고 경계 근처를 조용히 통과시키지 않는다. 향후 실제 원66 verifier는 이 거부를 PASS로 바꾸거나 후보별 tolerance를 조절하면 안 된다. 해당 행이 나오면 등록된 독립 source replay 또는 명시 실패로 처리하고 전체 완료 명단에 반영해야 한다. 지금 실제 경계 행의 존재 여부는 미검증이다.

poison4는 ET의 이후 시각/이후 날짜/다른 farm 값을 바꾸고 지정한 한 행의4stage 불변과 전체 변경 배열의 조건부 Decimal 일치를 검사한다. 모든 이전 행, LGB/MLP/PFN 입력 교란, 실제 query feature 미래 누수, SG2 choice의 미래 독립성을 전수 검사한 것은 아니다. library는 검증을 위해 전체 배열을 읽는다. 수학적 prefix 계산 독립성과 입력 파이프라인의 규정 인과성은 따로 확인해야 한다.

## 실제 적용 전 필수 조건

새 실제 driver/등록에는 exact66 query 명단, 25안×3seed stage 전수 명단, fresh producer/등록/module identity, bounds 학습 label 근거, frozen choice 및 reference membership·level 기원, 전수4stage 분모와 최댓값, 경계 거부 규칙을 연결해야 한다. saved result의4stage뿐 아니라 baseline/candidate 최종 출력이 이 stage와 같은 파일/행인지도 확인해야 한다. full producer/feature causal audit와 원 통계·truth-read gate는 이 조건부 산술 library 밖의 필수 작업이다.

합성 PASS는 새 모델 fit, 후보 성능 개선, 역사적 GPU baseline 동일성, 원66 완료, 후보 채택을 의미하지 않는다.
