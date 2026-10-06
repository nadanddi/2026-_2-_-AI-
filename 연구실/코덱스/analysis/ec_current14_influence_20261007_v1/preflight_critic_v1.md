# 현재 EC14 공개 검증·학습날 의존도 진단 사전 독립비평

2026-10-07 연구실 코덱스 독립 비평가. 검토 대상은 `runner_v4.py`, `PLAN_v2.md`, `sg2_ref_v2.py`, `test_sg2_reference_v2.py`, `real_checks_v1.py`, `preparation_v4.json` 및 원 제출14의 model/sg2post 소스다. 새 학습·후보 선택·성능 열람·원본 변경은 하지 않았다.

## 판정

현재 확인한 범위에서 1단계 실행을 막는 확정 결함은 없다. **사전 감사 통과와 성능 개선/채택은 별개**다. FULL 실수 특징 프레임과 PFN 캐시 RNG·문맥을 독립 전수 재계산한 상태는 아니며, 학습 완료 후 캐시·후처리·채점 전수 감사가 남는다. 2단계 삭제안 구현도 아직 이 검토의 대상이 아니다.

## 독립 확인

- `critic_preparation_verify_v1.py` 정상 종료. 준비 서명의 source 6개 SHA, 10fold의 ID 유일성·train/query 배제·동일 온실 query ±1일 purge·query 하루24행을 확인했다. 총8640행/360일, 일반329일·고EC31일, **공개 pass2 지원46일**이다. 실제 평가 pass2 60일의 전수 효용은 이 자료로 확인할 수 없다. FULL_R347/BASE_R323/PFN38열이다.
- `critic_sg2_preflight_v1.py` 정상 종료. 원 SG2와 adapter의 toy prepare·ref_calendar·correct 96행을 비교해 최종출력 차이0을 확인했다. 기존 prefix불변성 검사의 h≤5 대상12행은 모두 실제 gate=True, delta≠0이었다. 비활성 경로로 통과한 것이 아니었다. 합성자료 검사이며 모든 실제 날짜·결측·경계 경로를 입증하지 않는다.
- 원 `test_sg2_reference_v2.py`는 이름과 달리 원본 동등성에서 prepare만 비교한다. 이 검증 범위 누락은 이번 별도 독립 테스트로 보완했다. 원 실패 코드와 로그는 보존해야 한다.
- 코드상 R3/PFN raw 혼합 후 평활→clip→SG2→clip 순서가 맞다. 실제 ensemble은 세 seed rawR3를 먼저 평균한 별도 경로이며, seed별 최종값 평균으로 대신하지 않는다. PFN4문맥은 동일 train/query38열 hash·캐시 SHA·query ID·RNG context index/row ID를 assert 후 재사용한다. 원 import season 경로/SHA도 준비서명에 들어 있다.
- SG2 weather 표준화는 fold 학습 ref의 pass1만 사용한다. SG2 label ref/bounds도 fold train에 한정한다. 원 전체 제출 기준과 CV의 fold 기준을 구별한 현재 설명이 맞다. `real_checks_v1.py`는 실제 입력의 비계절 특징 불변성과 SG2 현재·이전 입력 경로를 검사하도록 작성됐지만, 해당 실행결과는 이 보고서 작성 시 독립 판정 범위에 넣지 않았다.

## 작업 안에서 보완할 사항

1. `PLAN_v2.md`의 own adapter 명칭 `sg2_ref_v1`은 실제 `sg2_ref_v2`와 다르다. 원문 보존 후 새 보완 기록에 실제 실행 source/버전을 명시하면 된다.
2. 완료 후 독립 검산에서 34560행의 실제 ID×seed/ensemble 유일성·8640행 coverage와 receipt의 기대 숫자를 대조해야 한다. 캐시 재사용 시 NPZ의 train/query ID·길이·유한값도 직접 확인한다. source/prep SHA 일치는 강한 경계지만 실제 배열 검사와 같은 의미는 아니다.
3. receipt의 `classic_fits=90`은 완성 기준의 모델 개수다. 재개 때 새로 실행한 fit 수와 구별해 보고한다. NPZ만 쓰고 sidecar 이전에 중단되면 현재 재개는 assert로 막힌다. 부분 산출물은 보존하고 완료된 쌍만 인정한다.
4. pass1/inactive, pass2/no_candidate, gate=False, gate=True의 분모와 delta/clip 활성량을 분리한다. candidate가 없는 행의 trace NaN을 label 누락이나 오차0으로 해석하면 안 된다.
5. 실제 ensemble과 seed별 최종값 평균의 차이도 전수 보고해야 한다. 현 runner는 실제 ensemble을 올바르게 계산하지만 그 차이의 집계는 아직 결과 분석 단계에 남아 있다.

## 진단 해석의 제한

D1/D2는 **ET 전체 파이프라인을 선택 날 제거 후 다시 학습하는 개입**이다. median imputer와 forest의 분기·잎·무작위 구축 모두 변할 수 있다. 고정 forest의 지원 가중치와 삭제 후 효과는 같은 값이 아니다. label은 다른 구성원·SG2 reference에 남으며 clip bounds/season/reference가 고정되므로 전체 학습자료 삭제 실험도 아니다.

ET 변화가 SG2 gate·candidate 보정의 적용 여부를 바꿀 수 있다. raw ET 변화, 혼합·평활·clip 변화, SG2 delta 변화, 최종 손실 변화를 따로 보여줘야 한다. 최종 개선을 전부 ET 지원 감소에 직접 귀속하면 과장이다.

139/231과 161은 이전 공개 정답·원인 추적을 보고 선택된 사례다. 선택 사례 의존도와 잔여 혼동 판정은 기술적 진단이며 독립적인 개선 증거가 아니다. 현재 EC14 전체360일·seed별·실제 ensemble·일반/고EC·61/268·pass1/pass2·농장·161 제외 나머지 결과를 모두 확인한 뒤에도 채택이나 물리 원인으로 확대하면 안 된다. 원본 정답/test/EL1/잠금 자료를 새로 읽지 않는 제한과 기존 위험 후보 REJECT는 유지한다.

결과 전 사전 검토는 완료했다. 효용·채택 판단 및 결과 전수 감사는 미완료다.
