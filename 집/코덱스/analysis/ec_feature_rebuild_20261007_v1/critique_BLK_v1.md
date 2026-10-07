# BLK 배치 독립 비평 v1 — 2026-10-07 집 코덱스

판정: **ID·라벨 가용성만으로 만든 구조 등록으로는 유용하지만, BLK 실행 허용은 아직 보류.** 숫자 정답으로 배치를 고르지 않았다는 코드를 확인했다. 신규 후보 학습·성능 채점을 하지 않았다. 새 BLK 요구를 최우선으로 검토하되 기존 TM/P2LOO/EL1은 유지한다.

## 구조 확인

blk_layout_v1.py와 BLK_layout_v1.json을 읽고 저장 ID를 독립 집계했다. 8블록, query1440행, 양측 gap384행, train6816행, endpoint flank1152행이다. 온실별 query 길이5×2·10×2이고, 각 flank는3기록이다. 현재 선택8블록은 모두 pass1이다. 후보 pool에는 pass2가 네 온실×길이 조합 모두0개다. F13에는 cross_pass가 길이5·10에 각1개, F47에는0개다. **2차 평가형 구조가 가능하다고 표시하면 안 된다.**

## B01 · P1 · 역사 locked ±1 규칙이 v1에서 빠짐

eligible_days=available-locked, training=available-hidden-gaps-locked이므로 잠금일 자체만 제외한다. 독립 ID 대조에서 기존 locked±1에 해당하는 기록 **54개**가 train에 남고, 그중 **F13_046·F13_166 두 기록은 고정 flank**다. 역사 기준선 및 v2 fold policy의 locked±1와 다른 구성이다.

조치: 숫자 정답/점수를 보기 전에 locked±1를 유지할지 명세한다. 유지하면 required support도 locked±1 제외집합으로 검사하고 결정적 backtracking을 다시 수행한 v2를 등록한다. 기존 flank가 소실되는 것을 fit 후 수정해서는 안 된다. 사용자 BLK 형상은 query±1을 gap으로 만들므로 그 둘은 삭제하고, query±2의 고정 학습 endpoint를 추가 purge해 없애지 않는다. '역사 locked 주변 purge'와 'BLK 양측 gap'은 다른 규칙이다. 선택불가면 STRUCTURE_INFEASIBLE로 멈추고 형상을 몰래 완화하지 않는다.

## B02 · P1 · raw gap 삭제와 query 접근 차단은 아직 문자열 계약

gap 삭제 제약은 올바르지만 현재 코드는 ID를 만들 뿐 raw loader·보조학습·달력·표준화·사슬·검색·후처리에 실제 차단을 적용하지 않는다. fixed public training endpoints는 뒤 날짜라도 참조할 수 있지만 뒤 **query** 입력은 모든 처리에서 금지다. test형 query1440행을 full-day 특징 생성기에 한꺼번에 넣으면 날짜/쌍/사슬 선택에서 우회누수가 발생할 수 있다.

조치: raw TRAIN store에는 train_ids만, query store에는 해당 행까지의 동일온실 prefix만 제공하는 loader를 구현한다. gap384행은 입력과 양 정답을 저장소 생성 시 제거하고, query 양 정답은 채점용 격리 store로만 둔다. 학습 endpoint 사슬/통계/모델은 query와 독립적으로 한 번 고정한다. gap 입력/정답을 임의 교란해도 모든 결과가 같고, 뒤 query 입력/정답/예측을 교란해도 앞 예측이 같아야 한다. 다른온실·순서·single-query 감사까지 최종 output에서 확인한다. source feature availability MASK도 유지한다.

## B03 · P1 · 고정 양끝은 존재하지만 지정 anchor 값 계약 없음

flank_ids1152행만 등록되어 있다. 각 블록의 왼쪽 gap 이전 마지막 학습일23시, 오른쪽 gap 이후 첫 학습일0시를 명시적으로 고정한 endpoint row_id·참조 가능 라벨·train membership·사슬 fit provenance가 없다. query를 넣은 뒤 체인을 재구성하거나 endpoint를 residual에 맞춰 재선정하면 고정 양끝 실험이 아니다.

조치: block_id, left_endpoint_id=(flank_left 마지막날23시), right_endpoint_id=(flank_right 첫날0시), support label ID, 사슬 코드/설정 SHA를 fit 전에 봉인한다. endpoint 값은 공개 train 정답 store에서 읽되 query/gap 값을 제거한 뒤 수행한다. 후보마다 동일 endpoint와 query1440행을 사용하고 없는 연결은 명시적 abstain+baseline fallback으로 처리한다.

## B04 · P1 · 정답 사슬 검증의 순환성

train EC 자정 차로 만든 사슬에서 자정 차가 작고 EC가 연속적인 것은 구성상 발생한다. 그 수치를 query 연결 정확도나 새정보의 증거로 쓰면 순환이다. query 정답에 가장 가까운 사슬/endpoint를 골라 성공률을 내는 것도 오라클이다. BLK의 row_id 연속은 배포 기록 순서일 뿐 물리적 출처 연속이 확인된 것은 아니다.

조치: 후보 선택/신뢰 게이트는 train 내부 nested pseudo-block에서 개발하고 query labels는 최종 채점에만 쓴다. train 체인 적합 품질과 heldout block 예측 성능을 분리한다. true query label로 endpoint를 고른 값은 명시적 오라클 진단으로만 보존한다. 연결 가능한 coverage·abstain·양끝 신뢰 충돌·출처 불명 상태를 보고한다. 블록 평균 수준과 일내 모양 오차도 나눠 평가한다.

## B05 · P2 · pass2 불가능과 cross_pass를 섞어 해석하지 말 것

v1의 pass 표시는 query 범위만으로 정한다. 후속 버전에서는 flank/gap까지 포함한 support 전체 pass도 기록해야 한다. cross_pass 후보를 pass2 확인자료로 쓰면 정의가 달라진다. 실제8블록은 pass1이며 SG2의 pass2 적용 이득과 2차 평가 일반화를 이 배치만으로 확인할 수 없다.

조치: 숫자 정답을 보기 전에 cross_pass 허용 여부를 고정하고 selected pass1/pass2/cross별 카운트를 보고한다. 완전 pass2 후보0을 그대로 남긴다. 순수 구조 배치 실험의 효과와 P2LOO/EL1 효과를 함께 보고하며 기존 validator를 자동 교체하지 않는다. 신규 배치도 기존 관찰 정답이므로 미관찰 독립 검증이 아니다.

## 최대3개 후보 권고

1. **PAST_ENDPOINT:** 왼쪽 고정23시 EC만 기준선 수준 보정에 사용. BLK 학습 내부 pseudo-block에서 고정된 보호/혼합 규칙을 쓰며 가장 단순한 양끝 실험 대조 역할을 한다.
2. **BOTH_ENDPOINT:** 같은 양끝의 공개 EC만 이용한 고정 기록거리 보간/수준 보정. 기록거리 보간은 물리 연속성 가정이므로 검증할 가설이지 사실이 아니다. 보호/혼합은 사전 고정하고 query 정답으로 조절하지 않는다. 앞/뒤 endpoint 기여를 별도 기록한다.
3. **CHAIN_PREFIX_GUARD:** train-only 정답 사슬과 양끝을 고정한 뒤 현재·이전 입력 prefix로 연결 신뢰를 판단, 불확실하면 baseline fallback. 부모의 PF1/PF2 중복 여부를 먼저 확인하고 같은 방법이면 재학습하지 않는다. 게이트/가중치 학습에는 outer BLK query와 gap 양 정답을 절대 넣지 않는다.

후보3개는 알고리즘 수이며 내부 가중치/보호 threshold를 여러 개 대조하면 각각 variant로 등록한다. 지정 baseline BLK 재학습/재현과 leakage audit가 완료된 뒤 동일3seed·동일8블록 비교를 진행한다. BLK는 블록이8개뿐이므로 행수1440을 독립표본수로 보고하지 말고 온실×블록 단위 손실과 paired 불확실성을 보고한다. 최종 채택은 사용자 고정 기존 validator/다중비교 조건을 유지한다.

부모에 대한 최우선 조치: **locked±1 정책을 먼저 확정한 새 ID배치와, raw gap 삭제·future-query 차단 loader 검증을 끝내기 전 fit을 열지 말 것.**
