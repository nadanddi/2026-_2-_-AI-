# BLK 독립 재비평 v2 — 2026-10-07 집 코덱스

판정: **BLK v2 구조와 loader 범위의 감사 PASS. BLK baseline·3후보의 사전 규칙 등록으로 진행 가능. 전체 모델 인과성·기준선 재현·효과 채택 PASS는 아님.**

BLK_layout_v2.json, blk_context_v1.py, BLK_context_audit_v1.json을 읽었다. 별도 읽기 전용 실행에서 loader를 실제 생성해 1440 query의 prefix를 전수 재확인했다. 독립 ID 집계: train5520/query1440/gap384, 역사 locked±1과 train 교집합0, 모든 selected support와 역사 locked±1 교집합0, 고정 양끝8쌍 모두 공개 train labels에 존재, query/gap 정답 공개0, gap의 입력/라벨 store 노출0. v1의 핵심 locked 주변 노출은 해결됐다.

기록된1560검사는1440 prefix 확인과120 future/other-farm 입력 교란이다. 코드와 범위가 맞는다. 본 독립 재실행은1440 prefix 전수와 store/anchor 구조를 확인했고120 교란을 다시 실행한 것은 아니다. loader는 ID를 걸러낸 뒤14원열과 라벨을 숫자로 읽으므로 gap 관측 숫자를 파싱하지 않는다. 파일의 CSV 문자열 토큰 자체는 순회하므로 '파일에서 읽지 않음'이 아니라 'gap을 관측으로 파싱·전달하지 않음'으로 표현한다.

남은 조건:

1. **최종 prediction까지 인과 감사:** loader 밖 원본 CSV 직접 읽기, `_query` 전체 접근, full-day 달력/검색/사슬 연결, query 기반 전처리 갱신을 candidate 코드에서 차단해야 한다. loader PASS만으로 전체 처리 PASS를 선언하면 안 된다. BLK baseline와 각3후보에서 future-query 입력/정답/예측 교란, 다른farm, 순서, 단일질의 불변성을 다시 확인한다.
2. **고정 reference 불변성:** packet이 reference dict를 직접 반환하므로 후보가 값을 바꾸면 다른 질의 문맥도 달라질 수 있다. 학습 입력/라벨/사슬/통계의 fit 후 SHA를 저장하고 매 predict 전후 불변성 검사 또는 읽기 전용 copy를 적용한다. 현재 검사에는 이런 mutation/update 거부가 없다.
3. **실행 계약 pin:** 현재 원본파일/loader/layout/모델 의존 SHA와 ordered train/query/anchor IDs를 실제 fit 직전 검증한다. score-only query labels는 context와 다른 격리 경로에 둔다. 후보별 수식·가중치·보호 threshold·fallback·수정 구성원을 baseline 점수/후보 점수 전에 등록한다.
4. **2차 적용 범위:** pool의 pass2/cross_pass는 전부0이며 선택8블록은pass1이다. 특히 SG2가pass2 전용이면 BLK에서 그 효과가 비활성일 수 있으므로 동작률을 보고한다. BLK 이득만으로 실제2차 평가 이득이나 공개 정답 사슬의 물리적 진실을 입증하지 않는다. TM/P2LOO/EL1 유지와 각구간 효과를 별도 확인한다.
5. **사슬 순환성/표본수:** train-only 정답으로 체인을 구축한 품질은query 배정 정확도 증거가 아니다. QUERY정답으로 anchor·연결·게이트를 고르지 않는다. 불확실한 연결은baseline fallback과coverage로 보고한다. 1440행보다8블록의 의존 단위를 강조하고 단독/조합 변형 수를 다중비교 장부에 남긴다.

다음 최우선 순서는 BLK v2 frozen layout/loader로 baseline과 PAST_ENDPOINT·BOTH_ENDPOINT·CHAIN_PREFIX_GUARD 최대3방법의 정확한 규칙을 등록하고, baseline 전체 단계 재현·인과 감사 후 동일 조건으로 실행하는 것이다. 기준선 재현과 후보실행의 상태를 나누고 기존3검증기를 대체하지 않는다. slow/fast 특징 감사는 별도 기능 증거로만 취급하며 본 재비평에서 그 결과를 독립 재검산하거나 전체baseline 효과로 확대하지 않았다.
