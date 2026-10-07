# BLK cached PFN 통합 독립 비평 v1

2026-10-07. receipt v2 / method registration v4 / assembly v4 / baseline verifier v3 / scorer v5 및 연결된 preflight·SG2·endpoint 소스를 읽었다. 코드 실행·모델 실행·보류 EC 정답 읽기·성능 채점·GPU 사용 0. 실제 4cached contexts/assembly/gate PASS를 확인한 리뷰가 아니다.

## IN01 — 채점 직전 transitive source pin이 불완전 (P1, 정답 읽기 전 수정 필수)

gate v3는 생성 시 R3 registration/dependencies/sources, PFN weight/library/import/matrix, layout/context, method_code_sha256를 확인한다. 그러나 반환 gate의 `verified_evidence_sha256`와 `verifier_lineage_sha256`가 이 모든 파일을 포함하지 않는다. scorer v5는 두 dictionary와 cached receipt의 저장 output/audit/registration/complete를 재해시하지만, 현재 method_code dictionary, R3 원 registration/dependencies/sources, PFN library/runtime/weight, layout/context, assembly audit 등을 전부 다시 검증하지 않는다.

따라서 gate 생성 이후 해당 미포함 파일이 바뀌어도 scorer의 label-read 직전 검사에서 모두 차단된다고 보장할 수 없다. 이는 현재 파일이 변조됐다는 발견이 아니라 fresh current-source 계약의 누락이다. 저장된 예측값 자체는 고정돼 있어도 기존 감사의 현재 소스 근거가 낡을 수 있다.

수정 방법: 새 gate에 현재 확인한 모든 transitive source/원자료/등록/weight/layout/context/assembly audit의 exact path→SHA를 모아 scorer에서 재대조하거나, scorer가 정답 읽기 전에 fresh receipt helper+method/R3 source 검증을 호출하여 gate 당시 receipt와 일치시키라. 누락 없이 명시 목록을 사용하는 것이 중요하다. 모델 재fit이나 보류 정답을 읽을 필요는 없다. scorer 자신의 source SHA도 실행 전 확정한 spec에 연결하고 기존 버전은 보존한다.

## 통합의 적절한 부분

- receipt v2는 정확7 library/module 키와 실제 import path를 강제하며 4개 output/audit/complete와 정확18 검사·trace·matrix SHA를 검증한다. KV는 train_shape=[1,2000], 정확12 layer 및 key/value FP32/shape 제약으로 강화됐다. complete 문자열만으로 incomplete context를 우회하지 않는다. 해당 helper 실제 PASS가 필요하다.
- assembly v4는 새 cached 출력만 읽고 fresh helper 반환과 별도 receipt를 비교한다. 옛 uncached 출력은 baseline 조합에 들어가지 않는다. 같은 2000 context/38FULL/weight/seed5~8/n_est4라는 recipe와 train-only internal stats 정책 변경을 구별한다.
- 순서는 seed별 R3=.6ET+.3LGB+.1MLP, PFN4문맥 평균, .6R3+.4PFN, shrink1회, 공개 reference train min/max clip, SG2, clip, endpoint 후보 및 최종 clip이다. raw member 조합 후의 shrink를 독립 fsum prefix로 대조한다. shrink를 미리 적용한 값을 다시 축소하는 경로는 새 assembly source에서 보이지 않는다.
- SG2는 complete0..h prediction prefix만 받으며 endpoint는 현재·이전 입력 prefix만 받는다. 양끝 공개 train anchor EC는 query 정답과 구별된다. future/다른 farm query 입력 교란288개에서 후처리 변화가 없는지 검사하는 코드가 있다. 이 검사는 fresh PFN engine의 query 통계 독립 source/수치 감사와 상위 feature/calendar 감사를 대체하지 않는다.
- RAW_PASS의 pass1 SG2 skip 덕분에 gate의 stdlib raw mix→shrink→clip 독립 비교가 성립한다. 고정 layout은 모두 pass1이다. 다른 layout에 재사용하려면 RAW_PASS skip 전제를 명시 assert해야 한다. 이 gate는 QUERY_ROLE의 SG2 및 모든 endpoint 수치를 별도 구현으로 전체1440 재계산하는 검사가 아니라 source·기존 source-equivalence·288경계 검사를 연결한 것이다.

## 통계·채점 검토

PowerShell로 registration v3→v4의 methods/fixed settings/seeds/normal-high split/position/statistics/screened_variant_count 7필드가 동일함을 독립 비교했다. scorer v3/v5의 bootstrap 및 손실 핵심 코드를 읽어 동일한 계산임을 확인했다.

손실은 mean prediction의 제곱오차가 아닌 각 행 seed3개 SE 차이의 평균이다. farm별4블록씩 복원추출한8블록의 SSE합/행수합으로 재표집하여5/10일블록을 행수에 따라 가중한다. bootstrap20000, rng2026100702, ties>=0, plus-one p/20001,6variant alpha.025/6, seed3개 모두 전체RMSE 개선 기준은 고정돼 있다. 일반/고EC 및 앞/가운데/뒤는 진단이며 기준을 결과를 보고 고르지 않는다. individual95% CI는 다중비교 보정 CI로 오해하면 안 된다.

scorer는 gate 후에만 query EC 값을 읽고 duplicate/finite/1440개 완전존재 및60일×24h를 확인한다. 모든1440행 baseline/candidate를 채점하여 fallback 행을 제외하지 않는다. guard0이면 guard 후보는 baseline과 같으며 p=1/all_improveFalse가 된다. 좁은 guard0 결과로 CH2 전체 학습사슬 가능성을 기각할 수 없다.

block8개만으로 공유 reference·시간 상관을 제거하지 못한다. p_worse는 이 고정 bootstrap 아래 진단량이며 BLK_screen_pass는 최종 채택 기준이 아니다. 6variant 교정이 기존196후보/전체 탐색의 다중비교 문제를 해결했다고 주장하면 안 된다. TM111/P2LOO/EL1/DIAG 원검증기와 미사용 seed/layout 최초1회 판정은 별도로 남는다.

## 한계 및 운영 메모

CPU-only wheel+cudaNone 및 명시CPU가 원 tensor device 미기록과 별도 근거라는 한계를 정확히 유지한다. 보류 정답미열람 표시와 실제 label-read source 경계를 구별한다. 캐시 snapshot 검증은 새 모델 재fit 독립재현이 아니다. 실패/partial scoring 시 spec/result 파일을 덮어쓰기보다 새 버전과 label-exposure 기록을 유지해야 한다.

기존6.373/374와의 차이는 uncached query-stat 경로를 학습 전용 cache로 고친 새 baseline이다. 같은 method/통계에 대한 새 기준선 진단이며 옛 numerical FAIL을 지우거나 역사적 성능 재현이라고 표현하지 않는다. IN01을 닫고 실제4cached context 및 전체gate 증거를 확인하기 전 채점을 허용하는 리뷰가 아니다.
