# Claude 최신 PF1/PF2 출력 독립 범위 비평

2026-10-07. metadata inspector3/저장 receipt/PF1·PF2·AF0·SG2 소스 및 파일SHA만 읽었다. 숫자 정답 열람·재채점·모델·GPU 실행0이다. 원CSV hash는 정답 문자열/bytes를 읽지만 숫자 변환·성능에 쓰지 않았다.

## 최신 가용성 정정

현재 metadata3는 PF1/PF2 각각10848행/66fold, DIAG8640/P2LOO1104/EL11104/TM2664, expected key 누락0·중복0·finite prediction/bounds PASS를 기록한다. 공유10848 key의 A~D/R3/bounds 최대차이는4.440892098500626e-16이다. 저장 inspector/code/CSV 등11개 pin의 현SHA가 모두 일치한다.

이전 PF2 6648행/34fold는 이전 시점 스냅샷이다. **현재 PF2 전체 가용성을 부분 상태로 계속 설명하면 오류다.** 다만 metadata 전체 가용성은 실제 모델 실행 환경·개별 fit 완료·규정 준수·성능 산술 독립 검증을 입증하지 않는다. 공유 예측 일치는 소스가 의도한 A~D/R3 재사용과 부합하며 독립 재학습 일치 증거가 아니다.

## 소스상 인과성·검증 경계 문제

1. PF1 `pfn()`은 `device='cuda'`와 default TabPFN create→fit→predict 경로다. `fit_with_cache`/kv_auto/학습전용 internal-stat trace가 명시되지 않고 query는 최대2000행씩 같이 predict된다. 앞서 현 TabPFN default uncached 경로에서 발견한 all-query 통계 위험과 같은 종류의 우려다. CSV에서 어떤 library/default/run환경으로 실제 생성됐는지 추정할 수 없으므로 최신 output의 실제 누수 크기나 cached 동등성까지 확정하지 않는다.
2. SG2 `prepare_structure()`는 fold 전에 train_X+test_X를 합치고 pass1 전체 weather의 mean/std를 계산한다. reference-only라고 설명한 calendar/signature 일부와 별개로 이 scaling은 평가 입력을 fit에 사용할 수 있는 구현이다. 이는 source상 구체적인 경계 위반이며 query input 교란/학습-only scale 감사 없이 causal PASS로 인정할 수 없다.
3. D/E reference는 `labset-vd`다. 모델 train은 validation±1/lock±1 purge로 더 좁다. 따라서 ref에는 그 fold 금지 purge 기록이 다시 들어갈 수 있고 reference scaling/calendar/anchor EC에서 같은 fold train 계약을 유지하지 않는다. validation 자신의 EC는 ref에서 제외되므로 **직접 query-label 사용이라고 단정하지 않는다.** 그러나 원purged validation과 같은 검증이라는 주장도 성립하지 않는다. lockd의 anchor 후보 제외는 calendar/scale 참조 제외와 같지 않다.
4. PF2의 cumulative difference 자체는 각 record 0..h expanding이라 현재hour 접두 관측을 쓰려는 구조다. 그것만으로 best-anchor/calendar/global-scale와 upstream p3.prepare/season의 전체 경계를 증명하지 않는다. train MASK, query same-farm current/past, recursive previous record fallback 및 금지 input 교란 검사는 실제 전체 flow에 필요하다.
5. 각 fold checkpoint는 파일 존재만으로 skip하고 source/environment/context/matrix/params/trace/완료 receipt를 재검증하지 않는다. 현재 CSV key 전체가 맞더라도 현 소스가 그 출력을 만들었다는 실행 lineage를 복원하지 못한다.

## 현재 CPU cached baseline과의 비교 범위

Claude 소스 blend는 WT0 저장 R3와 각 shrunk PFN을 혼합·clip하며 **final SG2 호출이 없다.** 현재 Codex 원66 prospective baseline은 train-only CPU cached PFN+fresh fold R3+고정 RAW_PASS SG2/fullgate다. weight(.6 R3/.4 PFN), 일부 seed/열수가 같아도 같은 기준선이 아니다. raw member shrink 저장 정책 역시 WT0 source/receipt를 확인해야 전체 shrink한번 동등성을 주장할 수 있다.

통계도 다르다. PF1/2 source는 seed별 방향 뒤 **3seed 예측 평균의 squared loss**를 bootstrap하고20k를 전체 farm×5day bins에서 표집한다. 현재 원domain은 seed별 squared-loss 평균, farm별 층화200k, DIAG/TM 교집합과84비교를 고정했다. Claude 결과를 그 규칙의 p/채택 증거로 이식할 수 없다. 이 리뷰에서는 Claude 점수를 재검산하지 않았으며 보고된 성능 방향·유의확률을 독립 확인했다고 말하지 않는다.

## 허용할 해석과 다음 작업

6.384/6.385는 이 GPU/default/reference/blend 구성에서 저자가 보고한 기각 결과로 보관한다. '이웃 정보 일반이 해롭다', '차이 정보를 배우지 못한다', '문맥 확대가 항상 해롭다'는 보편 명제는 구현·검증/평균화 조건 때문에 과장이다. 특히 D/E generic neighbor 특징 시험과 CH2 fixed-flank/source/prefix 방법 또는 바닐라 학습 원자료의 가용성은 서로 다른 범위다.

현재 generic neighbor 추가 fit를 이 보고만으로 재개하거나 원domain24을 제외할 근거는 없다. 원24 계획을 유지하고 latest PF2 전체 가용성만 정정한다. 나중에 이 출력으로 비교·채택하려면 새 source/runtime/model/context/causal 계약과 동일 cached/full baseline을 먼저 맞춰야 하며, 기존 파일은 보존한다. 원 최신CSV만으로 runenv/fit완료 receipt를 생성하거나 무정답 재현 완료라고 추정하지 않는다.
