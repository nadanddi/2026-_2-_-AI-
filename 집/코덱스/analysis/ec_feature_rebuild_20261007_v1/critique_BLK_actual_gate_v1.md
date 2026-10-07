# BLK 실제 cached baseline gate 독립 비평 v1

2026-10-07. analysis-verification 스킬을 적용하여 저장 증거를 PowerShell JSON/별도 산술/Get-FileHash로 독립 재대조했다. 모델·pipeline·scorer 실행, 보류 EC 값 열람, GPU 사용 0. 원자료 파일은 SHA만 확인했다. 기존 리뷰 파일은 수정하지 않았다.

## 현재 진단 채점을 막는 핵심 결함 발견 여부

현재 읽은 소스와 실제 저장 산출물에서 새로운 핵심 누수·수치 실패·불완전 cached context 우회는 발견하지 못했다. 현재 상태의 BLK 진단 채점에 대한 source/receipt gate 근거는 갖춰졌다(신뢰도 높음: 저장 증거 및 현 source의 일치에 한정). 모델 성능 개선·후보 채택·원 제출 재현 또는 전체 목표 완료의 허용은 아니다.

## 실제 증거 독립 재대조

- execution verify returncode=0, frozen_sources_unchanged=true, heldout_score_stage=false를 읽었다. 별도 gate source SHA를 현재 verify v4 파일과 대조했다.
- gate의 절대경로 transitive 69개 파일 SHA를 직접 모두 재계산했으며 불일치0. prediction/assembly audit/registered methods/PFN weight·library/R3 sources 등 현재 연결이 일치한다.
- cached context5/6/7/8 각각 status PASS, 정확18 검사, 최대 차이0.0, fit4/predict104,4cache 전후 snapshot 동일을 다시 집계했다. 전체 receipt는 CPU-only verified=true/cuda build null이며 기존 원 tensor device 미기록 한계가 명시돼 있다. 실행된 모델의 수치를 이 리뷰에서 다시 예측한 것은 아니다.
- assembled ID1440개/유일1440개,baseline6개,candidate tag6개,guard_active0,후처리 경계288개 기록을 직접 집계했다. assembled prediction SHA는 `357bd921c3ba74ea89aba45755fae5ca3b169947806a6b6a88431612f56020ba`로 현재 파일과 일치한다.
- guard candidate와 baseline의6×1440=8640개 값 불일치0을 독립 비교했다. source뿐 아니라 저장된 fallback도 확인됐다.
- stored raw_mix에서 seed3×1440=4320행의 same-record prefix 평균 shrink를 PowerShell로 독립 계산하여 stored one_shrink와 최대4.440892098500626e-16 차이였다. gate의 stdlib raw-member/mix/shrink/clip 대조 최대1.1102230246251565e-16과 비교한 대상이 다르므로 두 수치가 동일한 검사라고 표현하지 않는다. 이 리뷰는 clip 경계용 공개 학습 라벨도 읽지 않았다.

## 누수·재현 범위

공식 PFN feature cache를 학습2000행에서만 만들고 예측 중 mask/group fit을 금지하는 runtime trace, cache 통계·KV·target snapshot 불변, 작은 batch/순서/other-query poison/repeat full 수치 감사가 연결됐다. 상위 raw feature/calendar prefix 및 R3/SG2/endpoint 경계는 별도 소스·감사로 연결한다. train reference 공개 EC/sub_temp와 query/gap 정답을 구별하는 BLKContext 계약을 유지한다. 모든 가능한 입력 또는 실시간 배포 환경에 대한 보편적 인과 증명은 아니다.

새 gate는 cached recipe가 BLK layout/정의 아래 진단에 적합하게 연결되었는지 확인한다. pass1만 있는 BLK에서 raw-pass gate의 SG2는 skip이고 QUERY_ROLE은 별도의 평가 역할 적용이다. 숫자 day를 pass2로 변경하지 않았다. 실제 pass2 일반화와 원 검증기 효과는 아직 미입증이다.

## 원 제출 동일성 주장 제한

원 제출14는 .8R3+.2PFN 및 다른 seed/context를 사용한다. 여기의 최신 recipe는 .6R3+.4PFN, R3 seed47/1414/6464·PFN context5~8, BLK reference context와 CPU 실행이다. 더구나 기존 uncached 내부 query-stat fit을 학습 전용 cache로 바꿨다. 같은 사전학습 weight와 특징 recipe라는 점은 원 제출 값/역사적 GPU OOF와 수학적·byte 동일성을 뜻하지 않는다. gate.scope가 이 제한을 정확히 기록한다(신뢰도 높음: source와 registration 차이).

기존6.373/374 또는 uncached single-row FAIL을 지우거나 이번 PASS로 소급해 정상화하면 안 된다. 변경된 baseline 위에서 같은6variant를 새로 진단하는 것이다. guard0은 이 좁은 global EC mutual rule의 비활성 상태이며 CH2 전체 사슬 단서 실패의 증거가 아니다.

## 좁은 잔여 source pin 메모

SG2 adapter가 AST로 추출해 실제 사용하는 원 `집/클로드/submission14_ec_sg2/sg2post.py`는 source-equivalence audit의 source SHA로 gate 생성 시 검사하지만 현재69개 transitive 목록에는 없다. 이번 리뷰에서 그 audit SHA와 현재 sg2post.py SHA를 별도로 재대조하여 일치를 확인했다. 같은 조건으로 R3 future audit producer source 등 감사 생성 소스의 장기 재현 pin도 남는다.

현재 불일치·누수를 발견한 것은 아니므로 완료된 모델 fit을 다시 하거나 진단 전체를 기각할 사유는 아니다. 채점까지 source를 고정하고, 가능하면 실행 preflight의 frozen 목록에 sg2post.py를 포함시키거나 정답 읽기 직전 audit.source_sha256와 직접 재대조하면 이 작은 변경 감지 공백도 닫힌다. 현재 정상 저장 증거에 대한 허용과 향후 파일 변경을 자동 감지하는 완전성은 구별한다.

## 채점 후에도 남는 것

BLK6variant 선별과8block bootstrap은 전체196후보 탐색, 원 TM111/P2LOO/EL1/DIAG 검증, 사전 고정 미사용 seed/layout 최초1회 판정을 대체하지 않는다. 공유 reference와 시간 의존 때문에 bootstrap p를 강한 외부 일반화 확률로 해석하지 않는다. 성능 채점은 본 리뷰에서 0이며 후보 개선·순위 결론을 제시하지 않는다.
