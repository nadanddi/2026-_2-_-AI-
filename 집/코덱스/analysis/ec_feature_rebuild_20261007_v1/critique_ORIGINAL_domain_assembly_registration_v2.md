# 원66 도메인 조립 v2 실제 등록 독립 비평

검토일: 2026-10-07. 범위는 등록·소스·저장 합성 결과와 현재 파일 SHA의 읽기 전용 대조이다. 조립, 모델, query 추론, 평가 정답 숫자 열람, 채점, GPU 및 진행 중 worker 개입은 하지 않았다.

## 판정

고정 recipe나 조립 소스에서 새 모델·누수 핵심 차단 결함은 찾지 못했다. 다만 아래 AR01은 **합성 검사 실행 증거와 현재 소스의 연결을 실행 전에 보완할 항목**이다. 현재 등록은 전체 모델/채점 게이트가 아니며 원66 producer가 모두 완료되기 전 조립을 실행할 수 없다.

## 실제 등록 연결

- `ORIGINAL_DOMAIN24_pipeline_registration_v2.json`의 257개 source pin은 현재 파일과 모두 일치한다. 기본환경에서 pandas 단일 파일 읽기만 거부되어 공식 escalated 읽기 해시를 별도 대조했다. pandas SHA는 `3f0e93be963b0ccf84860992e2b9c5597132e1fbb51a14ba749a1b4f1b0df14b`로 일치했다.
- 이전 fresh-probe 등록의 243핀은 충돌 없이 상속된다. 현재 assembler SHA도 등록 값과 일치한다. 이는 파일 연결의 확인이며 등록 명령의 실제 런타임 실행 전체를 독립 재현한 것은 아니다.
- assembler는 실행 시 실제 로드된 11개 모듈의 `__file__` 경로와 등록 SHA를 대조한다. 지금은 이 코드 분기를 확인한 상태이며 조립 실행에서 11개 actual import가 관측되었다고 확대하지 않는다.
- raw66/5346과 PFN66/528 정확 manifest를 특징·학습 label 로드 전에 요구한다. PFN은 264 context의 prediction/audit 두 파일씩 528개다. docstring의 PFN264는 context 수이며 파일 수가 아니다.

## AR01 — 합성 receipt와 실행 코드의 직접 연결 누락

합성 receipt에는 검사 auditor와 checkpoint helper의 실행 당시 SHA가 없다. registrar2는 status, check_count=11 및 whole_pipeline_gate_passed=false만 검사한다. 따라서 예전 PASS receipt를 보존한 채 auditor/helper를 바꾸고 새로운 현재 SHA를 핀해도 이 검사가 통과할 수 있다. 현재 257 SHA 일치는 이 시간적 연결을 대신하지 않는다.

새 이름의 실행 receipt나 독립 crosscheck evidence에 auditor/helper SHA, 정확한 11개 check 집합, model_fit=false, heldout_truth_read=false를 함께 기록하고 현재 소스와 대조하는 보완을 권고한다. 원 핀 파일과 등록 v2는 보존한다. 이 지적은 알려진 실제 exit0/11검사를 부정하는 것이 아니라 재개·이동 시 기계적으로 검증할 수 있는 증거 연결의 부족이다.

## 고정 산술·규정 경계

등록 및 kernel2 구현은 `.6*(.6*ET+.3*LGB+.1*MLP)+.4*mean(PFN5..8)` 다음 같은 농장·날짜 0..h에 대해 shrink 0.5를 한 번 적용한다. 학습 EC 범위 clip, RAW_PASS SG2 day>=179, 마지막 clip 순서다. 도메인 후보는 ET만 교체하고 LGB/MLP/PFN과 시드47/1414/6464는 고정한다.

kernel의 SG2 callback에는 현재 날짜 예측 prefix만 전달된다. train-only 참조 선택은 별도 SG2 plan의 책임이다. 전수 vector 유효성 검사는 전체 예측 배열을 읽으므로 이 kernel만으로 streaming 입력 인과성을 입증하지 않는다. baseline 각 행×3seed source 재현은 candidate 전수 독립 산술·미래 교란 감사를 대신하지 않는다. 역사적 GPU/uncached 제출과 수치 동일성도 주장할 수 없다.

saved prediction, fold audit, complete 및 등록에 whole_pipeline_gate_passed=false가 유지된다. 독립 전체 산술·입력 인과성·producer 수치 검증과 truth-read 이전 scorer gate가 별도로 필수다. 통계 source를 핀한 것은 채점을 실행하거나 통계 기준을 바꾼 것이 아니다.

## 재개·저장 감사의 범위

합성 11검사는 기존 pair/부분 pair/complete의 정확 재개와 바이트 보존, 잘못된 prediction/audit·invalid JSON·bool/int 차이 거부, 소유 lock 성공/예외 해제를 다룬다. 두 번째 writer 거부는 같은 process에서 중첩한 검사이며 실제 두 process 경쟁이나 crash injection 증거가 아니다. foreign-token 보존, 추가 JSON 거부 등의 일부 분기는 소스상 방어가 있지만 이번 11건의 직접 검사에는 없다.

조립 v2는 fold를 항상 fresh 재계산하고 기존 두 객체와 비교한 뒤 누락만 저장한다. POSIX 명단132개, producer 출력/root·fold complete 및 조립 출력/등록 source 최종 SHA를 재검사한다. O_EXCL lock과 자동 stale 삭제 금지는 적절하다. 부분 lock 쓰기 실패, leaf symlink, 연속 파일 검사 사이의 동시 변경은 원자적 snapshot으로 완전히 차단되었다고 주장하지 않는다. 완성 producer를 불변으로 보존하고 하나의 조립 writer만 실행하는 운영 전제가 남는다.

전체 목표, 원66 모델 완료·수치/인과 게이트, 기존 TM111/P2LOO/EL1 검증 및 최초 미사용 확인은 완료되지 않았다. 이 문서는 조립 등록 검토만 다룬다.
