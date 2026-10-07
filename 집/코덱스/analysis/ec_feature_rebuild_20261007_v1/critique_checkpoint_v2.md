# 독립 중간 비평 v2 — 2026-10-07 집 코덱스

판정: **ET 한 셀 재현 및 체크포인트 합성 검사는 제한된 범위의 PASS. 정식 후보 개선 비교는 계속 차단.** 신규 후보 학습을 수행하지 않았다. 지정한 v2 계획·등록·후보표와 checkpoint/probe 코드·결과를 읽고 저장 예측을 독립 재산술 확인했다.

## 확인한 실제 증거

- baseline probe의 저장 예측912행을 원 WT1 CSV의 DIAG10/fold0 et_47와 row_id로 대조했다. 최대 절대차 **2.220446049250313e-16**, 독립 fsum RMSE **0.1445857323784848**, 등록 train6288/query912의 ID 교집합0을 확인했다. 이는 **seed47 ET의 shrink된 구성원 출력 한 셀**에 대한 결과다. 신규 변수 효과, 전체 baseline 재현, PFN·SG2·원시 ET 출력의 동등성 증거가 아니다.
- v2 후보표190행, 도메인 후보의 묶음 단위·ET only 적용 명시, 상호작용 문법 대기 상태, 온도/전체온실/row_id 관련8개 대기 후보 추가를 확인했다. 전체 후보의 정확한 생성 열 목록·계수·동일성 확인은 아직 없다.
- checkpoint 코드에서 필드 일치, query ordered ID, 예측 checksum, 산술 재계산, receipt 마지막 기록을 확인했다. selftest의16거부 사례는 이 제한된 구현 범위와 맞는다. 실제 프로세스·중단·재개 전체 검증을 뜻하지 않는다.
- 별도 causal feature audit는 부모가 실행 중이라고 전달했다. 본 리뷰의 CIM 조회는 접근 거부되어 live PID/시작시간/명령을 독립 확인하지 못했다. audit 완료/성공을 추정하지 않는다. 기존 session76791을 부모가 직접 poll해야 하며 이 조회 실패만으로 재시작하면 안 된다.

## V201 · P1 · critic issue ID 연결이 잘못됨

원 critique의 C03은 달력/전체 경로 ablation, C05는 CPU 동등성, C06은 반복 정답의 독립성, C07은 문헌/실패차이, C08은 시간경계, C09는 재개다. critic_fixes_v2는 C03에 자료추가, C05에 ablation, C06에 CPU, C07에 독립성, C08에 checkpoint, C09에 문헌을 연결했다. 따라서 해결 추적이 틀리고 **원 C08 시간경계 요구가 누락된 상태로 닫힐 위험**이 있다.

조치: 원 ID/원 제목/실제 fix/증거/잔여조건을 가진 새 매핑 v3를 만든다. C01 묶음 명세, C02 상호작용+자료추가, C03 ablation, C04 fold ID, C05 CPU, C06 확인편향, C07 문헌, C08 시간경계, C09 checkpoint로 정렬한다. 문서 보완·합성 PASS·실행 PASS를 별도 상태로 둔다.

## V202 · P1 · 코드·원본 실제 해시 검증과 의존성 pin 부족

probe의 input_sha는 현재 읽은 원본 파일에서 계산한 것이 아니라 source_contract에 저장된 SHA 묶음의 digest다. 새 runner에서는 현재 원본 SHA를 재계산해 등록과 비교해야 한다. probe contract의 code_sha는 probe 파일만이며 실제 실행하는 core/DC4/DP1 함수 의존성은 fold 등록에 기록했지만 fit 직전 재검사하지 않는다. checkpoint 모듈·env.py·원 모델 하이퍼파라미터도 실행 의존성이다. 현재 결과는 정확히 맞았지만 범용 재개 안전성까지 입증한 것은 아니다.

조치: 실행 전/각 fit 직전 실제 원본과 의존코드 SHA·모델 params·특징 schema를 pin 검사한다. 함수 동적 추출의 namespace 상수도 명세로 저장한다. 변경 원본/의존함수 오염을 selftest에 넣어 fit 전 거부를 확인한다.

## V203 · P1 · CPU 통합 baseline와 누수전체 미완료

ET 한셀 출력 일치는 중요한 진전이나 LGB/MLP/PFN 문맥·혼합·shrink·clip·SG2를 검증하지 않았다. 잠재 PFN context는 NOT_EXECUTED로 올바르게 표시되어 있다. public 학습 참조 허용 범위, hidden 양 정답 제외, 내부 OOF 재fit, 달력 prefix·단일query 예측까지 최종처리 감사가 필요하다. probe가 저장 기준선을 재현한다는 사실은 그 저장 baseline 규정 적합성을 자동 증명하지 않는다.

조치: 지정 baseline의 단계별 출력/ID/허용오차와 전체 처리 인과 불변성을 확인한 뒤 gate를 연다. probe status PASS와 candidate-fit 허용을 서로 다른 gate로 관리한다. raw/최종 atol1e-6을 못 맞추면 새 CPU baseline으로 별도 등록하며 사용자 기준선과 같다고 표현하지 않는다.

## V204 · P1 · 후보별 열집합·실험 runner 로그 부재

family 묶음이라는 정의로 후보수 모호성은 줄었지만 D01의 operators 목록은 모든 실제 생성 열·창·경계식을 지정하지 않는다. 후보별 ET만 바꾸고 나머지 구성원 fixed 조건에서는 'feature가 무효'라는 결론도 ET 추가경로에 한정된다. M/U 후보 params는 아직 빈 객체다. baseline 실제 ID가 한셀에 생긴 것과 190후보 runner의 train/anchor/context/inner 계약이 구현된 것은 다르다.

조치: candidate_id→ordered generated columns/formula/spec SHA→modified member→fit IDs→baseline fixed members→통합 출력 경로를 사전 저장한다. NaN/0시/전환 첫관측/min_periods 규칙을 확인한다. 각 fit 직전 상태와 source SHA·PID시작·명령·입력/출력 위치를 로그로 기록한다. M/U는 구체명세 전에 활성화하지 않는다. ET only 기각을 전체 모델계열/모든 활용 경로 기각으로 확대하지 않는다.

## V205 · P2 · checkpoint의 동시 실행·종료 상태를 추가 검증해야 함

atomic os.replace는 단일 파일 교체를 안전하게 하지만 complete의 receipt 존재검사와 쓰기는 동시 두 worker의 경쟁을 막지 않는다. partial predictions만 있는 폴더는 기존 내용을 새 출력으로 교체할 수 있으며 PID 시작시각은 probe RUNNING 상태에 없다. SHA는64문자 길이만 검사해 비hex도 통과한다. 중첩 fold와 feature/metric/bootstrap 명세는 현재 REQUIRED에 별도 필드가 없으므로 상위 fold/code hash가 실제로 포함하는지 계약을 더 명확히 해야 한다.

조치: 후보 cell의 단일 writer claim, partial quarantine/recovery 기록, process start identity, 상태 전이/실패경로를 구현한다. 동시 중복 COMPLETE·중간 crash·stale PID 재사용·누락 receipt·저장예측 오염·명세 변경을 실제 runner에서 재현하여 거부한다. hash 형식을 엄격히 검사하고 inner/calendar/feature/metric/bootstrap 명세를 계약에 포함하거나 정확히 어떤 상위 hash가 커버하는지 저장한다.

## 다음 허용 작업

자료 진단·특징구현·인과 감사·기준선 다른 구성원 재현을 계속할 수 있다. V201 매핑 수정과 V202/V204 실행계약 보완은 지금 진행 가능하다. 아직 후보 효과의 정식 PASS, 신규모델 개선, 전체 baseline 재현, 전수검사 완료를 보고하면 안 된다. 후보 실행 gate는 통합 baseline·전체 누수 감사·실제 runner 재개검증의 증거로만 열어야 한다.
