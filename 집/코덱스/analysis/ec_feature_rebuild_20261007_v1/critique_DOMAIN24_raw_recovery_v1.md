# 원 raw 중단·재개 독립 범위 점검

2026-10-07. 새 snapshot2/source diff, 보존 lock, checkpoint22/PROGRESS/HANDOFF/catalog6.387 및 등록·62파일 SHA를 읽었다. 모델 fit/예측/정답/채점/GPU 실행·프로세스 변경은 하지 않았다.

**기록된 복구 절차에서 새 핵심 blocker는 찾지 못했다.** 모델 완료로 오인하지 않고 동일 source/등록으로 재개하며 lock을 새 이름으로 보존한 방향은 타당하다. 다만 역사적 프로세스 부재와 session56114 live는 부모의 실제 관측/보고를 기록한 근거이며, 이 리뷰가 과거 프로세스 조회·worker poll를 다시 수행한 것은 아니다.

실제 v1→v2 diff는 허용 인수1→2와 출력명v1→v2뿐이다(마지막 빈 줄 차이 포함). 검증 규칙이나 학습 source를 변경하지 않는다. snapshot2 recorded codeSHA/등록SHA와 rawrunner3 currentSHA가 일치한다. 보존 lock PID18004/start_ticks134358392201732601의 등록SHA도 같은 실제 등록 파일을 가리킨다.

snapshot 파일목록은62개·전부PASS이고62개 현SHA 불일치0이다. 모두DIAG10_fold0의 부분 결과이므로 62fold 또는 넓은 원검증기 완료로 해석할 수 없다. 감사는 exact row IDs/finite pred/digest/runtime/model constructor/준비영수증/후보 저장 matrix SHA·열·allmissing/imputer 저장 합치·5 numeric error 요약/3 sampled-prefix를 검사한다.

**이 메타데이터 PASS는 fresh matrix·train median/y 숫자 재구성, 모델 재fit, source 함수 독립 반복, 전체5346 raw fit 또는 full mixed causal gate가 아니다.** baseline CID↔member/기초matrix·actual y SHA도 감사 자체만으로 완전히 재구성하지 않는다. 다음 registered runner3의 production loader fresh24matrix와 `fit_one` exact expected contract/y/imputer 검사를 거쳐야 기존62 재사용이 검증된다. 재사용 분기를 성공한 receipt/폴드완료·후속진행 로그가 확인되기 전에는 '기존62 재사용 완료'라고 기록하면 안 된다.

PROGRESS/HANDOFF는 이전handle 없음→정식 process/lock PID 없음→종료원인 미확인→metadata62PASS→lock 이동보존→동일 runner 재시작/live→재사용은 후속대기의 순서를 분명히 남긴다. checkpoint22의 termination_cause=unknown/fullgoalfalse/scorefalse와 catalog6.387도 그 범위에 맞는다. 종료 원인을 OOM·완료·환경오류 중 하나로 추정하지 않는다.

stale lock 해제 근거는 PID 하나의 종료뿐 아니라 동일 runner의 다른 writer 부재까지 확인해야 하는데 부모 보고는 두 조건을 포함한다. 앞으로 잠금 재사용/중단 때 PID 재사용을 피하도록 start_ticks와 registration을 함께 확인하고, 새 worker가 있는 동안 두 번째 writer를 시작하지 않는다. 이번 metadata 감사는 별도 새 버전이므로 원214핀의 학습 source 수정이 아니다. 이 리뷰는 실행 중 새 writer의 전체214 currentpin을 다시 전수 검사한 것으로 확대하지 않는다.

PFNfit0/메모리 guard/원raw 재개는 성능 선별이 아니며 시드·폴드·후보·통계 변경 근거가 아니다. 전체 모델 gate 이전 정답 parse 금지, 원24 전부·고정 통계·최초 미사용1회 규칙은 계속 남는다.
