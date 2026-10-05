# BOM 검산 보완 및 종료 helper 독립 비평

2026-10-06 · 집 코덱스 독립 비평 담당. verify_v5.py/finish_v5.py/verification_registration_v5.json/partial_audit_v2.log를 실제 읽었다. 원 worker·원 source 수정, 모델 학습·재시작·전체 완료 판정은 하지 않았다.

## 판정

**BOM 보완의 원인과 계산 불변성은 독립 확인했다. 종료 helper가 실행 중 worker를 재시작하거나 중단하는 코드는 없으며, 성공 검산 뒤 lock을 해제하는 흐름은 적절하다. 현재 경로를 유지할 수 있다.** helper의 실제 전체 완료는 아직 발생하지 않았으므로 완료 성공·최종 A 결과로 표현하지 않는다.

## 실제 검산과 교차확인

- 원 공개 CSV 시작3byte가 EF BB BF임을 직접 확인했다. utf-8-sig로 읽으면 row_id 헤더가 정상 복원되고 DIAG10 seed7 정답8640행이 구성된다.
- verify_v4→verify_v5 diff를 읽고 AST에서 runtime 검사 시작부터 numeric proof 구성까지 구간이 정확히 동일함을 확인했다. 계산조건·허용오차·특징·분할·flags 문턱·포괄성의 변경은 없다. 변경은 CSV decoder, 보완검산 source 서명 가드, 결과5 파일명, 설명 및 재현ZIP에 보완 소스를 추가한 부분이다.
- verification_registration_v5의 verify_v5.py/finish_v5.py SHA를 실제 파일에서 재계산해 일치했다. original_registration_sha도 실제 registration_v4 SHA와 일치했다.
- partial_audit_v2.log 실제 결과는 PASS_PARTIAL_AUDIT_ONLY, checked_components3, complete_contexts0/80, raw_mix_maxdiff2.220446049250313e-16, seed7첫재적합 기록2.220446049250313e-16이다. 이것은 R3 세 구성원 검산이며 PFN4 및 한 전체문맥조차 완료됐다는 판정이 아니다.

## helper 안전성 점검

1. 시작 시 보완 등록의 source SHA, launch_v4의 run_v4 source SHA, 현재 lock PID를 확인한다.
2. OpenProcess에는 SYNCHRONIZE(0x00100000)만 요청한다. WaitForSingleObject의30초 단위 timeout258은 다시 기다리고, worker종료0만 다음 단계로 넘어간다. 종료/재학습 호출은 없다. 처음 잡은 process handle로 기다려서 이후 PID 조회 반복의 PID 재사용 문제를 줄인다.
3. receipt4 COMPLETE 확인 후 별도 verify5 subprocess를 실행하고 exit0를 요구한다. proof5 PASS와 bundleZIP 실제SHA를 확인한 다음에만 동일PID lock을 삭제한다.
4. finalization5 상태는 `COMPLETE_TRAINING_AND_NUMERIC_VERIFICATION_CRITIC_REVIEW_PENDING`으로 독립 최종 비평 대기를 명시한다. exception에서는 failure5를 남기며 기존 학습/캐시를 덮어쓰거나 삭제하는 흐름이 없다.

## 남은 P2 및 개선 피드백

- **lock owner의 source 대조.** 삭제 직전 PID만 비교하며 lock.source와 launch.source_sha/run_v4 SHA를 함께 확인하지 않는다. 현재 run source는 별도 시작가드에 있으므로 명백한 충돌은 확인되지 않았지만 owner 조건을 PID+source로 강화하면 더 좋다. 살아있는 helper source를 수정하지 않고 추후 복구 새 버전이나 수동 해제 검산에서 이 확인을 보충한다.
- **원 등록 SHA 필드의 명시 대조.** verification_registration_v5에 original_registration_sha가 있으나 helper/verify5가 이 필드를 직접 사용하지 않는다. 이번 독립검토에서 실SHA 일치를 확인했고 verify5는 원 등록 항목별 소스 해시를 계속 검사한다. 그래도 향후 보완 검산의 provenance를 위해 원 등록 파일SHA 자체도 코드에서 비교할 수 있다.
- **helper의 중단 복구.** finish_v5의 save는 exclusive-create이므로 finalization5 또는 failure5가 이미 존재할 때 재실행하면 새 저장이 실패한다. lock 해제 뒤 finalization 쓰기 전 중단 시 최종기록이 없을 수도 있다. 계산 캐시를 변경하는 문제는 아니므로 새 helper 버전 또는 fresh standalone verify5/check-only와 새 finalization 기록으로 복구한다. 이미 만든 파일을 덮어쓰지 않는다.
- **재현 방법 구분.** 원 run_v4의 마지막 verify_v4는 BOM 오류를 그대로 갖고 있어 run_v4 단독 정상종료로 끝까지 재현된다고 말하면 안 된다. 기록에는 원 worker 종료 뒤 finish5/verify5로 숫자검산을 완료한 보완 절차를 명시해야 한다. 자동설명에 원 verifier 오류와 별도 helper를 명시한 것은 적절하다.

## 다음 결과 평가 게이트

PFN 첫 실제 캐시가 완료되면 문맥 ID·외부query 배제·원숫자·batch8/repeat를 별도 비평한다. 전체 완료 시 original run 종료로그/BOM실패 및 helper 수치검산 성공을 각각 보존하고, proof/receipt/bundle/lock owner를 재검사한다. 그 뒤 독립 최종 혹독비평과 문제별 개선책을 추가한다. 현재 보고서는 수정검산과 helper 설계의 평가에만 해당한다.
