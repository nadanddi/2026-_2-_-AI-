# 잔여폴드 드라이버 독립 비평 v1
2026-10-09 · parallel_driver_v1.py 소스 검토

판정: **실행 승인, 새 차단 문제 없음.** 중간비평 승인 후 동일 run_v4.P.run을 early[1..5]/late[6..9]로 실행한다. 원 fold0과 겹치지 않고 양파트도 비중복. 각 fold CSV/pending/meta 경로가 독립이므로 출력 충돌 없음. process-private RAW_INPUT/SG구조를 사용하고 모든 seed·후보·통계는 동일하다.

driver 자기SHA/원producerSHA/registrationSHA/part목록/fit_change=False/score=False를 비교하고 prepare의 정규화등록 일치를 검사한 뒤 파트별 x모드 lock을 생성. 동일파트 중복 fit은 lock에서 중단된다. 이미 저장된 fold는 원run의 registration/hash assert 뒤 skip. worker 자체는 score를 하지 않으므로 부분완료로 final score가 경쟁 생성되지 않는다.

운영상 한계: lock은 prepare 이후이므로 같은파트 중복프로세스가 prepare까지는 CPU/RAM을 쓰지만 fit에는 들어가지 못한다. 작업중 예외 시 lock이 남아 자동재시도를 막는다. 임의삭제 금지; 해당 PID/명령줄 종료 확인 후 사용자/주작업의 복구판단 필요. 미완 CSV+meta 비대칭도 원run에서 중단한다.

원RAM여유2GB 상황에서 두worker를 즉시띄우지 않는 주작업 결정은 타당. early 먼저 실행, 여유메모리를 확인하여 late순차 또는 동시 실행하며 숫자기준은 바꾸지 않는다. 최대후보90fit 그대로, final score는 모든fold성공 후 단일실행/독립검산 필수.
