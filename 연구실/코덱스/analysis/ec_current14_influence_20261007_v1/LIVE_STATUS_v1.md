# 현재 EC14 공개 검증과 ET 학습날 제외 진단 진행 상태

2026-10-07 연구실 코덱스. 사용자 “진행해봐”로 1·2단계 승인. main b8d40fb에 baseline 계획·source·준비서명과 사전 검산을 등록하고 실행했다. f570b01에 삭제안 코드·채점 코드·raw ET 편향 판정 정의를 추가 등록했다.

baseline 실행: runner_v4.py, 프로세스1072, 시작 2026-10-07 01:57:05. 도구 세션8096. 로그는 내 local/ec_current14_influence_20261007_v1/baseline_v4.log, 모델 캐시는 같은 폴더 base/에 있다. baseline.lock이 있으면 중복 실행하지 않는다. 완료된 npz+json 쌍만 재사용할 수 있고 단독 부분 파일은 보존 후 따로 점검한다.

모든 10fold×3seed R3와 실제 raw 평균 ensemble의 360일 검증이 끝나기 전 부분 성능을 채점하지 않는다. PFN은 기존 동일 특징·ID·정답 hash와 문맥 RNG/row ID를 검사한 4캐시 평균이며 새 PFN 학습은 0회다. 원 train_y/test_X/EL1/잠금 정답을 새로 읽지 않는다.

독립 사전 비평은 baseline와 삭제안 모두 모델 누수 결함을 발견하지 않았다. SG2 전체 prepare/calendar/correct 원본 동등성, 활성 prefix 인과검사, 실제 입력 인과검사가 완료됐다. 실제 데이터 F47의 dummy SG2 활성은 0행이므로 실제 완료 예측에서 경계와 활성 수를 별도 감사한다.

순서: baseline_receipt_v4 생성 → score_v1 baseline → 독립 전수 검산·1단계 검토 → ablation_v1 --prepare·등록 → 삭제안 실행 → score_v1 --ablation → 전수 수치·전체 트리 감사 → 최종 혹독비평·보완 → 보고서·공용 기록. 현재 삭제안 fit0, 채택/제출0. 두 판정의 ET 일편향은 raw24h평균으로 고정한다.

ablation_v1는 baseline trace용 ET1회와 D1/D2의 신규 ET최대60회를 구분한다. 다른 구성원·SG2 학습 정답 ref·bounds·계절은 BASE에서 고정한다. 삭제안은 선택 사례 의존도 진단이며 실제 학습자료 제거 해결책이나 새 제출모델로 채택하지 않는다.
