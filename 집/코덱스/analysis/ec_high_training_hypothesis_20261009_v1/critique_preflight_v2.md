# 독립 source preflight v2
2026-10-09 · run_v1.py → run_v2.py diff 검토

판정: **실행 승인, 새 차단 문제 없음.** critique_preflight_v1.md의 소스 점검과 주장 제한은 유지.

변경 확인: RAW_INPUT을 prepare에서 원 raw DataFrame으로 보관, jobs에는 structure/cal=None, 실제 처리 fold마다 같은 SG.prepare(RAW_INPUT,ref)/SG.ref_calendar(structure,ref)를 실행하도록 이동했다. 원 v1도 같은 raw/ref로 호출하므로 계산시점만 바뀌었다. 학습행·선정규칙·거리·seed·후보·가중치·정답·통계변경 없음. prepare에는 model.fit 없음. 원결과 files 이름은 유지하되 registration/replay/preflight는 v2를 참조하며 run_v2 자체와 PLAN_v3_reporting까지 핀한다. 기존 fold 파일이 존재하면 등록SHA 불일치가 중단시켜 버전혼합을 방지.

RAW_INPUT 전역은 동일 main 호출의 prepare에서 초기화되고 run 직전 사용. SG.prepare 이전에 원 raw를 수정하는 새 동작이 없음을 diff로 확인했다. 첫 baseline replay와 final baseline maxdiff assert를 통과해야 실제 수치 동일성 확인이 완성된다.

독립 verifier v2는 registration_v2.json/baseline_replay_v2.json을 참조하며 생산자 score_v1 산출물명은 보존한다. 최종 독립 검산 output은 independent_v2.json으로 새 이름. 실제 SG structure 전행의 독립재생은 수행하지 않았고 source 및 baseline재현으로 검증한다.
