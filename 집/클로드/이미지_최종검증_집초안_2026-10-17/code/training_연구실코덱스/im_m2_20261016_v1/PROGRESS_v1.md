# 진행 체크포인트 v1 — 2026-10-16 02시대
- 전용 서버: /home/adilab/storage/nadan/codex/im_m2_20261016_v1
- preflight_v1.log 및 preflight/state_v1.json(완료시), r5_production_v1.log 및 r5_integrity_v1.json(완료시)을 상태확인.
- R5성능/특징값을 출력하지 말고 production로그 완료단계/오류와 files·ok만 확인.
- 이미실행중인 preflight/production 재시작 금지. 단계가 끝났는지 먼저확인. 기존raw/im_set_r5/feats덮어쓰기 금지.
- 다음: preflight완료 stage폴더 회수→verify_m2_v1.py preflight→독립중간비평. 모두통과+R5준비PASS+11시전일때만 run_m2_remote_v2.sh open을 **1회**. 그전 R5_OPEN_ONCE.json없는지 검사. 실패면개봉금지.
- R5기계제작요청은 탐색실패여도 완료. 결과성능미열람 봉인보존.
