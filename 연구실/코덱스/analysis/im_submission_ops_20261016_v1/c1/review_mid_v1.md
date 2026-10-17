# C1 첫 폴드 중간 비평

결론: rest 진행 가능. 첫 폴드 체크포인트에서 중단할 수치/분할 구현 오류를 발견하지 못했다. 이는 후보 성능 통과 판정이 아니다.

독립 확인한 자료: first_fold_v1.json, first_train_v1.log, generated_audit_v2.json, 검사수정_방향_v1.md 및 ownlocal의 first_fold_v1.npz. 이전 검토한 train_eval_v3.py의 학습 조건과 함께 확인했다.

- 체크포인트7200행의 fold 크기=[1436,1424,1456,1448,1436]. fold0 1436개 OOF는 모두 유한하고 나머지5764개는 모두 NaN으로, 미계산 부분을 잘못 채운 흔적이 없다.
- fold0 점수 범위는 약2.8598464e-11~1.0. 확률1은 sigmoid 포화로 가능하며 지금 단계의 오류는 아니다. 향후 θ/θrc가1이면 사전 수치 조건에 따라 중단해야 하며 임의 clipping으로 살리면 안 된다.
- mean/scale/coef806차원, intercept1차원으로 low38+L14emb768과 일치한다. 모든 모델 배열이 유한하고 scale은 모두 양수다.
- first JSON의 train5059/held1436/held_real1077 및 finite=true를 확인했다. source_group_safe는 코드의 source별fold 일치 assert가 통과했다는 실행 증거이며, 체크포인트에는 source/group가 없으므로 이 NPZ만으로 provenance 전체를 새로 인증한 것은 아니다.
- 코드상 train은 fold!=0인 real/filter 행만이며 H.lrfit(C=.5)의 표준화+balanced LR를 사용한다. rest는 동일 fold 배열을 재검사하고 나머지 OOF만 채운 뒤 전체 유한성을 검사한다.
- 생성물 검사기록은1800장 전부 원본학습source/동일group/SHA/JPEG게이트 통과다. portrait-only 검사 오류를 기존 dist_cover의 가로/세로 양방향으로 바로잡은 것은 검사 가정 정정이며, 기록상 생성물/특징/학습/평가 기준 변경이 아니다.

남은 단계: 나머지4fold와 최종fit, 기존 H2 θ 산정, 고정 baseline 전행 대조 및 판정 배열 독립 재집계. 결과를 보고 θ풀/모델/문턱/기준을 변경하지 않는다.
