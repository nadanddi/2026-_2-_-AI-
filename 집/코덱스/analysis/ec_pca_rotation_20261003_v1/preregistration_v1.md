# EC PCA_ROTATION_ET 사전 고정 — 2026-10-03 집 코덱스

가설: 현재 season-last FULL38의 자정/누적/현재 특징은 중복과 상관이 크다. 원래 축을 따르는 랜덤 분할 대신 학습 입력의 상관을 반영한 직교 좌표에서 같은 트리를 학습하면 동일 정보의 활용 방식이 개선될 수 있다. 새 정보나 개선을 보장하지 않는다.

카탈로그/CAMPAIGN에서 PCA/주성분/회전/rotation/직교 검색: 이 EC 구현의 이전 시험은 발견되지 않았다. 기존 leaf4/daybag/l2fp는 이미 rl_ec_v1에서 시험했으므로 반복하지 않는다.

단일 변경: ET의 median imputation 뒤 StandardScaler와 full-SVD PCA를 삽입. PCA는 38개 축 전부 유지(n_components=None, whiten=False), 특징 선택/차원 축소 없음. 각 outer train에만 imputer/scaler/PCA fit. 검증 행은 frozen transform. ET600/minleaf1/maxfeatures1.0/seed7,101,2024, 입력순서 FULL-day+season, 원시EC 목적함수 유지. 날 단위 분할·±1 제외·잠금제외360일·기존22fold 고정. 현재 season_v2에서 ET 구성원만 .48 교체, 기존 누적평활/학습범위clip 유지. 날별 gate/문턱/시드/가중치 탐색 없음.

판정: 3seed×5validator=15칸 모두 RMSE 감소, DIAG10 farm별 5기록일 block 20000bootstrap p_worse<.025/13 및 보정CI상한<0. 공개 통과해도 독립 후속 검증 전 채택 없음. 원시 train_y EC, 잠금목록/잠금정답, EL1 재채점, test 값/예측/제출물 읽기·생성 없음. 라벨은 기존 공개 OOF만.

검사: PCA의 직교성/역변환으로 표준화 입력 보존, 첫foldseed7 새학습 재현, 전체 vs 첫8행 배치 불변, 현재보다 미래/다른 온실 query 입력을 바꾸어도 이전 예측 불변. 66학습 이후 일반 verifier의 독립 scalar clip/누적평활/RMSE/fsum/pandas bootstrap/분할 감사. 기각도 그대로 저장.

출력: 집/코덱스/local/ec_pca_rotation_20261003_v1. CPU 내부보정60252를 종료/대체하지 않음, ET n_jobs2로 함께 실행.
