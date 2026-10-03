# GPU 고정 배치 추론 감사 — 실행 전 2026-10-03 집 코덱스

이전 GPU 보조tool19414는 query 전체/첫8행의 예측이 1.5795e−5 달라 bit 동일성 검사에서 종료했다. 장치 난수 열임베딩을 CPU generator로 맞춰도 FP32 연산 차이가 남는다. 이번에는 모델이나 기준을 바꾸지 않고 query 계산 shape를 고정해 배치 불변성의 실패 원인을 해결할 수 있는지 검사한다.

단일 고정 방법: query를2048행 단위로 처리하고, 부족한 행은 고정된 학습 context의 첫 입력으로 채운다. padding행의 출력은 버린다. context2000/4estimator/seed1/season-last FULL38/CPU-seeded embeddings/TF32off/highestFP32 동일. DIAG10 fold1의 기존 canonical inner_train_id,query row_id를 그대로 쓴다. 원래 모델의 CPU checkpoint와 비교하되 원시허용1e−5·bitquery불변 기준을 완화하지 않는다.

두 번 새GPU fit 재현, 전체vs첫8·17행·query순서변경·뒤query특징변경 시 앞8행의 bit동일성, batch내가짜query가실제query출력에영향없는지 검사한다. CPUraw maxdiff<=1e−5도 별도 판정한다. 이 마지막 기준 미달이면 CPU와 같은 모델/대체실험이라 부르지 않는다. 이번은 예측성능시험/채택판정이 아닌 추론 계산 감사이며 통계가중치/문턱 선택 없음.

CPU60252 종료/대체/캐시수정0. 별도 local/ec_gpu_fixed_batch_audit_20261003_v1만 쓰고 설치패키지 변경0. 원시EC/잠금/EL1/test값·예측/제출물 사용0. 기존 공개OOF 라벨로 train만 fit. 이번 audit 통과 뒤에만 GPU 보조 전체 실행의 새계획을 검토한다.
