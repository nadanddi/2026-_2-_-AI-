# family20 집계 파일 보존과 CI 표기 정정 · 2026-10-04

실제 실행 소스는 run_v4.py, 준비 manifest는 preparation_v4.json이다. v3에서 모델·특징·시드·검증·후처리·채택 문턱은 전부 유지한다. 첫 학습 전 및 최종 저장 직전에 기존 oof.csv 존재 시 중단하고, 집계 파일 생성은 exclusive 모드로 한다. 이미 생성된 집계만 있고 fit_audit 저장 전 중단된 경우에는 재학습·덮어쓰기 없이 별도 검산으로 처리한다. 기존 v1~v4 문서·source·prepare는 보존한다.

비평v16의 첫 감사 재개 검사는 farm/hour와 SHA에 묶이며 day 값을 별도 재계산하지 않는 한계가 남는다. 실제 최초 감사의 생성은 고정 tr/query day를 직접 사용한다. 원자료 특징 인과 감사는 37개 core 특징이고 season 특징은 기존 mapping 구현/입력/manifest 해시에 묶인다. 이를 모든 fold의 season 인과성 재실행 증거로 확대하지 않는다.

모델 계산: 원 R3 0.8 + 고정 CPU TabDPT 평균 0.2 → 같은 온실·같은 날 현재까지 누적평균과 0.5 혼합 → 학습 fold 범위 clip. 전체 3시드×5검증기 15셀 모두 개선, DIAG10 p_worse < 0.025/20 = 0.00125 및 family20 조정 CI [alpha, 1-alpha]=[0.00125, 0.99875] 상한 < 0을 채택 기준으로 유지한다. 최초 감사 전 성능 점수를 계산하지 않는다. 새 소스/manifest/이 문서는 최초 fit 전에 main에 등록한다. 이 시점 fit/predict 0.

사전v5의 95% CI 표기는 작성 오류다. 아직 fit/predict/점수 계산 전이며, 사전v1의 Bonferroni 조정 CI를 그대로 명시했다. 식·코드·시드·실행 manifest 변경은 없다.
