# family19 学習前 재현 경계 보완 v2 · 2026-10-04

v1 문서·코드·준비 결과를 보존한다. 계산식·λ=.01·w범위[0,1]·검증15칸/DIAG p<.025/19·데이터/학습 범위는 그대로다. 실행 소스는 **run_v2.py**이다. 실제 fit 전 main에 커밋한다.

비평이 발견한 두 운영 문제를 보완한다. 첫째, 캐시 signature에 ordered IDs/목표뿐 아니라 내부·외부 학습/질의 FULL38+season 배열 hash, 현재 train_X와 공개 기준선 cache hash, core/season/env/support/guard/LOG 소스 hash를 포함한다. 현재 train_X는 기존66 R3 provenance의 입력hash와 같아야 한다. target 원시파일을 새로 열지는 않는다.

둘째, 첫cell의 CSV/NPZ/metadata JSON/첫 감사 JSON 중 일부만 있으면 **fit 호출 전에** 중단한다. 첫 감사 결과는 cell 파일3개와 독립 수식 검사가 성공한 뒤 저장한다. 기존 첫 감사 파일과cell이 모두있어야 재개 skip이 가능하고 signature도 맞아야 한다. 부분파일을 삭제/덮어쓰거나 실패한 셀을 몰래 다시 학습하지 않는다.

--prepare v2는 모든22CPU/88PFN가완료되었을때도새적합0이다. 원 matched calibration은 등록당시family9였으나 전체 결과는 현재 더 엄격한family19로 함께 보고한다. 새학습이익이 아닌 recipe 정합성 재시험의 검증 결과로 구분한다.
