# family19 학습 전 실행 환경 고정 · 문서 v4 · 2026-10-04

실제 모델 적합 전 마지막 운영 보완이며 실행 소스는 **run_v3.py**이다. v1~v3 기록을 보존한다. v1 학습식, λ=.01, w∈[0,1], 동일내부자료, 모든15칸 엄격개선 및DIAG p<.025/19는 변경하지 않는다.

비평의 나머지 경계인 런타임 버전도 고정한다. 공유 env 부트스트랩을 거친 실제 환경은 Python3.12.14/NumPy2.5.3/pandas3.0.1/sklearn1.9.1/LightGBM4.7.0이다. 이5개가 현재값과 기존66 R3 metadata의 environment에 모두일치해야 한다. source hash만으로 수치 라이브러리 동일성을 대신하지 않는다. runtime을준비기록/각cell signature에추가하여 재개에서도일치해야 한다.

첫cell 4artifact부분존재 중단 및 입력·season·core/season/env/원자료hash는v2와같다. v3는원시EC정답·잠금·EL1·test를열지 않는다. --prepare v3는적합0이며 이소스와문서를main에커밋한뒤실행한다. 완료fit_audit가있는 전체실험을 다시실행하지 않고 verifier만실행한다.
