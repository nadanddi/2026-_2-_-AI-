# 고EC 가설 실험 독립 소스 preflight v1
2026-10-09 · 대상 run_v1.py 및 원 recipe model.py, preparation_v4.json

판정: **후보 fit 실행 승인, 차단 문제 없음.** 등록 파일과 첫 baseline 재현 assert가 실제 통과한 뒤 실행한다. fit 결과나 성능 승인 아님.

## 소스 근거
- select(t)는 outer 학습행만 입력받는다. 하루정답/47특징 평균과 같은 농장·기록거리>1의 5NN으로 삭제를 정한다. 실제 FULL_R3 목록에 farm/day/sub_ec 없음. 자기 자신과 인접일 불허. CONTROL은 farm×pass matching, 무복원, 고정 RNG, 정상일만 포함.
- select 결과를 모든 seed에서 공통 재사용. ET imputer는 삭제 후 재적합; LGB/MLP/PFN/SG2/bounds/season은 원 학습 t로 고정. 조건부 ET 개입이라는 계획에 부합.
- 캐시 배열의 ordered train/query ID, matrix hash, PFN context RNG/행ID와 원 prepare SHA를 대조. raw train_X SHA 및 query 8640 비중복 강제. baseline ET seed7 및 최종 baseline cached predictions 재현 assert 포함.
- ET 원정의 n_estimators600/n_jobs2 확인. fit threadpool2, predict1. baseline 신규1 + 후보 최대90 fit.
- paired bootstrap은 정상일 고유 ensemble행으로 계산. arm별 같은 farm×block draw, seed평균 예측 후 채점. 정답층화는 채점 단계에만 있음. alpha=.025/3 및 세 사전 비교 저장.
- run 출력은 집/코덱스 신규 local 경로. 다른 폴더 prepare/write 호출 없음. 등록 파일의 핀과 캐시핀으로 source drift 통제.

## 실행을 막지 않는 보고 보완
1. groups에는 raw_et RMSE가 없고 et_prediction은 shrink+clip 결과다. 독립 결과검산으로 raw ET RMSE 추가.
2. 각 농장 고EC, farm×pass 일반/고EC, pass1, 일반일 fold RMSE분산 표를 독립 검산에서 추가. 전체 farm 일반표만으로 세부 손해를 감추지 말 것.
3. bootstrap block은 day//5로 정한 달력구간이다. 계획의 ‘5기록블록’과 정확히 같지 않으므로 원구현을 명시. 관측 누락 및 일반일 필터로 block 크기가 다르다.
4. ci95_delta는 서술용95% CI다. 엄격 판정은 alpha=.025/3의 p_worse. 다중비교 보정 CI라고 부르지 말 것.
5. 거리 median finite assert는 있으나 원행 ±inf 직접 검사는 없다. 기존 hashes와 train_X 고정/원피처 코드로 보호되나 full finite z 검증은 독립 선정 재계산 없이는 미확인.
6. fulltrain matrix snapshot을 저장하지 않아 독립 verifier가 5NN 모든 거리선정 자체를 처음부터 재생할 수 없다. provenance/assert 소스검토와 selection_detail 구조 검토로 한정.
7. baseline_replay_v1.json 존재 시 신규 replay를 건너뛴다. 중간 검토에서 registration 일치/maxdiff PASS도 확인할 것.

최종 주장범위는 critique_plan_v1.md의 ET 개입·target 평균교란·단일 CONTROL·평가 고EC 부재 검증불가 제한을 그대로 유지한다.
