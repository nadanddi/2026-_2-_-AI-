# N4 실행 코드 검토 v1

2026-10-10. `n4_run_v1.py`와 `N4_PLAN_v1.md` 실제파일을 읽었다. 이번 critic 새fit/predict/정답로딩/실험중인 audit 세션 접근 0. 원파일 수정0. 계획비평 후속이며 **현재 audit 완료를 확인했다는 문서는 아니다**.

## 계획 지적에 대한 구현 확인

- 각 새fold에서 `TM.codex_fit_predict(...726)`와 동일feature의9109을 재학습하고 `array_equal`을 강제한다. seed가 다른 IRR과의 차이에 포함될 수 있다는 계획 지적을 구현상 해소했다. 실제 모든fold 통과는 후속실행receipt에서 확인해야 한다.
- BASE는 원factory/fit_mean_w를 가져와 .65/.25/.10을 평균하고 PFN은 같은fold의49 context/model seed, 2000행 weighted sample, CPUV2/4estimators로 baseline/candidate에 공유한다. PFN49는한표본이라고 recipe/results에 명시한다.
- 혼합후보차 `(0.6−0.4g)*g*(irr−c)`, g=0 exact동일을 합성 및 실제출력에서 assert한다. NaN gate1도 합성 확인 대상이다.
- train loader는선택 train_ids 외 대상 train행을 sx로 이동하고 TRAIN세계에서 sx 입력을 NaN으로 만들어 query/purge/test 센서분리를 의도한다. train label도선택IDs만 남기고 query label열을삭제한다. weights/season은mask train ctr에서 계산한다. 이것은 source 확인이며 실제다중경계 실행검사를 대신하지 않는다.
- 초기 recipe는 source/runtime/원3CSV/checkpoint,각fold orderedID·feature/physics·새split/seed/alpha를 저장하고 cache는recipeSHA+파일SHA로제한한다. 원패키지전파일 SHA도 top-level 검사한다. 예비adoptionFalse/posthocTrue를 명시한것은 정확하다.

## 남은 보완점

### 1. `--first` 재실행의 의미가 달라짐

현재 cache가있으면검사후 `continue`하므로 `--first`를같은폴더에서다시호출할때FD[0]에서종료하지않고 **다음미완성fold를 새로학습한뒤종료**한다. `--first`라는이름과첫fold 독립검토약속만으로추가학습을막는안전장치가아니다.

첫fold가없는이번최초호출은FD[0] 완료후종료할수있다. 다만재개때는그상태를먼저직접확인하고재시작하지않아야한다. 새버전에서`--first`는정확FD[0]의cache검증후에도종료하도록정하거나,‘다음미완성1fold’별도mode를명시한다. 전체실행은독립중간검토후에만한다. 이미존재하는원파일을덮지않는다.

### 2. bootstrap은 농장별 층화가 아니라 전체 block pool

구현은farm+rawday//5 key전체를한array로만들어거기서복원추출한다. **농장×block을단위로삼은pooled bootstrap이며농장별표본수를고정하는층화bootstrap이아니다.** 코드상의통계량은차이RMSE이고CI분위수는alpha/1−alpha다. 계획/결과에이정확정의를적고이번계약을변경하지않는다. 층화로바꾸려면실행전새버전등록이며결과본뒤좋은검정만고르지않는다.

### 3. audit의 범위를 축소해서 보고해야 함

audit는첫DIAGfold의모델featureframe을확인한다. 각농장첫query날h8 cut에대한미래변조와전체타농장변조,holdoutlabel sentinel 및TRAIN query/purge/test센서변조,gate합성검사다. **모든fold·모든시간경계·완전모델predict의인과성/배치독립검사도, weights/season/PFN context의전체인과성검사도 아니다.** 반환key `future_otherfarm_mask_label_pass`를전체원모델인과PASS로표현하지않는다.

최초actualaudit의exit/receipt/recipeSHA를확인한뒤첫fit로갈수있다. 실제첫fold에서registeredmatrix/weights/season/contextIDs·subset배제·label삭제 및train-query ID·원시혼합검산을하고, 원래계획의모델배치/feature미래불변성은그검사범위에맞게별도로기록해야한다. PFN캐시원위치assert는모델다운로드금지/원checkpoint사용에대한증거이지모델query배치안전성증거가아니다.

### 4. 코드에 없는 보고 항목을 후속검산으로 고정

결과코드는3validator 전체RMSE/p/CI만쓴다. 계획비평의농장×후반/gate구간·변경행지원·집중도진단과BASE3seed별차이는현재원시NPZ만으로모두완결되는것은아니다(BASE개별seed가저장되지않는다). 예비평균모델결과를모든BASEseed일관성으로설명하지않는다. 추가성능변경fit없이필요한저장/검산범위를사전정리해야한다.

## 진행 판정

현재코드는계획의주요seed교란/gate/paired혼합문제를구현상닫았다. 위문구·재개정책과audit완료/등록pin대조가닫히면최초 `--first` 진입은가능하다. 이코드비평은사용량제한때독립검토없이전체fold계속해도된다는허가가아니다. 단일PFN표본예비실험/사후6번째후보/신규layout독립자료아님/원9회차정확재현미해결이라는한계는그대로유지한다.
