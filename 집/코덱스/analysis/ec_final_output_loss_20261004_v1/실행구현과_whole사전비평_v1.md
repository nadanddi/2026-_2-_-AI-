# 잔차 MLP 실행 구현과 whole 사전 비평

2026-10-04 · 집 코덱스 · 실제 EC 모델 학습·예측·점수0

앞선 `설계와_혹독한비평_v1.md`는 LGB custom-loss 검토이며 현재 실행 후보가 아니다. 사용자1번 실행은 **기존 actual계절v2 raw에 BASE14 zero-head 잔차 MLP δ를 더하는 새학습기**다. FINAL_LOSS/RAW_LOSS의 손실만 달라지는 대조다. family22/23이며 별도 Tweedie 정확leaf family24까지 포함해 alpha=.025/24를 공통 고정했다. 등록은root담당이고 이subagent는커밋/실제fit을실행하지 않았다.

## 구현과 준비 결과

- 최종 runner는 `run_v3.py`, SHA `be6bab3303e37decad5bd078e56edf569cda5bf0baaea470df971bc53043b172`다. 준비 `preparation_v3.json` SHA `df4ff2c3b7617676c61e4cee4b8e9690f9312a7cca448c0caec189cb5051b860`, PASS_PREPARATION_MODEL_FIT0_PREDICT0_SCORE0다. 공개input/원queryR3/PFN4/22innerCPU/88innerPFN/분할orderedID·±1/원source·runtime/공개label·기준선재구성을 확인했다. 전처리b22회fit은 모델학습과 구분해표기했다.
- 기존 matchedinner 단일split에서a는원구성원학습, b는그heldoutquery다. 잔차MLP는b의BASE14·공개정답만학습한다. 전체outertrain모든행의완전crossfit이아니며학습되는b는부분집합이다. innerseason(a,b), outerseason(tr,q)를별도로재구성한다. 기존PFN·ET·MLP를재학습하지않았다.
- torch2.14.0+cpu/float64/2threads/interop1/deterministic을핀했다. torch Adam의betas(.9,.999),eps1e−8,amsgrad/foreach/fused/capturable/maximize/differentiable=false,lr.001/coupledweight_decay.01을명시했다. 400step의마지막state만사용하고전체401loss trace를저장한다. weight_decay는AdamW가아니며MSE만로그에표시하므로penalty포함손실로오해하지않는다.
- imputer median/keep_empty_features=True→StandardScaler를b에만fit하고stats·n_samples·weights·format/features/loss trace를NPZ에저장한다. 부모whole는별도NumPyforward로재생한다. 입력BASE14는raw MASK 제외5열을포함하지않음을corepin/name으로확인했다. 이는전feature생성source를독립다시증명했다는뜻은아니다.
- 같은seed의두modeinitial_weights_SHA가같도록각model초기화에서torch.manual_seed하고출력head weight/bias를0으로한다. 초기가중치SHA를manifest/meta.train/first에묶었다. 실제training호출전queryδ=0이cachedactualbaseline과1e−12일치하고innerδ=0이scalar후처리와1e−12일치하는지확인한다. training전체gradient/weightsfinite를매step검사한다.
- 132cells의CSV정확15열·mode별66cell/83160행 및combined166320행을출력한다. checkpoint/CSV/meta/firstpartial조합은STOP_PRESERVE며출력을덮어쓰지않는다. source/runtime/input/cache/prep를매fit직전검사한다. first각modeDIAG0seed7에fresh400step/order/single/otherquery/prefix/scalar/trace와zerohead감사를저장한다. originalconfigloss외부점수/정답은epoch선택에쓰지않는다.

`schema_description_v1.json`에manifest/signature예시 및CSV/checkpoint/meta/first/train key 구조를저장했다. 최종source수정이필요하면새버전코드·새prepare로다시고정해야하며v3를고치지않는다.

## 합성 검산

`synthetic_run_v3.json` PASS_SYNTHETIC_ONLY다. old groupedloop와vectorized gather→reshape(-1,24)→cumsumdim1→index_copy의loss/gradient최대차0, scalar/cumsum정합, autograd vslossfinite-difference최대3.4534e−13이다. FINAL/RAW400stepfresh재학습예측·trace최대차0이며zerohead/order/single/allNaNcolumnmedian0/전처리정합도통과했다. 합성초기/최종loss는FINAL .002007815→.001898423,RAW .000719763→.000717002이다. **각각다른loss이므로이두숫자로성능우열을판정할수없고실제EC결과도아니다.** 합성학습1600step과실제EC학습0을구분했다.

## root whole_v1 읽기에서 발견한 수정 요구

1. **점수gate위반:** percell감사중loss_records에outer_rmse=M.rmse를계산했다. 전체132meta/checkpoint/aggregate감사PASS뒤로외부점수계산을이동해야한다. innertrainingloss 재구성은감사에필요하지만outerRMSE는전체gate를지킨다.
2. **원R3guardcaller:** 함수내guard가검사하지않는provenance.feature_order/sharedinput/core/environment항목을caller에서직접검사해야한다. oldr3/bounds와raw의직접대조도명시한다.
3. **새v3schema:** first/meta의초기가중치SHA·finiteflags,inner/outerbounds,lossinitial/final과checkpointtrace,transformfeatureSHA,config전체고정값을검사한다. 일부config만읽어PASS하지않는다. 각모드정확15개key/132fileinventory·modeaggregate/combined순서를확인한다.
4. **전처리moments:** largevariance의roundoff에는statsrelative1e−12문턱을분리할수있지만actualprediction/후처리재현문턱은absolute1e−12를유지한다. allNaNcolumn은runnerkeep_empty_features=True의0과같이재생한다. 실제입력에없더라도합성지원과표현을일치시킨다.
5. **경계판정:** fsum/NumPyRMSE의엄격방향이같은지확인하고bootstrapNumPy/manualadjustedCI수치뿐아니라상한<0의bool도같은지검사한다. 같은RNG를의도적으로재사용하는산술검증과독립확인표본은구분한다.
6. **보존·출처:** verificationreceipt가currentrunner/prep/prereg/currentwhole/currentmath를모두핀하도록schema검사한다. 결과파일이이미존재하면점수계산전중단하고scores/segments/training/comparison/resultSHA를함께보존한다. receipt상태문자열은최초fit전Git등록시점의대체증명이아니므로root별도등록증거를남긴다.

이요구는root에즉시전달했으며수정whole_v2정적감사는별도다. 현재보고는runner/prep/합성완료이지실제132cell전체완료또는후보성능/채택보고가아니다. 저장first감사를검사해도독립400step재학습을whole에서수행한것으로과장하지않는다.
