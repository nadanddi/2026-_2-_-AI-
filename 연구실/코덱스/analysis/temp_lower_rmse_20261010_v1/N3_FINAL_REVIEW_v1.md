# N3 최종 독립 비평

2026-10-10 연구실 코덱스. `n3_results_v1.json·n3_decomposition_v2.py·n3_decomposition_v1.json·n3_model_v2.py·n3_registration_v1.json·verify_n3_v2.py·n3_independent_verify_v1.json` 및 계획/중간 비평을 읽었다. 이번 critic 새fit/predict·정답CSV/NPZ 재로딩·외부자료 접근 0. 기존등록 및 저장JSON의 산술/현재source SHA는 별도 PowerShell로 확인했다. 원정답 재채점·20k bootstrap의 이번 추가 재실행은 하지 않았다.

**판정: N3 구성원 screen FAIL·전체혼합 미진입/채택0은 타당하다.** DIAG10 .742598→.900470(+21.259%, p_worse1, 등록99% 차이RMSE CI[+.097267,+.228914]℃), EXT10 .889502→.988754(+11.158%), EXT12 .914291→1.577262(+72.512%)로 세검증 모두 악화했다. EXT10의 F13후반96행 국소개선으로 전체방향실패를 덮지 않는다.

## 근거 확인과 실제 검증 범위

- 현재 `n3_run_v3.py` 및 `n3_model_v2.py` SHA는 등록 source/model SHA와 각각 일치했다. runner source만 검사하는 기존 verify 코드의 범위를 넘어 이번 model pin도 직접 확인했다. 모든 dependency/NPZ source에 대한 새로운전수재생검사를 한 것은 아니다.
- 등록alpha .005/20k/seed20261010과 shape with_meanFalse·interceptFalse가 실제model과 정합한다. y0는train0시정답으로만 학습하며 `assert 'sub_temp' not in va.columns`가fit보다 먼저 나온다. level은원h0행 w0,shape는원시간행 w를 쓴다. shape0는예측값을차감하며 실제query y0를 사용하지 않는다.
- 기존 독립검산receipt PASS86건/max2.49e−14는 원queryID/parent예측/weights/season·전체3검증RMSE·bootstrap 및 farm×후반분해를 확인했다. 이번 critic은 그 코드를 읽고 산출JSON과 대조했으며 이를새로86건 재실행했다고 주장하지 않는다. 기존중간비평은첫fold960행의원시항등식직접검산이다. 중간범위를전체newcausality검사로확대하지않는다.
- `new_fit22/total_component_fits25`는저장결과의회계다. candidate12fold의level+shape head 24개와baselineprobe1의집계이며각head가Ridge/LGB를포함한다. 전체BASE/PFN fit수가아니며 N3를새seed여러개검증한증거도아니다.

## 수준·모양 분해의 정확한 해석

decomposition은 **예측 종료 후 진단**에서 공개train_y의row_id/sub_temp만읽는다. `truth0`를query라벨에서 가져오는것은이사후진단에한정되며 추론모델로다시전달하지않는다. row_id 고정문자열로h0ID를만드는코드이므로현재ID서식의정합성을전제로한다. 다른서식까지일반화한loader가아니다.

|검증기|E[e0²]|E[eshape²]|2E[e0·eshape]|최종MSE|기준MSE|
|---|---:|---:|---:|---:|---:|
|DIAG10|1.028038|.614448|−.831639|.810846|.551452|
|EXT10|1.230706|.857241|−1.110313|.977634|.791214|
|EXT12|3.349335|.837763|−1.699341|2.487757|.835928|

이번별도JSON산술에서 세행의 `level_mse+shape_mse+cross_term=total_mse` 오차최대1.11e−16, sqrt(total/baseMSE)와결과RMSE차최대1.11e−16이었다. **cross_term은비중심교차항이며평균을뺀공분산과같지않다.** 이전N3계획비평의‘공분산’표현은실제진단에서교차항과평균편향을구별해읽어야한다.

EXT12 수준오차MSE3.349가크고shape와음의교차항이부분상쇄했으나전체는크게악화했다. H1/H2의‘shape국소개선이나level악화’선행과연결되는반례다. 다만기준모델을같은y0분해로비교하지않았으므로 **최종악화의몇%가level원인인지** 기여율로단정할수없다. 교차항때문에각항을독립기여율로더할수도없다.

shape0차감후MSE−rawshapeMSE는DIAG +.008698,EXT10 −.046250,EXT12 +.000987이다. anchor강제의효과가검증기마다다르다. 이사후분해에서‘차감을빼면좋겠다’고원명세를수정하거나계수를최적화하지않는다. rawshape만다르게최종합산한새후보는별도실험이며이번N3통과가아니다.

## 결론의 제한

N3는실제0시온도를두머리로학습한고정구현실패다. 모든배지state모델/전문가설명/0시anchor방식의불가능성은증명하지않는다. 그러나현재근거에선수준·모양분리경로를같은파라미터변형으로반복할정당화도없다. 특히모양만좋은칸·F13후반EXT10만골라gate를붙이지않는다. 원9회차정확재현문제는여전히별개이며N3결과로해결하지않았다.
