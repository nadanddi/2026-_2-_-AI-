# 일반일/고EC일 분류 모델 — 사전 계획 v1

사용자확정2026-10-09: 고EC=공식일평균sub_ec>=1.2, 일반<1.2. 질문의결과물은분류기와검증,EC수치보정/제출은별도지시없어0. 데이터삭제안사용/원자료고EC모두포함.
예측시점: 같은농장의당일0..현재h 입력으로 당일24시간평균EC의고EC여부를판별. row_id는그룹/시각파싱만,day를학습특징에넣지않음. 정답/다른온실현재입력/대상시간이후입력/앞날정답은평가특징에없음. 미래상태목표를예측하는모델이며h8 EC순간값의분류는아님.
자료: 공개CT1 DIAG10 원정답8640행360일과train_X의그키만. 잠금40정답/공식test값/이전대회자료미사용. CV점수기록뒤최종학습에만공식400일9600행target를읽음,40일재검증0. 기준1.0으로사후변경/선별삭제0.
입력고정: 적격RAW14(외부temp/hum/rad/wspd,실내temp/hum/co2,환기/팬/난방/보온/차광/CO2/포그);각현재값·h0·현재까지mean·관측된값0비율. 시각sin/cos와h,농장flag. 기존DP1 9운영prefix + 현재까지낮9..min(h,15)mean-밤0..8mean(팬/차광/보온3개, h<9 NaN) + 대비available1. 총70특징, 실제생성순서manifest기록. 현재prefix누적사용,학습MASK는기존3.7/6.114사용불가5열제외. 외기전체하루집계/season/future이웃정답은없음. 신규특징추가/제거/튜닝0.
모델: 상수prior(해당foldtrain일양성비율) baseline,median→StandardScaler→LogisticRegression(C=1,maxiter2000,unweighted,일sampleweight1/24) baseline1,median→ExtraTreesClassifier(300trees,maxdepth10,minleaf24,maxfeatures.7,njobs2,class_weightNone,seeds 8383/1919/7171,일sampleweight1/24) 실험모델1. 주최결과는3seedprob평균(미보정점수),각seed도보고. 시드는CT2와같지만이분류목표에새학습이며새자료독립증거가아님. 어떤시드/모델을결과보고선택하지않음.
검증: 일그룹대상24행전체외부holdout. DIAG10은기존CT1원query_ids 360일각1회. FRESH7은2차각농장정렬일3개씩packet,fold=(packet번호+F13:0/F47:3)%7(새로등록한배치,독립새자료아님),pass2전체46일1회. EL1은2차각농장정렬일5개씩10fold. 모든outertrain공개360일에서같은farmquery±1/다른farmquery±3/잠금40±1제외;logit/imputer/ET모두이training만fit. 보정/calibrationfit0. no-leakreceipt train/queryids·featureshash·classcounts저장.
평가: 각h0/8/15/23에서일당1표본,각seed/ensemble별ROC-AUC,averageprecision(PR-AUC),Brier,logloss,precision/recall/F2/confusion(.5고정),.2는고정민감도보고만(최적threshold선택금지). 여러hour값을독립일로합치지않음. 고EC양성수/일반수와farm/pass별오류표보고. 우선h15,후시각h23가좋으면조기예측성공으로확장금지.
실험분류의유용성진단: h15 DIAG전체/FRESH7/EL1에서3seed모두prior보다Brier작고,DIAG농장×상대기록번호//5 block20000회 seed202610095의ensembleBrierΔCI95상한<0/p_worse<.025이면'prior대비유용성조건지원'. logit대비도같은방향/동일bootstrap별도보고하되선택기준대체아님. 분류유용성은EC회귀RMSE개선/기존채택기준통과아님/모델이득규모튜닝0. PR/재현율 CI는2차양성수적으니분모중심으로보고하고검정은Brier1family만.
캘리브레이션: h15 OOF고정bins0/.1/.25/.5/.75/1의n/mean_score/양성비율;데이터보고임계값조정/같은OOF에보정다시fit0. 확률표현은'고EC확률점수(보정전)'으로한정.
품질/재현: 전체수·키·일24h·타깃일평균·전체member fitreceipt/missing/유한값/codesha 고정. 별도비평3단계: 계획→DIAGfold0..1 중간→나머지CV+score→full400 최종3joblib저장/추론CLI/재로딩프로브+미래·다른farm변조불변test→최종독립score/산술/모델경로비평. 원파일덮어쓰기0,새실험/own폴더만.
이전중복: H3(6.94)/H15(6.111) 고EC분류기기반EC보정기각. 이번은구분품질그자체/새입력대비/같은시각확률/엄격분할・분류artifact제작이며보정기각을번복하지않는다.
