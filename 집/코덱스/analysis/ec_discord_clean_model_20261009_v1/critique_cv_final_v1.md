# 정제 EC 모델 CV 최종 독립 비평 v1
2026-10-09 · ec_discord_clean_model · CV부분만완료, 최종400일 모델 제작은별도진행

판정: **현재 삭제규칙으로 모든 R3 구성요소를 다시 학습해도 두 공동검증 모두 실패하고 전체 효용은 악화했다.** 사용자 요청한 새데이터/후보모델은성능과무관하게끝까지만들되공식채택0/제출0. 모델저장검증완료를성능개선과혼동하지않는다.

## 실제 독립 검산 근거
verify_independent_v1.py final 실제exit0,final_independent_v1.json PASS. 생산자run/recipe import0. public/fold원X/Y·등록cache/source핀·기존360정답목록·각fold48시점purge·24행통삭제원값보존검토. selector_snapshot47열의거리/tie/자기±1배제/동일농장5NN/이웃중앙EC/gap/삭제ID를public와10fold모두독립재생했다. sourcefeature생성전체는재작성하지않아snapshot생성및matrixhash는소스검토범위이다.
원q360일8640행비중복,고EC31일/일반329일. 조건부q는18일432행제외해342일8208행이며남은고EC13일도포함한다. 조건부삭제는각outertrainselector참조의q target선별이고globalpublic삭제12일과달라야하는것이정상. CVtrain은globalclean사용0.
각fold9개ET/LGB/MLP×3seed receipt=90모델,훈련IDs/훈련rows/IDsha·등록meta전수확인. baseline재현3개ET0/LGB0/MLP3.1086e-15. BASEcache개별raw.6/.3/.1혼합/seedensemble, CLEANmixraw seed평균 및모든arm prefix .5/.5/훈련clip/상수예측전수산술을재계산했다. 모든그룹3지표RMSE/bias/상수RMSEfsum일치,rawmixRMSE추가. farm×day//5상대기록번호구간bootstrap20000두scope의p/CI도일치.
프로파일public8640→8352행(12일삭제),originalkeys/24h완전/target유한·음수없음/원값유지 확인. source/hash변경없음. 원train_y의추가40일은이번CV검산에서수치읽기/새채점0.

## 사전 판정
두공동조건 filtered/all 각각모든3seed 개선및p_worse<.0125를요구했다. 실제는두scope 모두7/101/2024에서악화. filtered p.6881/allp.9998, 둘다기준실패. 전체및pass2all≥2%손해guard도둘다위반. 좋은하위집단으로주평가지표나임계값을바꾸지않는다.

| 범위 | 일수/행수 | 기준RMSE | 정제R3 RMSE | 변화 |
|---|---:|---:|---:|---:|
| 원q전체 |360/8640|.1851531699|.2263310230|+22.2399%|
| 조건부q유지 |342/8208|.1248909203|.1302841471|+4.3183%|
| 조건부q제외 |18/432|.6239205307|.8378600233|+34.2895%|
| 일반 |329/7896|.1107173171|.0988796965|−10.6918%|
| 고EC |31/744|.5177003112|.7007960119|+35.3671%|
| pass2전체 |46/1104|.2550615151|.3472668929|+36.1503%|

공동조건delta filtered+.0053932267813/95%CI[−.0137120,.0257341],all+.0411778531169/95%CI[+.0184656,+.0660285]. CI95는서술용이며사전alpha.0125판정을대체하지않는다. filtered표본은q실제label선택으로실제평가에서선정할수없는모집단이다. 그유리한조건에서도개선은확인되지않았다.

## 농장·구간 및 반례
| 층 | 원q전체변화 | filtered변화 | 일반변화 | 고EC변화 |
|---|---:|---:|---:|---:|
| F13 pass1 |+23.9473%|−2.2111%|−6.3551%|+39.2235%|
| F13 pass2 |+23.4208%|−22.3688%|−22.3688%|+39.0901%|
| F47 pass1 |+13.5411%|+13.9404%|−9.7863%|+22.7652%|
| F47 pass2 |+49.2261%|−16.2855%|−16.2855%|+67.3629%|

‘제거대상만손해니유지대상은괜찮다’의반례:조건부유지고EC13일중12일손해,net SSE손해30.886991. F13 pass1 유지고EC5일RMSE+13.0790%,F47 pass1 유지고EC8일+48.9293%. 일괄삭제가평범한고EC지원도잃게하거나전처리/season/bounds를바꿔생긴손해일수있고원인을각경로로분해한실험은아니다.
제외q18일은전부SSE손해(net135.100479),전체고EC31일중30일손해(net165.987470),반면일반329일172일개선/157일손해(netSSE이득19.591002). 따라서일반집단의이득은고EC손해에상쇄되어전체악화. 일반gross이득의상위5일비중70.4801%로일부날에집중된다.

## 학습격차·폴드분산
CLEANensemble학습RMSE receipt의10fold평균.03305212/범위.02768016~.03638794. 원검증foldRMSE평균.21659817,filteredfold평균.12487327로학습보다크다. trainraw원예측미저장으로학습수치자체는receipt/source검토이고격차산술만독립확인. baseline학습오차는없어정제만추가과적합을만들었다고단정하지않는다.
원qfoldSD BASE.06373453→CLEAN.07037298,filteredSD.02409699→.03824806. fold별단순평균/SD는가중전체RMSE와다르며부트스트랩/사전판정대체0.
CV CLEAN각kindraw및CVjoblib미저장으로각.6/.3/.1항의독립수치재생은불가;소스혼합계약검토로한정. 저장mixraw이후계산은전수검산완료. 최종9joblib은각구성원직접predict와실제bundle경로일치로보완예정.

## 결론 공격과 범위
‘온전히새데이터로학습했으니좋아졌을것’: cleanseason/ET/LGB/MLP/imputer/scaler/bounds전부재fit했지만원q조건부·전체둘다실패. 깨끗해보이는proxy삭제가정답오류수정의증거는아님.
‘일반일만보면성공’: 일반329일은사전공동주모집단이아니고고EC를평가에서미리배제불가. 사후일반지표로재정의하지않는다.
‘삭제날은평가에없다’: 이전별도집계하한진단은전체평가고EC부재를반박했다. 이번proxy불일치날의존재/부재는아직모른다. 점수분석으로새계수/모델선택은하지않음.
‘이모델을만들면안된다’: 사용자가실험모델생성을명시했으므로요청은완수한다. 채택/제출은별개.400일finalfit은CV완료뒤추가40일을훈련자료로쓰고40재채점0,360CV값을400finalmodel의직접검증성능이라고말하지않는다.

현재는CV검산완료이며최종9artifact완성/independent 구성원mix+실제predictionpath/8dayh5·h17causality검증은남았다. 전체작업완료선언은그관문후. A/B·새holdout·새seed확증없음,채택0/새제출0.

근거: final_independent_v1.json,final_cv_critic_supplement_v1.json/.py,data_independent_v1.json,원등록/profile/foldmeta,독립verify_independent_v1.py. 원본·기존결과변경0.
