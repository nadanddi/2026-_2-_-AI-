# 실제 ET 학습지원 추적 사전 독립 비평

2026-10-06 작업계획 검토. PROTOCOL_v1.md/run_v1.py/leaf_core_v1.py/test_leaf_core_v1.py/preparation_v1.json을읽기전용검토했다. reviewer모델fit0. 새코드/원결과/계수/훈련자료를수정하지않았다. fit전run_v2의기록정합보완이예정되어있다.

## 사전 판정

계획과기본계산논리는타당하다. 현재검토범위에서6ET재현을막을학습누수/가중합공식결함은발견하지않았다. 이는재현·결과검산PASS가아니다. 기록단계의아래문제를fit전에새버전으로보완하면사전프로토콜정합성이좋아진다.

실제leaf지원은임의10변수KNN과다르다. bootstrap=False,균등훈련가중치,squared_error일때각tree의queryleaf훈련y평균을forest600개평균으로합한정확한예측계산이다. 이번추적으로고정ET가어떤학습정답을평균내높은점을만드는지확인할수있다. 해당행을제거하면오류가해결된다는재학습효과나물리원인까지입증하지는않는다.

## 실행 전 보완·주의

1. **지원 입력평균의NaN처리는원모델과맞아야한다.** run_v1 profiles는weighted_co2/heating에raw.fillna(0)를사용한다. 실제forest는medianimputer출력(float32)을사용하므로결측지원이있으면가짜0평균이된다. 원모델학습/예측은정상이며이문제는지원설명표기다. 예정run_v2에서실제X의해당열로평균을계산하고raw결측의지원weight를별도로기록하는보완이타당하다. raw와imputed/float32값을구별한다.

2. **parent정답통계가기록표에빠져있다.** 프로토콜은첫분기parent/양child의n/EC평균/고EC율을요구하지만현재first_split은parent_rows만저장한다. parent_mean_ec/parent_high_row_fraction/parent_high_day_fraction/parent_days를추가하고parentnode.value와평균도검증하면직접판독가능하다. fulltree+X/y로사후재구성은가능하므로현학습계산의결함은아니다.

3. **float32경로와입력열순서를실제추론과일치시킨다.** X/Q를float32로변환하고forest.apply에넣는현구현은적절하다. threshold는float64이지만비교입력은실제float32여야한다. rawfloat64를직접비교하는경로와threshold근접값에서달라질수있다. imputer출력열수38과feature_names_out의M.FULL순서를확인한다. 단순히열수만같은것보다순서명시가강하다.

4. **재현범위는모델별로다르다.** common3seed와purged1은4대상일96행,actual0은실제heldout2일48행,actual1은heldout1일24행으로합456query행이다. 원fold에서훈련에포함됐던F13_98등의예측을성공OOF처럼추가하지않는다. 원캐시queryID선택/ID순서/trainhash는현재prepare가드로확인한다.

5. **정확한보존식을행단위까지확인한다.** w≥0/Σw=1/y·w=예측에더해bad−good의ΣΔw=0/y·Δw=예측차를확인한다. weightShapley는그룹별Σrowφ=0, y·φ=기존predictionShapley, Σgroupφ=Δw를행마다확인한다. core의마지막두보존식과leafcount/weightedcount대조는적절하다. 전체treearray를저장하므로reviewer가이제첫공통분기이전의경로도독립재구성할수있다.

6. **가중치와자료의분모를구별한다.** high_row는y≥1,high_day는같은훈련날하루y평균≥1로별도표시한다. 지원row/day수는예측에쓰인훈련지원의크기이지독립검증표본수아니다. common3seed와600tree를독립사건으로세지않는다. signedΔw/φ는양수·음수모두있고일부높은y지원증가만보고낮은지원감소를누락하면왜곡된다.

7. **raw와centered정답기여순위는유일원인이아니다.** 총Δ예측은ΣΔw=0이어서훈련y에상수를빼도같지만행/날별순위는centering기준에따라달라질수있다. raw기여와훈련mean기준centered기여를같이제시하는계획은타당하다. top날을인과영향도나삭제추천목록으로읽지않는다.

8. **첫분기양child통계는leaf통계와다르다.** child의훈련y평균/고EC율은그아래전체subtree자료다. query가최종사용하는leaf지원과같은집합은아니다. 이후분기가계속되므로처음CO₂/난방분기하나에전체예측차를귀속하지않는다. groupweightShapley도고정0시5가족/32인공조합범위로한정한다.

## 최종 감사에서 요구할 근거

6모델source/preparation/cacheSHA와원train/queryID,bootstrap/criterion/훈련가중치조건,imputer·X/Q 재현을확인한다. 저장fulltree배열에서훈련/query/coalition경로를재계산해leaf지원·weighted/n_counts·노드y평균·예측합을독립확인한다. 지원profiles/행·날합계/Δw/weightShapley/첫parent-child통계를전수대조한다. 실제학습기록과입력분기가연결되는계산메커니즘이확인돼도전체모델/물리원인/현EC14효용/기각후보재개를선언하지않는다.
