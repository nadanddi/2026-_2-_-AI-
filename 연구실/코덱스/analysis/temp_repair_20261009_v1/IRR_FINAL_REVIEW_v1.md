# IRR4 최종 독립 비평

2026-10-09 · 연구실 · 코덱스 critic. irr_results/보완_종합보고서/verify_irr 구현 및root검산receipt를검토하고별도무학습실제검산했다. 새fit0/EC정답0/외부자료0. 공식train_y는usecols=row_id,sub_temp만읽었다.

판정: **IRR4 구성원진단 완료·사전전체진행기준미달. 전체W40GS미진입/채택0 판단은 정확하다.** DIAG−.786383% 및p.00665<.008333/중앙98.333CI상단−.000253824℃는등록한DIAG통계조건을통과한다. 그러나EXT10+.135121%로방향조건을실패했다. EXT12−1.238777%나DIAG유의성으로이를덮거나사후cold문턱/farmgate를바꾸면안된다. 구성원실패가전체혼합효과부재증거는아니다.

## critic 직접실제 검산

12개IRR NPZ/source/recipe 및parentNPZ의SHA·metadatarecipe일치,각queryID순서·유일성·shape·유한candidate·parentBASELINE과의raw전수동일성을확인했다. 모든weights분포와seasonnotes는parent와exact같다. foldfeaturemaskreceipt셀수=trainrows×93,훈련day의동일farm검증day±1일purge도전수확인했다. 원패키지/공식CSV/고정seasonSHA보존도확인했다.

별도math.fsum검산에서DIAG9600/EXT10 2040/EXT12 5256행의기준·후보6RMSE가root와같다(EXT12후보표준산술차이약1.11e-16). F13후반DIAG .542401078→.552208935 악화,F47후반1.083644184→1.073478789 개선도직접계산했다. 각모델/농장의효과가동일하지않아임의farm선택은후속새가설이며현재증거에기반한채택이아니다.

root verify_irr_v1의86검사·최대1.11e-14·20k bootstrap/CI재계산PASS는실행receipt와구현을검토했다. critic은이번에별도20k재실행하지않았다. 이독립산술코드는같은저장예측에서계산을교차확인하는것이며별도모델훈련이나신선한검증자료확인이아니다. 첫baseline재fitrawexact는첫fold한번실증,나머지11fold는원검증된parentcache전수동일성으로연결된다.

## 보고서 비평

총49는CO₂36+IRR12+baselineprobe1의CODEX pipelinefit이고각각Ridge/LGB를포함한다. 새BASE/PFN0,전체혼합검증0,원제출정확재현문제해결0으로설명한것은타당하다. 18,878,880/6,388,464셀은반복feature비교셀합이며독립관측수·PFN8규정검사로확대하지않아야한다.

전문가설명에서발상한IRR4는실제멀칭/수온/센서depth가측정됐거나열전달식이검증됐다는뜻이아니다. BASE에기존유효일사정의가있으며CODEX잔차로이식한좁은후보라는설명이정확하다. float32/PFN/memory위치민감성으로원정확재현차이를단정하는내용도없다. PDF직접추출은root의자료검토범위이며critic이별도추출했다는주장이아니다.

소소한정의보완: 종합표의EXT10/12 '10/12℃미만날'은현재원시공기온도단독임계가아니라기존`ph_in_temp_3`일최소값의임계로정의된날이다. full입력세계에서정의한기존cold검증분할을보존한것이고실제배지온도10/12℃분할이아니다. 각cold분할의선택규칙을최종재개메모에붙이면해석이더명확하다.

세후보의screen미달이모든온도개선불가능/정보소진/CO₂무의미/일사물리무효를증명하지않는다. 동일DIAG자료를반복본탐색이라3후보bonferroni가전체과거탐색의선택편향까지보정했다고말할수없다. 향후일반화는새seed/context full검증및독립확인범위가필요하다. 현재근거에서원모델교체안함/새제출안만듦결론은일관된다.
