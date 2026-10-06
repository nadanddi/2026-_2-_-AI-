# 단계1 완료 결과 독립 혹독비평

2026-10-06 연구실 코덱스. stage1_v2 완료receipt·32캐시·summary·explain_stage1_v1 결과 검토. reviewer는추가fit/원train_y/test/EL1/잠금정답/후보선택을하지않았다. 원source/캐시는수정하지않았다.

## 검산 판정

**PASS_COMPLETE_DIAGNOSTIC.** verify_stage1_v1.py --complete 정상종료 및verify_stage1_complete_20261006T183052480823.json 기록. worker.lock부재·receipt/outputSHA·32완료캐시서명·source/preparation/train/queryhash·원순서ID·4target±1배제·PFN16문맥RNG선택을확인했다.1152행에서R3→PFN4문맥평균혼합→현재누적평활→trainbounds clip→3seed평균을독립math.fsum으로재현하고16일점수를계산했다. ET600tree×2querypair×4context의입력열·분기반대방향·tree평균예측/rawET차이를확인했다.

원fold0 heldout48행replay R3최대차2.44249e-15,PFN최대차5.42402e-6과문맥ID/순서동일성을직접확인했다. 이는사전PFN허용1e-5를통과한수치재현이지bit동일또는fold전체출력재현이아니다.

verify_stage1_summary_explanation_v1.py는16day/8same-modelpair/4trainingvariation/모든ETfirstsplit bucket을전수대조했다. 설명4모델쌍각32개저장coalition예측에서exactgroupShapley를별도math.fsum으로계산하고sum=bad−good과가중변화량을재현했다. 각양끝coalition은이미완료검산된commonseed7 rawET/LGB0시출력과일치한다. hybrid모델추론자체를reviewer가재fit한것은아니며그부분은설명코드model.predict(f)검토와원설명가드근거를사용했다.

## 지지되는 결과

같은4개학습조건안에서F47_160과F13_98은모두RMSE≤.1성공이고,대응실패날F47_161/F13_112는모두RMSE>.1이다. 서로다른원fold의차이만으로원성공/실패대비전체를설명할수없다는반례가지지된다. 공통193일학습에서도F47예측 .738437/1.040214, F13 .406748/.675217의차이가남는다.

그러나'심각실패(|일편향|≥.2)가모든학습집합에서남음'은F13에대해틀리다. F47_161은4/4심각, F13_112는common/trainfold0의2/4만심각이다. trainfold1/8에서는편향 .141676/.176213,RMSE .142871/.176668로성공기준은넘지만심각문턱은넘지않는다. 원선별의'실패날'이름과재학습조건의문턱판정을구분한다.

0시공통seed7 ET의F47 raw예측차 .543243은grouped Shapley에서CO₂+.275892/난방+.250063 등이큰양수항을만든다. F13 ET차 .264322는난방+.158826/season+.102136 등이큰양수항이다. LGB는F47차 .035396에CO₂항이음수(−.032692)라ET와방향이다르고,F13차 .324426은season+.183662등으로크다. 이수치는해당고정회귀함수의두입력간연산설명이다.

## 심각도별 제한과 단계2 개선책

1. **높음 — 설명범위를ensemble오차전체로확대하지않는다.** Shapley는공통학습/seed7/0시/ET·LGB에만해당한다.3seed평균/PFN·MLP/평활후24시간편향전체의기여분해가아니다. ET .48/LGB .24를곱한mix_change도0시raw차에대한해당구성원부분이며최종실제정답오차를모두설명하지않는다. PFN원인을파악했다고쓰면안된다.

2. **높음 — LGB는전역안전anchor가아니다.** F47상황에서LGB가낮고상대적으로잘맞았으나공통F13실패112의0시LGB .720483은ET .671933보다더높다. 따라서ET높음/PFN높음이면LGB로당기는무조건보정은사례만으로정당화되지않는다. 실제후속gate/reference를하나의고정후보로등록하고일반EC이득과고EC과소/오차악화를전체eligible/noneligible상태에서검증해야한다.

3. **높음 — 4사후선택날은효용검증이아니다.** 기존실패와좋은대비날을선택한것이므로모델입력불충분/물리상태혼동/보정효과의일반화는미입증이다. 본결과는학습집합동일성의혼동을줄였고실제모델입력의출력차연산까지추적했다. 학습집합·feature설계·불측정상태가어느정도오류를유발했는지는분리되지않았다.

4. **중간 — 입력가족의인공치환은물리조작이아니다.** raw/heating_h0/tdm/tdz처럼같은입력가족을함께옮겨0시특징관계의일부를유지한groupShapley는원리상타당하다. 그러나season과제어/센서의상관을무시한32hybrid입력은실제운영분포밖일수있다. 기여도는goodreference·group정의·해당model에의존하며온실의CO₂/난방을실제로바꾸면같은효과가난다는주장불가다.

5. **중간 — first-divergence bucket은Shapley가아니다.** 한tree의bad−good출력차를첫분기feature에전부귀속시켜평균낸표는정확한tree차이계산이지만그이후분기의영향이포함되어있다. feature비중/인과기여로해석하지않는다. fulltree직렬화가없어reviewer는저장분기의대소관계와feature값을검증했지만더앞노드가공통이었다는실제경로전체를재평가하지않았다.

6. **중간 — 학습조건을늘려본것과새확증은다르다.** 새4조건은중첩학습집합이며3seed도같은데이터와PFNbag를공유한다.4×3결과를12독립증거로세지않는다. trainimputer/MLP내분할/PFN문맥/훈련season·bounds는recipe에따라함께달라졌으므로개별학습기록효과분해는아니다. 이번queryhash4조건이같다는것은입력frame동일성을확인하는좋은근거다.

7. **낮음 — 실제fit횟수와진단fit0표기를분리한다.** stage1은replay포함고전45모델fit/PFN17fit이며후속explain은ET/LGB2fit이다. summary/독립검산fit0은자신의집계작업에대한뜻이다. 단계1전체학습0으로요약하면틀리다.

## 단계2 사전등록 진행 조건

단계1의현재요구범위는완료됐고다음위험후보프로토콜을새로고정할수있다. 실제feature50열화이트리스트/risk라벨/C/reference/eligibility/threshold/cap와부분4outerfold×3seed의한번선행기각gate를학습전에고정한다. query라벨·inner_j·미래전체하루통계는피처에넣지않는다. 모든seed/지원표본/고EC보호를보고하고선행실패시기각·같은스냅샷계수탐색금지를지킨다. 통과하더라도전체nested/DIAG/A/B확증및현최신EC14비교가없으면모델채택·제출을선언할수없다.

현재진단수치의완료선언을막는결함은없다. 다음후보를등록하는것은가능하지만그후보의효용/통과가능성을본진단이보장하지않는다.
