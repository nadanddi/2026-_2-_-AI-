# BLK 사슬 범위와 추가 제한 배치 비평 — 2026-10-07

부모전달현재30links/201components/1cycle/양끝samecomponent0/guard활성0은globalECmutual규칙의진단이며실제동/학습정답사슬전체의유효성기각이아니다. 기존3방법은사전등록대로채점하고guard0이면그규칙이baseline과동일한것으로보고한다. 결과보고threshold를완화하거나다른CH2를같은후보라고부르지않는다.

원소스 `집/클로드/research/ch2_label_chains_training_v1.py`를읽었다. CH2는trainEC endpoint trend+weather midnight+indoor/actuator비용의Hungarian1:1배정이며ECmutualmargin만쓴현재방법과다르다. 순수함수ref-only이식은새boundedbatch로정당하다. 소스top-level원CSV읽기·공유local쓰기·deep_cal/dong 비교를그대로가져오면안된다. 본검토는코드실행/fit/GPU/채점없음.

## 중요한 원CH2 구현 주의

`jump=(B0-A23)-.5[(A23-A22)+(B1-B0)]`는source/target양쪽hour22/23/0/1을쓴다. train-only사슬fit에서는모두공개train정보여서가능하다. **query h0에이jump를그대로적용하면queryB1입력미래누수이며queryEC0/1는모든시각에서금지**다. inputquerymatching에는h0용현재0시대비train끝/시간별서명과h>=1용관측된0/1시추세를명시적으로구분해야한다. 현재h0가주어졌다고B1입력을보간/forwardfill로몰래복원하고'관측했다'고처리하지않는다.

dummy설계C0=12의설명과구현도분리해야한다. 링크a→b를하지않으면a무후속12+b무이전12를둘다지불하므로고립쌍의링크비용은대략**24와경쟁**한다. '원CH2는cost<12만연결한다'고이식하면원코드동등하지않다. 여러링크전역경쟁에서단순cutoff와Hungarian도동일하지않다. 원dummy행렬을그대로고정하고독립소규모최적배정검산을한다. SCC/cycle탐지·종료를추가해야하며원while사슬길이탐색은cycle에서멈추지않을수있다. cycle처리는시간순서가확정된것처럼임의cut하지말고보류/fallback정책을사전정한다.

## 순환논리와gap의물리적길이

trainEC자정연속을비용에넣고연결EC가연속적이라는것은정의상성공이지query출처배정의증거가아니다. deep_cal/dong역시같은입력/라벨을재활용한추정치면독립정체성검증이아니다. label-conditionedchain은공개라벨구조가EC예측에쓸정보를제공할수있는가설이고누수없는보류행예측이그가치검증이다. 사슬자체fit품질과보류RMSE/coverage를분리한다.

BLK의rawrecord5/10개와gap1은배포상제거모양이다. 실제같은출처의physicalday가5/10/혹은다른길이인지확정되지않는다. CH2의next-physical-day자정비용을양끝5/10recordgap에그대로적용하면목표가다르다. 양끝의source-compatible연결/전이단서와경과시간은별개변수이며후자는unknown/복수후보로보존한다. 원recordhour보간은이미등록한가설로남기고'물리시간복원'으로명명하지않는다. query전체후반입력으로gap시간을추정하는것은금지다.

## 새배치 최대3방법 제안

1. **CH2_REFONLY_GUARD:** 원CH2비용/weights/SC표준화/dummy행렬을5520train-only자료로이식한다. 고정양끝flank가같은비순환label-informedcomponent에속하고currentqueryprefixinputassignment가그component를지지할때만기존BOTH보정,그외baseline. SC의0/NaN·missing비용은fit전에정의한다. 이것은CH2한이식경로의검증이다.
2. **FLANK_SOURCE_MATCH:** 단일endpoint만보지않고왼쪽3/right3공개학습flank의서명과trainCH2component/source가정일관성을사용해source-compatibleendpoint조합을선택한다. fixed각flank안의어떤23시/0시를선택할지규칙과양쪽불일치fallback을성능전에고정한다. 임의nearesttrain전체로확장하면genericneighbor이므로PF1/PF2와대조후중복이면별도실행하지않는다.
3. **PAST_QUERY_PREFIX_STATE:** frozenCH2/flanksource가설에동일farm앞query입력과현재0..h를누적해assignment확신/보류를갱신한다. query예측의과거prefix를쓰면이전예측만허용하며모델/표준화/threshold를query로fit/update하지않는다. 기록간물리연속unknown과source변경가능성을남겨두고다음query나현재h이후를보지않는다. 단일질의호출에서도이전queryprefix를재생해동일예측을낼수있어야한다.

이3개는추가boundedregistry이며원3방법결과와같은variant로병합하지않는다. alpha/clip/SG2scope는현재봉인baseline과같게고정하여한가지출처할당변경을비교한다. 데이터에맞춰계수학습이필요하면train내pseudo-blocknestedOOF로만하고outerquery양정답은절대전달하지않는다. 새stage의familycap·variant수·전체다중비교누적장부와기존validator확인조건을사전등록한다.

## 실행전요구

CH2純함수fit_reference/immutableartifact/queryprefixtransform API,원소스비용의합성동등성/최적배정·cycle검산, h0B1금지·futurequery/otherfarm/order/singlequery최종불변성,rawgap삭제,두정답격리,PF1/PF2featurelineage중복감사를통과한다. 후보성과와train사슬희소성/순환component/assignmentcoverage/绝对거리진단을분리한다. 정상labelchain의정보가없다/전체피쳐불가능이라는결론은이좁은시험으로내릴수없다.
