# N5 실행 전 코드 비평

2026-10-10 연구실 코덱스. `N5_PLAN_v2.md·n5_run_v1.py`를 실제읽고검토했다. critic N5 실행·gate실자료효용계산·모델fit0이다. N4최종비평은별도로완료됐다.

**판정: 코드의주요가설·상속·계산제어는계획과정합한다. 아래finite/packet/감사범위항목을등록·무효용감사에서확인한뒤첫fold계산이가능하다. 실제auditPASS를이문서가대신하는것은아니다.**

## 닫힌 항목

- 원rawID로farm/day/hour를복원·정렬하고day별 expanding min으로관측prefix만사용한다. 현재NaN이어도이전에유효관측이있으면그이전최저를유지하며전체missing prefix만q1로처리한다. 원현재g는그대로다. 전체일미래min/전날rolling을쓰지않는다.
- parent12fold의NPZ/receiptSHA·원등록SHA/recipeSHA·3입력SHA·orderedquery/trainID를새recipe에묶는다. 새모델·weights·season/PFN을fit하거나변경하지않는다. N5사용출력은원공개fold예측에한정되고새실제평가행예측파일을만들지않는다.
- 수식은 `.4BASE+(.2+.4(1−g))*(CODEX+q*(IRR−CODEX))+.4gPFN`이고원BASE/PFN의혼합gate를q로바꾸지않았다. q0 exact와전체보완차식assert가있다. alpha7/pooled20k/원seed와사후7번째/채택False가고정된다.
- first는parentfold[:1]만계산하고cache가있어도전체집계전종료한다. N4최종비평/계획비평/해당recipe감사file을요구하며부모가첫중간전전체실행하지않는절차를유지해야한다.

## 첫 효용 계산 전 보완

1. recipe는‘finite in_temp’라고쓰지만구현은pandas expanding의NaN제외규칙이다. 먼저공식raw의비결측값이모두유한인지assert하고통과범위를receipt에남긴다. 비유한값이있다면계획과코드를정합시킨새버전이필요하다. source만으로원입력비유한0을확인했다고말하지않는다.
2. 원raw각farm/day의hour0..23지원·중복0을무정답단계에서확인한다. row_id유일성만으로중간hour누락이없는지는알수없다. 현재공식완전날입력만처리한다는계약이라면정확한지원assert가단순한해결이다. 누락을건너뛴prefix를계획의완전prefix라고묵인하지않는다.
3. 실행audit는sub_temp열negative만실제로호출한다. `sub_ec`도sourceassert에있지만`label_columns_rejected`라는실행결과문장은두열각각negative를하거나다룬범위를정확히구별한다.
4. 현재q≤g assert는있으나same-day q비증가/q=g이면N4exact/원g0 exact를직접검사하면계획항등식검산이완결된다. 이것들은효용튜닝이아니며합성·비채점검사에넣을수있다. q=g인행에서실제저장N4 raw와정합하는지도후속첫중간에서확인한다.
5. 23개suffix검사는전체farm/day의해당hour뒤값을한꺼번에바꾼것이고,shuffle·타farm변조는gate함수만의불변성검사다. 원부모모델/전체pred인과성감사로확대하지않는다. recipe는현재q신규학습특징이아님을명시하므로부모TRAINMASK감사상속과새q검사를구별한다.

## 추가 관찰

후속전체집계는cache receipt/recipeSHA를검사하지만그집계블록에서모든storedrow_id를다시등록queryID와비교하지는않는다. 계산단계에서는parentID일치를강제하므로현재immutable파일계약에서는차단결함으로보지않되,최종독립검산은전체orderedID·중복·24h·parentbaseline배열/수식전수를확인해야한다.

`second`가원FEATURE_COLUMNS에이미있는것을새후보로만들지않은판단은타당하다. 입력에phase대리값이존재하는것과그값이효과적으로학습됐다는것은다르지만,중복열추가를새정보발견으로포장할근거는없다.

N5는current gate의한계를사후해석한단일후보다. 현재계획·코드검토시점에서기각/통과수치계산0이며새채택/제출0을유지한다.
