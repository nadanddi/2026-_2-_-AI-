# N5 현재까지 최저온 gate 계획 비평

2026-10-10 연구실 코덱스. **검토대상은한국어완전계획 `N5_PLAN_v2.md`**다. N5 결과·첫fold효용 계산0/fit0. N4최종수치·비평은별도파일에서닫았다.

**판정: 현재읽고검색한공식온도코드범위에서IRR4에동일prefix-min q를쓰는이미기각된구현은찾지못했다. N5는N4와다른새사후가설이지만현계획만으로이미개선단서가있다고볼수없다. 아래결측·인과·상속·중간종료계약을코드/등록으로명확히한뒤첫fold까지만계산할수있다.**

## 기존과의차이·반례

N4는현재rawair gate g를보완CODEX에도썼다. N5는원혼합의g를유지하면서보완크기에같은날prefix최저온 q를쓴다. 현재관측이유한하면prefix최저≤현재이므로0≤q≤g다. 따라서고정멤버에서보완절댓값이증가하지않고,추웠던뒤시각의보완을추가억제한다. 그러나보완의부호·오차와의정렬은알수없으므로 **보완축소=RMSE개선은아니다**.

원TF cold가중예외와EXT의하루최저분할은이미있지만이들은훈련가중치/검증정의이고query보완prefixgate와다르다. 기존temp_pfngate_v1/v2는현재온도게이트,조건혼합G1은인과12조건으로비중학습,IRR4/N4는등록한일사보완이다. 같은보완의prefix-min gate는읽은이범위에없었다. 공식온도코드의cummin/minimum.accumulate/expandingmin/prefix-min 검색에도동일기각실행은발견되지않았으나전체git/ignored파일전수부재를증명한것은아니다.

EXT10의원조건은연속EWMA3h의 **완전한날최저<10**,N5는당일현재prefixrawmin의8~10 gate다. 시간필터·리셋·threshold·가용시간이모두다르므로‘EXT10조건에맞춘합법구현’이나‘같은cold-day를탐지’라는말은부정확하다. 정당한새가설문장은‘현재최저관측으로추위가지속될수있다는보수적보완제한을시험한다’이다. 최저값하나의잡음/복원값이뒤따뜻한시간의유효보완까지하루내내차단하는것은강한반례다.

## 첫계산전고정할계약

1. **결측규칙:** valid를유한값으로할지notNaN으로할지고정한다. 현재NaN이어도이전에유효관측이있다면prefix최저는그이전최저다. `cummin`이현재NaN을반환하는경우이를곧장q=1로바꾸면계획과달라진다. 전부missing인prefix만NaN→q1이며관측이새로생기면누적최저를적용한다.
2. **day/순서경계:** 원raw farm/day/hour로정렬하고현재0..h만누적한다. 계절치환된day로그룹핑하거나전날min을이월하지않는다. full-day min/rolling24/forward/backwardfill로미래값을채우지않는다. 불연속hour·중복ID·잘못된daypacket은거부또는사전정의대로처리한다.
3. **혼합수식:** `Δp_N5=(.6−.4g)*q*(IRR−CODEX)`이고원PFN/BASE비중은g다. q를원혼합전체gate에대입하지않는다. q0 exact/g0 exact/q=g일때N4exact,유한현재관측에서q≤g 및day내q비증가를합성/실제입력단계검사로봉인한다. 실제N5효용은이합성검사에서계산하지않는다.
4. **상속:** 12개원N4 NPZ/receipt/registration/runtime/source/공식입력SHA·orderedquery/train/contextIDs를새등록에고정한다. baseline·BASE/PFN/CODEX/IRR raw배열을불변재사용하고새모델fit/기존weights/season수정0. N4feature감사의제한범위와source계보를명시한다.
5. **인과검사범위:** 원foldquery의같은날0..h rawinputpacket을사용한다. 각h0..22의suffix변조·다른farm변조·행shuffle·전날변조와q결과불변성을검사한다. h23은미래suffix가없고검증의미가다르다. 모델배열불변재사용을‘새fullmodel인과검사’라고하지않는다. 모델fit0인상태의TRAINMASK독립성은원모델출처를보존하는검사이지q학습누수검사가아니다(q는학습하지않음).
6. **판정과절차:** 첫fold계산뒤효용으로설정변경하지않고독립중간검산전전체계산금지. pooled block/20k/seed20261010/ΔRMSE/alpha.025/7를그대로등록한다. 같은정답을재사용한사후7번째후보라원9회개선/신선한확인/최종채택으로설명하지않는다.

## 결론

현재조건과하루최저조건의차이를깨달은것은해석정정이지성능개선발견이아니다. N5는prefix최저가뒤시각보완의신뢰성을설명할것이라는미검증추가가설이다. 한개고정안으로만시험하고FAIL이면q회복시간/threshold/농장별gate를사후추가하지않는다. PASS여도단일PFN49예비결과를채택하지않으며별도full8·실제seed·확인배치·전체예측인과확인이필요하다.
