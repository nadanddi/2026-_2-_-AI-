# IRR4 전문가 단서 후보 계획 독립 비평

2026-10-09. EXPERT_PLAN_v1 및앞서읽은전문가_온도반영_v1의사전4열제안을검토했다. 새fit0/원파일변경0/외부자료0.

판정: **별도1후보의고정구성원진단은타당하며코드·첫baseline검토후진행가능**. IRR4는CO₂결과전에문서에작성된구체4열이고DROP5/MED3의계수·창을바꾼재탐색과다르다. 다만동일OOF를보는추가탐색이며후보간통계독립성을증명하는것은아니다. 계획의'독립'은별도입력가설이라는뜻으로한정하고신선검증이라는표현을피한다.

## 실행 전 고정

1. **4열정의**: rad_eff의fillna100·%변환·clip여부(원코드처럼clip없음)를고정한다. EWMA는farm/day내실시간순서,시간격자누락처리,adjust/ignore_na/min_periods를명시한다. 원BASEseg필터는ignore_na=True이므로이식이라고부르려면같은값을쓸지분명해야한다. hl1/3/8은CODEXreset필터와같은창이나BASEseg의1/2/4/8과완전히동일4열이라고표현하면안된다. 기존common에서radiation정의재사용과지연창선택을구분한다.
2. **변경영역**: 원89열·물리19·target·weights·season/order전수불변,신규4열만추가를assert한다. 평가querywindow에는현재/과거허용inputs만;학습은검증+purge입력도NaN으로변조한훈련세계동등성을확인한다. act_shade/thermal100의장치물리상태는기존관행이며전문가의확정방향설명이아니다.
3. **baseline캐시증거**: 각12fold원NPZ/recipe/queryIDs/row순서/trainIDs/±1일purge/data/source/weightscounts/seasonfitkeys일치검사. 첫fold동일baseline재fit rawexact는현재환경코드·설정의연결증거이며다른11fold재현성의직접실증은아니다. baseline재fit이불일치하면원인확인없이허용오차를사후추가하거나cache와새후보를혼합하지않는다. 양쪽새baseline을다시생성하거나그상태를보류한다.
4. **등록호환**: 이전registration_v1은보존하고IRR4 별도등록에원baseline recipe/sourcehash,현재runtime/run_irr SHA,12foldNPZSHA를연결한다. 원등록을IRR4소스로덮어써동일모델로보이게하지않는다. cache사용은이번새환경CODEXbaseline의범위이지원제출W40GS정확재현이아니다.
5. **통계**: 3후보탐색family에alpha=.025/3및중앙98.3333%CI상단=99.1667분위가명확하다. RNG20261009·20k 동일블록SSE/count를사용. 이전2후보의실패를이완화/후속결과로번복하지않는다. 후보선별정책으로모든검증기구성원개선을요구한다는뜻이며전체혼합효과를기각하는기준이아니다.

## 해석과 최종 검토

전문가가멀칭효과/근권지연을말했다고이4열의효과가증명되지는않는다. BASE에이미있는유효일사를CODEX잔차에추가해**구성원예측오차를줄이는가**를시험한다. 흰색멀칭·수온·sensordepth상수/입력을만들지않는범위는적절하다. 기각되면radiation물리가무효거나전체온도정보가소진됐다고말하지않는다.

첫중간검토는4열현재/과거/MASK/격리/셔플감사와firstfoldbaseline rawexact·원cacheSHA/등록을확인한다. 최종은공식sub_temp만읽어12fold예측·ID·purge·훈련season·RMSE/블록20k/p/CI·사전진행판정을독립산술로검토한다. 구성원통과도새seed/PFNcontext full단계등록을거쳐야하며새채택/제출아니다.
