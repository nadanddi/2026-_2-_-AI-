# N4 전체혼합 예비계획 독립 비평 v1

2026-10-10 연구실 코덱스. `N4_PLAN_v1.md` 실제파일과부모명세기준검토. 새코드/fit/predict/정답접근0. 판정은 **무학습준비진행가능,아래원시seed대조및실행계약을등록/코드로닫은뒤첫fold진입**이다. 현재문서만으로실행전검사가완료된것은아니다.

## 1. 이계획이확인할것과아직불충분한것

N4는IRR4의기존구성원실패를소급채택하는것이아니다. 일사4열보완CODEX를기존현재온도gate로부분반영하는 **새후보**다. 기존IRR4 cold실패를본뒤발상했으므로사후발상임을명시해야한다. gate의8/10℃가원모델에이미있어도새적용위치를독립적으로미리설계한것은아니다. alpha .025/6은이번등록된6후보보정이지모든과거탐색편향을완전히보정한것은아니다.

BASE3seed평균+새PFN49한문맥의paired예비검증은현재환경에서‘그혼합의CODEX차이효과’를보인다. **9회차원PFN1..8/원weights/원season/원출력이나새여러seed의일관성증거가아니다.** 통과해도최종채택0이며원8문맥과추가문맥/BASE시드별·사전확인layout의동일방향검증이별도로필요하다. seed49라는숫자만으로미사용이증명되지않으므로공식과거등록/캐시에서재사용여부확인또는‘이번새추출’로표현한다.

구성원IRR4의EXT10 +.135%가gate로해결될것이라는증거는없다. EXT10은일별 `ph_in_temp_3` 최저<10으로고른날이며그날의모든현재raw행이≤8인것이아니다. warm시각에IRR변경이여전히들어간다. 추위gate가예상한방향을보호한다는것은실행전가설이다.

## 2. 첫fit전필수보완

### CODEX seed726/IRR9109의교란

문서의기준CODEX726와IRR9109는입력feature뿐아니라seed도다르다. 이전726/727결정성기록이9109·새분할·새전처리에그대로적용되는것은아니다. **각새fold에서기준feature89 baseline726 vsbaseline9109 no-op 예측동일성검사를저장**하거나양쪽을동일seed로등록해야한다. 같지않으면feature+seed두변경효과로바로실행하지말고새버전에원인을구분한다. ‘CODEX동일’표현은정확한source·parameter·matrix·seed대조로지지해야한다.

### gate·혼합의수학계약

`c_new=c+g*(irr−c)`이므로전체최종예측차는

`Δp=(0.2+0.4*(1−g))*g*(irr−c)=(0.6−0.4g)*g*(irr−c)`.

단순 `.4g*(irr−c)`가아니다. g=0이면모든BASE/PFN상태에서baseline/candidateexact동일,g=1이면Δ=.2*(irr−c),NaN현재온도는g=1을합성검사로봉인한다. 원문‘0~8℃’는실제함수상**현재온도≤8℃모두**다. 결측temperature를이후imputed값으로gate에넣지않는다. candidate별gate재계산또는후처리추가를금지한다.

### 분할신선함과누출경계

`(i+2)//5%10`은앞뒤끝조각2/3개를포함한새block경계다. F47fold+3은해당농장분할라벨뿐아니라F13과동시에제외되는날조합도바꾼다. 실제old/newqueryID·pairtrainID·일별한번coverage·24h완전·buffer1을원시ID로등록하고동일split재이름붙이기가아님을확인한다. 새layout에도기존정답날이전부남으므로독립holdout자료라고말하지않는다. 다른농장같은날씨/같은시기의정답지원은여전히남을수있으며foldoffset이이를제거했다는주장은금지한다.

BASE는연속history가있어검증·purge입력NaN경계를통과한뒤훈련특징으로간접유입할가능성이있다. **전체모델의TRAIN world는test·query·purge센서NaN,QUERY world는같은농장현재이전실관측만**이라는둘의loader를명확히고정한다. feature전체를먼저full생성하고나중에행만제외하면안된다. weights의rank/coldEWMA와season vectors도outertrain입력만fit한다. 특징하나의미래불변검사만으로BASE/PFN전처리전체누출0이라고결론내리지않는다.

EXT10/12의FULL기반일최소정의는원분할선정규약으로만유지하며modelpreprocessing으로유입하지않는다. bootstrap의rawday//5와CV의정렬된날i5개block은다른단위이며두정의를등록한다. 농장층화인지farm붙인전체block에서pooled재표집인지 **‘농장×block’ 자연어만으로결정하지말고draw구현을고정**한다. CI통계량은ΔRMSE또는ΔMSE중선택하여sameRNG/draw와함께봉인한다.

### PFN/BASE의paired계약

baseline/candidate는동일BASE원3seed평균및동일PFN49예측을공유해야한다. PFNcontext의orderedtrainIDs/weights확률/RNG추출순서·float32행렬·checkpoint·V2/4estimators·CPU·query배치·dtype를저장한다. IRR전용4열을PFN에추가하지않으며PFNfeature목록은원89/계절자리로유지한다. BASE물리/잔차/Ridge/Nystroem参数와3seed평균を원패키지와맞추되fold-local변경점은원9회차조건과구별한다.

### 등록및계획·중간·최종검산

원source/전이적dependency/runtime/3CSV/checkpointSHA,정확새foldID·matrix·PFNsample·baseline/no-op契約·6후보목록·gate식·bootstrapdraw/policy를fit전에등록한다. cache는새foldrecipe와정합한원시구성원만재사용하며기존OOF를기준으로끼워넣지않는다. query라벨삭제/target변조는loader/feature와model입력경계까지확인한다.

첫fold후sourcepin/orderedIDs·trainquerydisjoint·weights/season·PFN샘플·BASE3seed평균·baseline726/9109no-op·gateexact·전체혼합식의독립검산을남긴다. 계획한독립비평없이나머지fold로계속하지않는다. 최종은예비status로고정하고모든3validator/농장×전후반/현재≤8·8~10·≥10·NaN/상위날집중도를보고하되그결과로threshold를고치지않는다.

## 3. 후속판단

조건부예비실험으로서N4는구성원만연속탐색하던한계를넘어전체혼합기여를확인한다는점에서목적이명확하다. 다만새모델학습과검증기재배치는원9회차대비평가RMSE개선증거를자동으로만들지않는다. IRR4실패본뒤동일DIAG정답재사용/단일PFN문맥/평균BASE한점의한계를그대로보고하고,예비gateFAIL이면그대로종료한다. 예비PASS일때도별도사전등록된full8·실제stochastic변동·확인layout이통과하기전채택/새제출/확정구성0이다.
