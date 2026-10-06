# 부분 nested 위험학습 입력 준비 독립 비평

2026-10-06 연구실 코덱스. assemble_risk_inputs_v1.py/risk_input_preparation_v1.json/local risk_inputs_v1의12train/query쌍을 검토했다. 새verify_risk_inputs_v1.py로112구성원캐시와16inner분할을전수감사했다. 위험모델fit0. 단계1 --complete 검산은요청대로실행하지않았다. 원train_y/test/EL1/잠금정답이나다른AI코드변경없음.

## 판정과 검산범위

**PASS_PARTIAL_INPUT_AUDIT_ONLY.** 독립결과 verify_risk_inputs_v1_20261006T181913151863.json. 자료조립단계의outer/inner정답누수·중복학습·후처리불일치를발견하지않았다. 이는위험모델학습/성능/채택PASS가아니다.

- 각outer0~3의원train/queryID순서가원DIAG10캐시와일치한다. innertrain과innerquery는outertrain의부분집합이며서로분리되어있다. 각outertrain행은4개innerquery중정확히1회만등장한다.
-16inner의trainID는outerquery와innerquery의같은온실day±1을모두배제한다. innerquery에outerquery행이나innertrain행이들어가지않는다.
- 각innertrain/query의38FULL/14BASE 배열SHA,공개target배열SHA,학습타깃bounds가원preparation_v4와일치한다. season변환은각innertrain에서재산정되고원queryfeature배열을재현한다. 외부query도원outertrain기준season배열과일치한다.
-112cache쌍의NPZSHA/recordSHA/preparedSHA/seed/rowID/trainID를확인했다. PFN4문맥의원2000행RNG선택과context_row_id를독립재현했다.
- R3 .6/.3/.1,최종.48/.24/.08/.20혼합,구성원각각현재+.5누적평균,혼합평활및inner학습타깃범위clip을csv/math.fsum으로재계산했다. 합본train.A는기존nested OOF snapshot와일치한다. 외부query의실제계절v2구성원/평활/baseline도원공개OOF와일치한다.
-24개CSV의각farm-day가0~23순서완전24행임을확인했다. 따라서sort없는group.expanding으로작성한prefix_A도이파일에서는현재·이전예측만의평균으로재현된다. 파일순서를바꾸면향후다시sort해야한다.
-12manifest파일SHA/행수·train-query비중복·라벨의공개OOF일치를확인했다.12개는4outerfold×3seed로같은4개분할을반복한것이지독립분할12개가아니다.

## 후속 위험학습의 중요 제약

1. **높음 — feature화이트리스트가다음실행의필수경계다.** CSV에sub_ec와inner_j가포함되어있다. sub_ec는훈련라벨/평가채점전용이며query.sub_ec를riskfit·preprocessing·clip범위·문턱선택에넣으면outer누수다. inner_j는innerOOF분할메타이며외부query에는없으므로피처로쓰면안된다. 숫자열자동선택/모든열drop하나로feature를구성하지않고causal입력·모델점수열을명시한다. row_id/farm/day/hour도사용여부를별도등록하고고유ID를외워보정하는방식을피한다.

2. **높음 — 데이터조립라벨분리와위험모델의라벨분리는다르다.** 현재기본모델은innerquery라벨로학습하지않고외부outerquery는그outertrain에서도배제된다. 위험학습은각train파일의nested예측과훈련라벨만으로정의해야한다. outerquery라벨은등록된선행판정에한번사용하며query결과를본뒤후속계수/문턱/모델을탐색하지않는다. 성공/고EC보호기각gate를사후완화하지않는다.

3. **중간 — inner와outer모델의상태차는nested예측의자연스러운차이이며검증해야한다.** 위험훈련행의모델점수와season은더작은inner훈련집합에서나온반면외부query는원outer훈련집합에서나온다. 입력배열재현이맞아도위험점수분포/clip범위/모델불확실성분포가같지는않다. 이를모델상태전이의검증문제로취급하고같은조건으로조립됐다는이유만으로일반화성능을보장하지않는다.

4. **중간 — partial4fold는전체80nested/DIAG/A/B검증을대체하지않는다.**0~3완료스냅샷은독립선행선별용이다. 다른6fold와A/B,모든seed보호검증이누락된부분범위이며원작업의완료순서/Drive가용성에따른선택이다. 부분이득을전체이득으로선언하거나p/CI/Bonferroni기준을대체하면안된다. 같은날을공유하는seed를표본수3배로세지않는다.

5. **중간 — day全体라벨과prefix피처를구별한다.** 훈련위험라벨이하루정답평균/최종오차를사용하는것은등록된지도학습라벨로가능하지만queryfeature에하루미래정답/미래입력/미래예측평균을넣으면안된다. 현재prefix_A는시간순서검산을통과했으나A의미래시간값을한꺼번에요약한위험피처는새등록없이허용된것이아니다. 훈련일별whole-day제어분류를평가시각0에사용하는것도구별해야한다.

6. **낮음 — 원레시피와실제제출최신구성의범위를분리한다.** 준비자료는원계절v2(A)의nested/outer예측이다. 현재EC14에DP1/SG2가더해진실제최신모델의위험보정성능이아니다. 준비단계fit0은새자료조립에대한값이며112캐시가과거학습없이생긴것이라는뜻이아니다. 전체nested가진행중이면동일worker를새로시작하지않는다.

## 결론

현재입력묶음은원nested/outer예측과특징을정확히재현했고라벨/ID분리도검산됐다. 위험모델을만들기전라벨·메타제외화이트리스트,지도학습라벨,보정문턱및부분선행gate를고정해야한다. 그등록과코드경계를확인한후에야위험학습의누수여부와선행판정의정당성을추가감사할수있다.
