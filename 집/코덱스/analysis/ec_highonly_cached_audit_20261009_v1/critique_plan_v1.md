# 고EC전용저장수치회귀 감사 독립 계획 비평 v1 — 2026-10-09

검토: PLAN_v2.md/run_v1.py와원MX1/WT0 members소스. 아직숫자분석실행전이다.
판정: 사용자질문의수치회귀해석에기존H캐시의조건부성과를확인하는감사는할만하다. sourcehash등록후처음2fold집계를진행할수있다. 이는일반일정보를특징/후처리까지전부삭제한순수신규실험과같지않으며, 사용자가엄격삭제를요구하면이감사만으로그요구를완료했다고말하면안된다.

학습/원source: MX1은해당outertrain의일평균>=1.2를hi_tr로만들고H용R3에tr[hi_tr]를전달한다. WT0는전달된tr만ET/LGB/MLP 및MLP전처리에사용하는소스구조다. BASE는같은feature구성과시드이지만전체outertrain으로fit한다. <3high이면H=BASE fallback이있는데현재감사는모든fold high>=3과cachedcount일치를assert하므로fallback을고EC전용학습이라고잘못부르지않게했다.

purge/정합: MX1의동일농장query±1/다른농장±3/잠금±1 규칙은기존DIAG분류기등록과소스상같다. 현재코드는고유8640row키와fold/farm/day/hour/y를CT1과정확대조하고meta.high로평가26일을선택하며상수평균은원공개outertrain중고EC행만계산한다. 학습고ECcount도분할마다대조한다. 독립검산은키/일24행/전체highcount뿐아니라실제train복합키세트를재구성해야한다.

공유일반정보한계: H의season은전체fold입력에서이미생성됐고lo/hi clip은전체outertrain정답범위다. shrink도공통후처리다. 고EC만supervisedfit한사실은일반날의모든입력/통계/라벨후처리를삭제했다는뜻이아니다. 원기존코드의간접의존CT1/core와full재학습은이번감사범위에서완전히검증되지않으며cache작성당시코드SHA를역으로보장하지도않는다. CT1 sourcepin도가능하면등록보강하고전체누수검증완료주장을피할것.

평가/원인: 고EC26일은정답을미리알고선택한conditional모집단이다. hourlyRMSE와dailymeanRMSE/MAE를구분하고일당24시간을26*24 독립날로세지말것. constant-highmean은고EC임을아는조건의크기기준이지분류baseline이아니다. 새LOOCV가아닌묶음DIAG10이며기존라벨노출자료/3seed중첩query는새확증이아니다. 숫자가좋아져도실전전체/정확한고EC선별/회귀gate효용을확정할수없다.

실행보강권고: score열과target에isfinite확인;ddof=0 std의정의명시;상수평균의trainhigh samplecount와fold범위기록;transitive소스검토한계를마지막까지유지. subgroup빈표본은skip과n분모로명시한다. 처음2fold의high일이극히적으면중간해석은버그확인에한정하며good/bad를전체26일로확장하지않는다. 최종에모든고정scope/seed/농장구간과일평균오차기준수를독립math.fsum으로재계산하겠다.
