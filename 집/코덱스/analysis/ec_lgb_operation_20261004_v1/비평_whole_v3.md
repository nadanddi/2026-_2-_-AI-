# family21 whole verifier v3 정적 사용 판정

2026-10-04 · 집 코덱스. verify_full_v2 전체와 v2→v3 diff, synthetic_verify_full_v2/v3 저장결과를읽었다. run_v3/preparation_v3/preregistration의SHA를PowerShell로재확인했다. 실제candidate CSV/aggregate/firstfit결과를읽거나fit/predict/실제score를실행하지않았다.

판정: **verify_full_v3는전체66셀/aggregate완료후root가whole감사에사용하는것을허용한다.** 현재그학습의완료/재현/성능을PASS판정한것은아니다. 최초baseline실패때중단하는조건과등록된실험범위는유지된다.

run03f9f6030fe3fbe84625b1c9dfef592ba6c265a87016ef1a61ac7cbae2f7e5a4, prep6a5bf8b1f30b73d11d63cad8b52a8f2c61aa5b4299355bd825b9232c9aebdea2, prereg381719e67e40495a9f07d74a26184f543950965dd82f701ef4aaef118879bff3의현재디스크SHA와v3pin이맞는다. v3는최적화Python을차단하고runtime모듈/배포버전,source/입력/cache,prepared/fit/first/CSV schema와서명을확인한다. 원14/새23/full38train/query서명을재생하고원R3guard66개를독립호출한다. first감사날짜/시각목록도정확히검증하며NaN·음수·문턱초과오차를거부한다.

원ET/MLP/PFN·baseline·bounds와candidate식을재계산한다. raw차=.24*(newLGB-oldLGB),단일shrinkclip,math.fsum scalarcandidate를대조한다. 예상22fold×3seed66cell,83,160행과완전aggregate를확인한뒤에만RMSE/구간/bootstrap으로들어간다. 엄격15칸방향과각DIAGseed p<.025/21 및조정CI상한<0를동시에요구한다. publicPASS도pendingreview로표시하므로새holdout채택판정으로확대하지않는다.

합성저장결과는13오염거부/40,292체크PASS다. 소스의오염은firstaudit NaN·음수·초과·날짜·누락키·서명·추가키7개와CSV추가열·중복ID·floatkey·비유한값·aggregate값/순서6개다. 이는해당로컬validator의거부동작과후처리/constantlossbootstrap산술을검사한다. 원cacheprovenance·전체66manifest·실제재학습을합성으로끝까지검증한것은아니다. 일정lossdiff의bootstrap은이중합산/RNG결속을확인하지만일반분산표본의경계p/quantile민감도를폭넓게시험한것은아니다.

bootstrap block은farm별날짜정렬후**관측된일5개씩**묶는다. 달력상반드시5연속일이라는검사는없다. 농장별층화는있으나공유외기날짜의농장간의존성은공동resample하지않는다. 이는사전고정된정의의재현이며본비평을계기로좋은결과에맞춰변경하지않는다. 향후보고서에서는'5관측일farm block'이라고정확히표현한다.

한계: 저장firstaudit수치/서명만확인하고모델predict를재실행하지않는다. source/manifest재생이모든fold의인과합법성을새로증명하는것도아니다. 사전등록시점은부모의Git ca9d339 등등록근거에의존한다. public반복탐색·실제lab대비범위·MASK저장근거의제한은이전비평과같다. 부분출력실패는기존파일보존후새verifier출력버전이필요하다. 실제score0인본읽기검토는candidate결과의개선또는실패를알지못한다.
