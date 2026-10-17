# N2 대상 누적 prior 계획 독립 비평

2026-10-10. N2_TARGET_PLAN_v1 정적검토. 대상fit0/EC수치0/외부자료0. 타농장LOFO gate를별도독립검산해실제통과를확인했고대상효과와구별한다.

판정: **별도고정누적1열대상후보의사전감사·첫fold진행은타당하다. 다음경계를코드에고정해야한다.**

1. prior훈련지원은LOFO등록eligible48개pair전체그대로,원30열/inf→NaN/medianempty0/Scaler/Ridge100/lossfarmweight규약을보존한다. source/runtime/dataSHA+eligible키/rowIDs/전처리통계/coef/targethash를보관한다. priormodel의대상정답접근0을원코드경로로확인한다. 대상fold마다prior를refit안하는것은대상F13/F47가priorfit에서완전제외되므로타당하다.
2. 대상Δ추론은samefarm/day의h−1→h 연속입력, h0=0. 중간현재/직전in_temp나연속시간결측이면ΔNaN이고skipnaFalse누적은그날끝까지NaN. 기타featuremissing은타농장fold에서고정된imputer로처리하되inf→NaN을동일적용한다. 정답lag/진짜배지h0를anchor로쓰지않는다. h0부터현재h까지만의Δ합으로각query인과성을확인한다.
3. **누적드리프트**: 1시간MSE개선이23h합의안정성을보장하지않는다. 증분잔차간상관·1hprior평균편향이누적되어구간오프셋을만들수있다. 시간대별range/NaN빈도/극단값은서술감사로기록하되결과후누적clip/alpha/표본gate/농장gate를붙여선택하지않는다. prior학습이Δ라서실제타깃수준편차를자동제거한다고말하면안된다.
4. 원89+물리19/season/weights/row순서불변및새1열만추가를전수assert. officialtestMASK·globalfutureT·otherfarm·shuffle·validation/purgeinputNaN세계의trainfeature동등성을실제검사한다. 타농장고정prior의featurepreprocessingstats는target변조에불변이어야한다.
5. baseline12foldcache출처/원source/runtime/dataSHA·parentNPZ/query IDs와predraw전수/weightseasonexact를연결하고firstbaseline재fitexact. 실패시사후tol완화없이출처/환경원인부터해결한다. 원9CSV재현성미해결을이현재환경componentbaseline연결로해결됐다고하지않는다.
6. 4후보familyalpha.025/4=.00625·중앙98.75CI(.00625,.99375)·블록20k/seed20261009/3검증기strict방향이명확하다. candidate하나에사후여러누적규약을시도하지않는다. componentgate는비용정책이고전체혼합효과기각증거가아니다. target효과검사에는타농장LOFOmacro수치를섞지않는다.
7. full진입계약의3새BASEseed/2PFN문맥family/3validator/새layout은추후fit전구체등록이필요하다. 새layout의day/purge/MASK/seasonweights정의와p다중비교·seed×family모든칸판정을확정한다. 새seed는자료독립성이아니며이미본DIAG의새p를신선자료처럼부르지않는다. fullbaseline/candidate사이에는BASE/PFN자체feature불변이면동일freshmembercache를공유할수있지만전체source/row/context/checkpoint/querybatch계약이같아야한다.

계획은전문가원인확정·타농장baseline손쉬움·LOFOPASS를대상성과로승격하지않는다. 목표는원9보다낮은실전RMSE이나지금완료는가능후보검증절차의일부다. 원자료/제출소스보존·새답안0을유지한다.
