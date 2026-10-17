# N3 0시 수준·당일 변화 두 머리 · 사전 계획

기존6b.46 H1/H2는physics(y)+daymean잔차level+평균을뺀shape 구조였고기각됐다. N3는같은두머리계열이지만진짜훈련y0를level목표로하고y(h)-y0를shape목표로한다. 확인한코드범위에서동일명세실행은미발견이다. family전체새발명이나실패반증이라고하지않는다. N2Ridge누적prior는이번안에서사용하지않는다.

## 모델과 자료 경계
- 후보하나. 원CF89/물리19/MASK/day/weight/season/분할유지. 하루키는season교체된day가아닌farm+원시t//24로고정(PAVplateau날합침방지).
- level훈련: outer훈련날h0행만. 목표그h0 sub_temp; weight도그행원w0만(하루meanweight아님). PHYS19+season(day)20열 median/scaler/Ridge100→CF89 잔차LGB120/.03/leaves8/minchild12/lambda15/seed9109. querylevel은queryh0input만으로추론;query0시정답은생성/학습에쓰지않는다.
- shape훈련: outer훈련날의각행y(h)-훈련y0. PHYS19의current-minus-same-day-h0 차분을medianimputer/StandardScaler(with_mean=False)/Ridge100(fit_intercept=False)로fit. h0차분은정의상0(원값NaN에도0),다른h의NaN은훈련median. 이는delta전처리규약이고실제NaN입력복원이아님. CF89 잔차LGB220/.035/leaves12/minchild100/lambda15/seed9109.
- query최종값=level_hat0+shape_raw(h)-shape_raw0. 모든query날h0행이있음을assert. shape0차감은동일날현재/과거입력에근거하며0시hiddenlabel아님. shape학습목표와차감뒤최종목적이다른한계유지·사후계수수정0.
- 원same-env CODEX baseline12fold/source/runtime/data/IDs/weights/season과재fitprobe를검사한다. 각fold훈련외MASK에서원CF89/PHYS19동등확인. 결측규약·행shuffle/globalfuturecut/다른farm/전체MASK 특징검사PASS범위는기존root결과와연결,queryy0숫자반입검사는별도독립source/negative접근검사.

## 검증·판정
- 모든훈련/검증날complete24/h0포함;query/purgey0라벨은목표생성에서완전히제외. prior/외부label0·EC숫자0.
- 한명세/seed9109의구성원진단. 이번누적target5후보(DROP5/MED3/IRR4/N2/N3) 기준alpha.025/5=.005/중앙99%CI(.005,.995),farm×floor(day/5)20kbootstrapseed20261010. DIAG10/EXT10/EXT12전方向개선+p/CI통과일때만full원W40G-S/freshBASE/PFNcontext/layout단계進入(구체full등록전fit금지).
- 원9전체개선/score.5127감소는구성원진단으로확정하지않는다. 구조실패후headweight/anchor시각/농장게이트/온도문턱튜닝0.
- 사전진단:실제queryy0는모든predict끝난후평가에만사용. e0=level0hat−y0,es=차감shape−(y−y0). 전체MSE=e0²평균+es²평균+2e0es평균을보고한다. 평균잔차직교분해가아님. rawshape/차감shape 두오차도같이기록. 각각좋아진다고전체개선이라고하지않는다.
- first모델후독립중간비평→남은12fold→독립산술/최종비평. 새제출/공식평가예측/확정구성0. 원9회정확재현미해결보존.
