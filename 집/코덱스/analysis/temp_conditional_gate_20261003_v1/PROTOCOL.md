# 条件別温度混合 G1 · 실행 전 고정

사용자 제안: 모델이잘하는분야에맞춰행마다비중을조절. 기존W30G도현재온도g에따라변하는gate이지만추가운전/시간/동역학조건은없다. 이번G1은12조건으로BASE와CODEX의상대비중을학습하며PFN .3g와≤8℃보호를유지한다. PFN의비중까지새로학습한3전문가gate라는주장은하지않음. 카탈로그6.67~70 고정coldgate/6.91 조건탐색/6.149 ECwarm혼합실패와구별,모든유사연구부재주장없음.

- outertrain 내부3fold(농장정렬일5일묶음순환,±1daybuffer)에서 **BASE전체(.65물리+LGB/.25Ridge/.10Nys)** 및원CODEX를새로fit,innerquery예측을생성한다. 기존글로벌다른fold OOF를메타학습에재사용하지않음(outerquery정답이다른fold적합에들어가는간접누수차단). BASE seed7/101, CODEX726/727.
- gate조건12열: farm_id/sin/cos/in_temp/in_temp_mean/in_temp_std/in_temp_diff/delta/out_rad/act_vent/act_heating/in_hum. median/scaler는outertrain조건만fit. Z=[1,standardized조건],g=원W30G온도gate,B/C=innerOOF, beta0=(.4+.1g)/(1−.3g).
- Ridge100 no-intercept: design=Z*g*(B−C), target=y−beta0 B−(1−beta0)C. 가중치원TF. θ적합후 delta=g*clip(Zθ,−.15,.15). 학습모델이판단한분야별우열로연속적인비중변화,극단1모델몰빵방지. inner학습목표는PFN을제외한B/C normalized pool의RMSE이며PFN최적비중학습이아님. 전체W30G외부검증으로실효판정.
- 최종wB=.4+.1g+(1−.3g)delta, wC=.6−.4g−(1−.3g)delta, wP=.3g. 세가중치비음수/합1,≤8℃delta0. 기본모델예측은변경없음. effective CODEX=C+(1−.3g)delta(B−C)/(.6−.4g)는기존검산runner호환용표현이며새구성원모델학습이라고해석하지않음.
- outerBASE/CODEX/PFN캐시는원모델동일행만재사용. gate fit에outerquery정답/조건적합없음. 같은온실현재·과거입력만,train MASK. 입력전처리/experts의imputer/scaler는각innertrain만fit. TF기존global입력rank고정한계.
- 후보1안,기존15+G1=16保守本페로니. 원DIAG10/EXT10/EXT12,2seed×2PFN문맥=12칸全RMSE감소及DIAG20k농장5일블록 p_worse<.025/16,99.6875% ΔMSE CI상한<0. RNG20261003. 통과하면전체W30G 날씨GUARD재학습을자동계속하고그후채택. 실패면이번구체gate기각.
- 내부학습계산/외부분할/계수closed-form/독립가중합/콜드불변/원시labels/fsum/bootstrap/첫foldraw재학습 검산. 내부gatefit점수는낙관적훈련점수이며성능증거아님. 원외부CV도반복사용·미사용홀드아웃아님. 실질CODEX시드독립아님. 제출/test평가예측/EC잠금0. 소스main사전커밋·원본수정0.
