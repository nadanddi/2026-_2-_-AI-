# 고EC 양성일 LOOCV 독립 계획 비평 v1 — 2026-10-09

검토: PLAN_v1.md/run_v1.py, 지적반영 PLAN_v2.md/run_v2.py의핀보강.
판정: 일반일·고EC 양class학습을유지하고고EC 한일24행을통째로holdout하는일단위LOO 진단은사용자질문에적절하다. 모델/특징/시드/임계값을그대로두고검증묶음만바꾸는범위는타당하며 v2로prepare+첫2일학습을진행할수있다.

학습/누수: 기존독립검산한공개360일73특징과metadata를고정hash로재사용하며원train_y값/40일정답추가로드가없다. heldout당일과동일농장±1/다른농장±3/잠금±1의외부train제외,fold내전처리와양classassert는일관된다. 기존run_v1을OLD경로로명시하여import하는것도실행본이__main__인구조에서맞다. high26일각각LOO하더라도purge로다른high가빠져train양성이항상25일이아니라는계획표현은정확하다.

평가/선택: 양성only에서는constant score1도완벽recall/positiveBrier를얻을수있다. 따라서점수가높아져도구분기전체개선/일반일오탐개선/확률보정좋아짐이라고말할수없다. AUC/AP/precision/FPR/accuracy를보고하지않고기존DIAG일반일성능을이번LOO오탐으로쓰지않는계획은좋다. h15주기준과.5/.2고정민감도,각seed와ensemble을모두보고하므로사후threshold/시드선택도없다.

paired/통계: 같은26일의기존DIAG와비교하면조회일차이는없지만train규모·구성·positive비율이함께바뀐다. 따라서더많은고EC를줘서좋아졌다는원인단정불가. 26고EC일은연속시기/같은농업사건일수있고LOO train이겹치므로bootstrap p는관측OOF차이의탐색근사이며검증훈련변동전체를반영하지않는다. 새독립확증/일반분류채택기준대체금지는끝까지유지해야한다.

확인된수정: 최초v1의score는구DIAG10 cvCSV10개를읽으면서sourcepin은읽지않는h15CSV를등록했다. v2에서실제로읽는10CSV와자체code/PLANv2의SHA를등록하는것으로해결됐고수정전파일은보존됐다. 학습/평가설정은변경없다.

중간bootstrap예외: 비평가가실제NumPy환경에서 rng.choice([],size=(1000,0),replace=True)를직접실행해shape(1000,0)로정상동작함을확인했다. 첫2일이한농장뿐이면다른농장빈pool이관측되는것은오류가아니다. 그경우2일/해당블록수에만기댄퇴화CI를유의확증으로부르면안된다.

다음검토: prepare의26복합farm/day분할·trainclass count·필수sourcehash, 첫2일24행query·공개target1·prior비율·ensemble평균, .5/.2탐지/positiveBrier/logloss/paired gainloss와bootstrap를별도산술로검증한뒤rest24진행을판정한다. 추가fit튜닝/새최종400모델/보정·제출은필요없다.
