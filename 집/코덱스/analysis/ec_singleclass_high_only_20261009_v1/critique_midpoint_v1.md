# 단일class 고EC분류 독립 중간 비평 v1 — 2026-10-09

판정: 예상한퇴화가실제첫seed에서확인됐고자료/분모/산술오류는없다. 고정나머지2seed만실행해최종확인을마무리할수있다. 새학습규칙·threshold선택·104fit·모델저장은불필요하다.

critic_recheck_v1.py midpoint를직접실행해등록sourceSHA와모든26LOO학습subset이양성label1만포함하는것을확인했다. 대표LOO000 train은20고EC일480행이며held고EC24행/일반8016행과교집합없다. 실제8383seed의8040행score와판정은전부1,각335일24시간완전그룹도확인했다. 따라서held1고EC일1/1탐지·일반334일334/334오탐이정확하다.

해석: classes_=[1]인기존supervisedET와positive_score정의에서모든입력score1은한class분류기의퇴화다. held고ECrecall100%만강조하면무의미한성공인상을주므로일반FPR100%와같이보고한다. score1은현실고EC확률이검증된100%라는뜻이아니다. 모든positive-only/이상탐지기법을이ET실험으로불가능하다고일반화하지말것.

검토범위: 비평가는학습코드와receipt 및저장추론을독립검산했으며모델을새로fit/저장한것은아니다. 실제1fold/1seed실행이고26fold전체는label구조만확인했다. 나머지2seed완료후전체3seed와held일/일반일분모를같은checker로검증하면충분하다. 수치회귀감사중단과40성능재검증0을유지한다.
