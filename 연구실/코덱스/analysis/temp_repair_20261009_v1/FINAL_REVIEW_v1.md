# 온도 CO₂ 구성원 진단 최종 독립 비평

2026-10-09 · 연구실 · 코덱스 critic. results_v1/보고서_v1/feature_audit_v2/independent_verify_v3 및 verify_v3 코드를읽고, 별도무학습검산으로12개fold NPZ/metadata/registration/공식sub_temp를전수확인했다. EC수치정답은pandas usecols에서제외, 외부자료0, 새fit0.

## 판정

**두후보모두사전진행기준미달이며전체W40G-S검증에진입하지않는판정이정확하다.** DROP5는DIAG−.044617%/EXT10−.732634%/EXT12−2.609643%로평균방향은모두개선이나DIAGp_worse.48195/CI상단+.008432℃이다. MED3는DIAG−.497192%/EXT10+.830100%/EXT12−1.515736%,DIAGp.0226>.0125/CI상단+.000442℃이고EXT10방향도실패한다. 유의한전체온도개선/CO₂오류원인/전체혼합효과없음의판정이아니다. 기준/후보모두새foldweights·foldseason을사용한CODEX구성원screen이다.

## critic이 직접 실제 검산한 범위

- run_v2.source SHA=등록SHA,정렬등록JSON recipeSHA=모든foldmetadata 및결과recipeSHA. 12개fold파일SHA와metadataqueryIDs,train/query불교집합,query등록foldday,train동일farmquery±1일purge,모든seasonanchor훈련key소속을확인했다.
- 각fold세예측배열shape·유한값/검증ID유일성을확인했다. metadatafeaturemask셀수=trainrows×89×2,weightscounts=trainrows도전수일치했다. 이featuremask항목은root실제assert통과receipt검토이며critic이feature값을독립재생성한것은아니다.
- DIAG9600/EXT10 2040/EXT12 5256행의원sub_temp를usecols로읽어표준math.fsum 제곱합RMSE를별도계산했다. 9개RMSE의root값과차이는0. 두후보전검증기방향·alpha/CI조합screenFAIL을별도평가해결과JSON과같음을확인했다. 원패키지/input/season51+3파일SHA도보존됨을다시확인했다.
- 20k bootstrap·그룹통계4997건은**root의독립산술verify_v3 실행receipt 및구현검토**를확인했다. critic이별도20k를다시실행한것은아니다. verify_v3은NPZ출력+공식sub_temp+등록/metadata에서RMSE/purge/anchor/블록SSE와count/중앙97.5CI/p/그룹/판정을계산하고actualdataSHA를원재현Pin에대조한다. 이수치검산은독립모델재학습검증이아니다.

## 품질·보고서 범위 비평

feature_audit_v2의18,878,880셀은여러교란반복의비교셀합이다. 서로다른독립관측1888만개가아니며모델예측규정검사와도다르다. 미래T한점/farm격리/MASK/alltest/5열변경영역/shuffle/foldweights검사의관측결과다. 두cut과선정교란을검사했으므로모든가능한입력변형을증명했다고말하지않는다. 원weights1=7259/.2=2318/.04계열23 총9600으로문서보완은유효하다.

36회는baseline/drop/med×12fold의**CODEX구성원pipeline학습**이며각pipeline내Ridge+LGB를포함한다. 최초3+후속33이같은등록을사용했고기존TK/CUDA원OOF를새기준으로대체하지않았다. 같은seed여러번학습을독립강건성근거로세지않은설명은정확하다.

DROP5의EXT12개선만보고새채택문턱을선택하거나MED3p.0226을일반.025에비교해통과로바꾸면안된다. 현후보2개familyalpha.0125가사전고정됐다. 반복DIAG탐색이므로독립신선검증/잠금확인이라고하지않은보고서제약은필수다.

전문가관련보고서는멀칭/수온/센서depth를원인확정·0.8℃상수로바꾸지않으며BASE유효일사기존구현을CODEX잔차로옮기는다음가설을미실행으로구분한다. PDF2쪽직접추출이라는root범위변경은root의현재자료확인경로에대한주장이고,critic이PDF를직접독립추출했다는뜻이아니다. 이전EXPERT_REVIEW_v1의PDF미추출은그비평당시critic범위로유지한다.

보고서의결과/비채택/원모델유지/새제출0/현재정확재현불일치미해결표기는근거와맞는다. 이번screen완료를전체온도품질개선완료또는재현문제수정완료로표현하지않는다. 별도유효일사후보는사전등록·기존시도검색·다중비교·계획비평후에만진행하는새실험이다.
