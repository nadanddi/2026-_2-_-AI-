# N2 대상 v2 코드 실행 전 독립 비평

2026-10-10. n2_prior_v2/n2_target_v2 정적읽기만, criticfit0/EC수치0/외부자료0.

판정: **경로·실행순서2개를새버전에서고친뒤첫모델진행**.

## 실행 차단점: imported save의 실제 출력 폴더

runtime_v1에서import된save는runtime모듈의globalHERE(temp_repair_20261009_v1)를참조한다. n2_target에서HERE=TASK로재할당해도함수정의모듈global은바뀌지않는다. 현재regp는TASK/n2_target_registration_v1.json인데save는BASE_HOME/n2_target_registration_v1.json에써서등록검사경로와출력경로가달라진다. resume시TASK등록없음→BASE_HOME파일x충돌이발생하고results도원폴더에잘못써진다. target전용save를새코드에정의하거나regp/targetresultpath를명시open('x')로쓴다. 원runtime/이미등록소스는수정하지않는다.

## 실행순서: prior fit 전 출처 검사

`from n2_prior_v2 import MODEL`은helper末fit_prior()를바로실행해타농장prior1fit가target의actualCSV/parentruntime검사보다먼저된다. helper에서fit전에REGruntimeSHA=actualruntimeSHA 및actualCSV3SHA=REGDATA=P Pin,LOFOsourceSHA/지원등록을검사하거나target에서import앞으로검사를옮긴다. 현재helper는REGDATA=Pin만확인해실제파일변경을먼저검사하지않는다. 모델cacheSHA+sourcesame확인과actualRAW출처를구별한다.

## 나머지 정적 경로

- priorfit는F13/F47/F32제외및LOFOeligible48자료만으로동일day/hourdiff1/finiteΔlabel·Δair를사용한다. 지원row수/농장별paircount를LOFO와대조,동일30열/inf→NaN/medianempty0/scaler/Ridge100/lossweight다. targetlabel은priorpredictor에안들어가고target_prior는입력14열만받는다.
- predictor는farm/dayhour정렬후current/diff·sin/cos를만들고target은same-dayΔ를원rawinput으로추론한다. h0=0/중간invalidstepNaN/skipnaFalsecumsum로하루끝까지결측전파·다음날reset한다. 미래label/배지h0anchor를사용하지않는다.
- 원89+physics19동등·1추가열·globalTfuture/otherfarm/shuffle/MASK/full24h 및fold훈련외NaN세계exactassert가있다. dtype/None프로브는첫fold0에서만0.0,그외None으로metadataJSON에정상저장된다.
- parentsource/runtime/dataSHA·NPZsha/recipequeryIDs/weightcounts·seasonnotesexact·baselineprobeexact·NPZ키N2/collectorN2가일치한다. ALPHA.025/4·중앙98.75quantile/20k블록정의가유지된다. 잔류docstring 'irradiance' 및IRR변수명은가독성문제이며모델식은N2누적1열이다.

두수정보완을실제source에서확인한뒤타농장prior1fit+첫대상foldbaselineprobe/candidatefit가허용된범위다. 후속firstmetadata에서는priorschema/훈련farmkeys/전처리통계/source/data/modelSHA/등록경로/한열featureaudit/parentexact를검산한다. 코드존재만으로규정/효과통과를주장하지않는다.
