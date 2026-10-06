# run_v4 사전 보완 확인

2026-10-07 연구실 코덱스. run_v4.py/preparation_v4.json을검토했다. sourceSHA74b9a42f13fc95f73851b9dc47bf27b97d6ae59f6cecfc23cbd7a68ccf591b96. reviewer모델fit0.

**사전 실행 차단 결함은 발견하지 않았다.** 원보존6ET재현·후보/보정/채택0 계획은유지된다. 최종결과PASS는원캐시재현과독립전체tree감사이후판정한다.

-old모듈stage1_v2 및그preparation_v2.json 참조가복원됐다. 자기출력만preparation_v4/results_v4로명명됐다. v3의FileNotFound는전역버전문자열치환으로의존파일버전까지바꾼준비실패이며모델fit이나원결과변경은아니다. 실패원파일을보존했다.
-weighted_co2/heating및h0요약이실제imputer/float32 X의해당열을사용하고현재raw결측의지원weight를별도로표시한다. 이전raw.fillna0 설명문제는해결됐다. h0결측지원까지해석할경우current결측율과동일한값이라고가정하지않는다.
-first_split에parent_mean_ec/high_row_fraction/high_day_fraction/days/n이추가돼사전프로토콜의parent/양child지원설명을충족한다. 저장fulltree로parent평균과앞선공통경로를독립대조할수있다.
-common3seed/purged1은각96행,actual0heldout48행,actual1heldout24행으로총456행이다. train4632/6432/6288/6552와train/queryhash는이전구성범위와정합한다. actual모델에캐시검증행밖의훈련날예측을OOF처럼끼우지않는다.
-예측조건·leaf가중합·weightShapley규칙·원모델/seed/특징/정답은바뀌지않았다. 가드1e-10,bootstrapfalse,squared_error,600tree·weighted/n-count조건을원파일에유지했다.

이미기록한해석제약은유지된다:leaf지원은고정ET의계산원천이며학습행삭제효과/물리인과가아니다. firstchild는최종leaf가아니다. signed지원변화와raw/centered기여,고ECrow/day,seed/tree반복의분모를구별한다. PFN/전체ensemble/현EC14는추적대상이아니다.

작업절차개선:앞으로자기output버전과읽기전용의존version을별도상수로구분하면전역문자열치환오류를줄일수있다. 현재는--prepare의의존존재/hash가드가fit전에오류를막았고v4에서복원되어,등록완료후직렬재현을진행할수있다.
