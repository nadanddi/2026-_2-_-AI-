# N4 v2 실행 전 후속 비평

2026-10-10 연구실 코덱스. `N4_PLAN_v2.md·n4_run_v2.py`를 실제 읽고 v1→v2 전체diff를 확인했다. critic fit/predict/정답접근0. audit 실행 완료를 직접 확인한 것은 아니다.

**판정: v1의 첫fold 재개 제어 문제는 해소됐고 새 핵심 코드 차단점은 발견하지 못했다. 같은 v2의 실제 audit 정상종료와 등록 recipe SHA 일치를 부모가 확인한 뒤 최초 `--first` 진행 가능하다. 첫fold 뒤 독립 중간검산 전 전체fold 진행은 허용한 것이 아니다.**

- `--first`는FD[:1]만순회하며cached첫fold에서도전체집계전에종료한다. 재호출이둘째fold를새학습하던v1문제가닫혔다.
- 변경은PerformanceWarning억제/등록·audit·result새v2이름/first제어뿐이다. 모델·gate·feature·split·bootstrap계산식 변경은diff에서없었다. 경고억제는성능경고에한정돼실제예외를숨기는변경은아니다.
- 계획v2가pooled farm×rawday//5 block bootstrap의비층화정의,seed726/9109매fold no-op exact검사,전체혼합차이식과단일PFN49예비한계를명시했다. 결과기준/threshold를사후수정하지않았다.
- actualaudit의범위는여전히첫DIAGfold/각농장첫querydayh8 cut의feature union 변조검사다. 전체시간경계·season/weights·실제PFN predict의인과성/배치독립성완료를뜻하지않는다. 이범위제한보고약속을유지한다.
- recipe에는runtime/source/data/checkpointSHA가있고원package전파일은시작시P의해시표와비교한다. 다만원package해시표자체가recipe안에복제된것은아니다. 원preparation P파일SHA 또는그해시표digest를첫fold독립감사에서함께보존하면전이적출처설명이더정확하다. 현재top-level원파일검사와모든의존성의cache pin완전성을같은말로설명하지않는다.
- v1/v2의local산출경로는같다. v1모델fit0상태가맞다면충돌은없지만,부모는실제파일/등록상태를확인해야한다. cached예측은v2 recipeSHA와맞아야재사용하며 mismatch때덮어쓰지않는다.

신규layout은기존정답을다시사용하고IRR4실패후gate발상한6번째후보다. singlePFN49+BASE3seed평균이원9회차full8/여러실제seed일관성을증명하지않는다. 계획·첫결과·최종비평및별도full확인전에채택/제출0상태를유지한다.
