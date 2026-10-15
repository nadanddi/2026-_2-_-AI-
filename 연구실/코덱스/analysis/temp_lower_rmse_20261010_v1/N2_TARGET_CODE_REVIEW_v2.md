# N2 대상 v3 코드 재검토

2026-10-10. n2_prior_v3/n2_target_v3의수정실제소스를직접읽었다. 이전두차단점해소.

- target모듈에TASK를참조하는전용save가정의되어등록regp/결과출력경로가일치한다. 원runtime의save/globalHERE는변경하지않았다.
- helper는MODELfit_prior호출전actualruntimeSHA/실CSV3SHA/LOFOsource및집계NPZSHA/원Pin/LOFOgate/schema를assert한다. RAW는import시읽지만모델학습전확인이완료되는순서다.

원30prior·48pair지원/1누적열·CODEX물리/params/parentcache/alpha.00625/CI/NPZN2collector가변경되지않았다. **추가차단점없으며--first 진행가능**. 현재cache/실제assert결과의중간검토와대상효과판정은후속실행자료로한다. criticfit0/EC0/외부0.
