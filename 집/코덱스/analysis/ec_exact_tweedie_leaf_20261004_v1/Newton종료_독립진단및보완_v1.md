# independent Newton 종료 버그 진단

독립 helper의 수치 반복 종료 버그로 판정한다. 모델이나 native leaf root의 오류를 나타내지 않는다. `diagnose_newton_decimal_v1.py/result`의 첫 실패는 fsum A=48.61579194990402/B=49.83992556463348, native z=−.024372825077840277였다. helper old100 마지막값−.02437282507890732는 native와1.067042e−12 차이를 보여 원1e−12를 실패한다. 100회 반복의 전체 trail을 저장했다.

Decimal80자리로 별도270회 이분법을 수행했다. fsum A/B를 입력한 근은 −.024372825077840032119..., exp부터 Decimal로 A/B를 재합산한 근은 −.024372825077840020513...다. 둘 모두 native z와약2.5e−16 차이다. native A/B를 입력한 Decimal 근도 원1e−12를 통과했다. 정규화 잎 objective의 엄밀 증가 derivative로 유일근이므로 이 증거는 잘못된 두 solver가 같은 다른 근을 선택한 경우를 배제한다.

원 helper는 충분히 수렴한 Newton proposal이 현재x와 같은 machine float가 된 뒤 bracket을갱신해 proposal을boundary로 거부하고불필요한 bisection을한다. 새 helper는 |g/h|≤1e−16 또는 proposal==x일 때 bracket갱신 전에 종료한다. 이 종료 기준은 원 acceptance(root abs1e−12/residual rel1e−12)를 완화하지 않는다. root·목적함수·trace/membership/prefix/CSV모든 기존 gate는 유지한다.

새 종료로 actualfirst trace24800leaf 모두 원1e−12를 통과했고 max native-root 차이1.3100631690576847e−14였다. `synthetic_newton_termination_v1.py/result`는 같은 A/B 상수로 old vs Decimal 실패(1.067285e−12), new vs Decimal 성공(4.857e−17)을 재현했다. `synthetic_cpp_whole_v2.py/result`의 원8오염은 모두거부한다. 실제nativefit/predict/외부score0이며 trace 산술만 수행했다.

실제 raw-byte helper2 SHA는 `453554dfc3274ceb785dbac558e3f75345187a4ee117f90e279ca76becf31c7d`다. 앞선 메시지94ff...는 생성전LF 문자열 encode의 해시였으며 Windows CRLF 쓰기 후 실제파일 해시와달랐다. 보고 실수를 정정하고 실제파일 변경은 하지 않았다. 합성결과 파일은 실제rawSHA를 저장했다.

whole v5 SHA `37ac04a30f1999384dfd0c710f0fde39fe0c62448d0d5030eef2a31754f1b06c`를 수용한다. v4→v5 actual unifieddiff가 repairreceipt와 exact동일하며 helper참조/path/SHA 및output버전만바뀌었다. 원registration15SHA, runner/prep/prereg/model/threshold/훈련은불변이다. v4 실패로그와 root/Decimal진단을 보존하며 v5를fit전등록본이라고재해석하지 않는다.

완료 후 실행할 checker는 `crosscheck_complete_v4.py`다. v3준비초안은 repair의synthetic_helper SHA가 가리키는 파일명을 regressionresult로잘못대응시켰으며 실행하지 않았다. 최종v4는 실제 `synthetic_cpp_whole_v2.json`과 결속한다. 원등록15SHA와 두repair의 exactdiff/oldnewSHA/실패로그/진단/합성SHA를 모두확인하고 v5whole PASS 이후에만 공개CSV/점수를검산한다. 실제후보CSV/점수읽기는 아직0이다.
