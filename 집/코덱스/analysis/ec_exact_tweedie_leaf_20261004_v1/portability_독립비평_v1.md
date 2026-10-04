# 한글 체크포인트 경로 보완의 독립 검토

최소 보완 수용. `verify_full_v4.py` SHA `5ea91e3b768c889a6d333ff486f14cb45d79c48d991643b8a498f828c1171f0a`를 디스크에서 확인했다. v3→v4 전체 unified diff가 `portability_repair_v1.json`의 diff와 동일했다. 변경은 세 줄뿐이다: SHA 검사된 체크포인트를 Python UTF8 read_text로 읽어 Booster(model_str) 전달, whole 출력명을v4로 변경, synthetic 출력명을v4로 변경. 모델·학습·입력·판정 공식·채택 문턱은 바뀌지 않았다.

model_file과 model_str는 저장된 동일 모델을 같은 native loader로 로드한다. 이 수용은 모델을 다시 학습하거나 보정한 것이 아니다. Unicode 파일명 접근을 Python에서 처리해 native path 접근 오류를 피한다. 이전 실패와 source/prep/prereg 핀은 보존한다. v4는 실제fit 후의 verifier portability repair이므로 v4가fit전에등록됐다고 설명하면 안 된다. 원 v3 사전 등록을 보존하고 별도 보완receipt를 결속한다.

`crosscheck_complete_v2.py` 준비: SHA `44b48bbf0bd72a4591c2cc2b02206f64c91e9b23ec0d54820c98efa6d973eff8`, AST-only PASS. v4whole 완료 gate/output 경로를 적용했고 원 등록15SHA검사를 그대로 유지했다. 추가 gate는 old/new verifier핀, 전체source가허용된3literal치환과exact동일, repair unifieddiff exact, runner/prep/prereg 불변, rules_changed=false/retrain0/input_modified=false, v3실패로그SHA다. 독립15Decimal45RMSE·3manualbootstrap·66textcheckpoint/actualcounts·83160exactaggregate 검사 계획은 유지한다.

root보고 v4synthetic13오염/40292 PASS는 root실행 증거다. 여기서는 whole 또는 candidate검산을 실행하지 않았다. 실제후보CSV읽기/점수계산/새nativepredict/fit/rawEC/test/EL1조회0이다. v4whole 완료 통보 뒤에만 crosscheck v2를 실행한다.
