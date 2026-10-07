# CH2 실제 gate·직렬화 수정 독립 검토

2026-10-07. 실제 gate3/등록3/새 score등록2 및 원·수정 verifier/scorer source를 읽고 현재 hash를 비교했다. 모델·query·정답·score·checker를 실행하지 않았다.

## 판정

현재 실제 gate의 제한된 BLK 진단 채점 경로를 막는 core 오류는 발견하지 못했다. 이것은 score 향상이나 채택 판정이 아니다. 원validators/모든seed/최초 미사용 확인/전체 goal은 미완료다.

현재 registration94개 pin이 gate98개 pin에 모두 같은 값으로 포함되며 gate98개 SHA mismatch0이다. predictions 현재 SHA와 gate 기록이 일치하고 새 scorer2/verify3 SHA도 score등록2와 각각 일치한다. 저장 gate는25920 독립 scalar check·최대차이1.1102230246251565e-16·1440 freshprefix·1560소비log·360sample boundary를 기록한다. 원 source와 등록된 실행 모집단에 연결되는 실제 receipt이며, 독립 작업에서 이25920 숫자를 재실행하지는 않았다.

## 수정을 수용하는 근거

원 verifier2→3의 텍스트 diff는 `diag==저장diag`를 `json.loads(json.dumps(diag,allow_nan=False))==저장diag`로 바꾼 부분과 새 gate 출력경로뿐이다. 메모리 component tuple은 JSON 배열/list로 저장되므로 JSON 표현에 맞춘 비교는 정확한 직렬화 동등성 검사다. 수치 tolerance·선택·anchor·baseline·출력·감사 조건을 완화하는 변경은 없다. first mismatch 진단 파일의 normalizedexact 설명과 부합한다. 원 실패를 새로운 PASS로 덮어쓰지 않고 실패/원 verifier를 보존한 처리도 타당하다.

scorer1→2 diff는 gate3/verifier3/score등록2의 참조 포인터뿐이다. 기존 score spec의 scorer/verifier SHA·created_before_inference를 제외한 공통 필드를 비교했고 통계 차이0이다. 새 spec는 `created_before_inference=False`와 serialization-only repair 범위를 명시한다. ‘새 spec도 inference 전에 만들었다’는 설명은 틀리므로 원 spec1이 사전 봉인됐고 spec2는 truth 채점 전 pointer repair라는 실제 순서를 유지해야 한다. 원 print 문구보다 이 receipt의 명시 필드를 우선한다.

prediction 파일을 고치거나 inference를 새 weight/문턱으로 재실행한 수정이 아니라, 이미 생성된 prediction의 JSON 표현에 맞춰 검증한다. topology coverage가 관측된 뒤 변경됐지만 label·성능을 사용한 방법 선택 또는 통계 규칙 변경의 근거는 없다.

## 남은 제한과 score 조건

gate의 supported_rows(0/25/14)는 적용 범위 진단이다. source assignment 정확도·개선 효과가 아니며 guard0을 CH2 전체 실패로 부를 수 없다.25/14행 희소 적용은 향후 점수의 작은 변화·block 의존성과 함께 설명해야 하고 coverage를 보고 문턱을 바꾸면 별도 새 탐색이다.

actual gate는 sample360 미래/다른block 교란+전수prefix·로그재생+25920 독립 endpoint 산술이다. 전수최종출력 미래교란 또는 choose거리의 독립 구현 증명으로 표현하지 않는다. 동일 CPU cached baseline의 BLK/pass1 한계를 유지하며 historicalGPU/submission14 동등성을 주장하지 않는다.

scorer2는 기존 gate/currentsource/spec/pred/원자료SHA 검사를 truthread 전에 그대로 수행한다. 실제 scorer가 이를 통과하고 독립 Decimal checker3 및 사후 결과 비평을 마치기 전에는 점수 결론을 보고하지 않는다. 고EC/위치/hour 기술 구간에서 유리한 부분만 새 정책으로 고르지 않는다.

checker3 source에서는 정확6variant 및576cell key집합과6개 평균 delta 독립 검산이 추가돼 직전 중복/요약 지적을 보완했다. 아직 미실행이므로1152RMSE/48SSE/6bootstrap PASS라고 쓰지 않는다. 기존 Decimal 분할·runtime/base_context 엄밀 증거 범위 권고는 유지하지만 현재 gate의 source·산술 결과를 무효화할 새 오류는 찾지 못했다.

이 검토는 blocked domain assembly의 우회 실행이나 원자료 바닐라 전수 완결을 허용하지 않는다. 허용 범위는 새 고정6 CH2 BLK 진단의 기존 gate를 통한 제한된 채점이며 adoption은 여전히 False다.
