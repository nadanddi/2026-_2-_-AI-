# CH2 등록3·context2·verifier2 독립 재검토

2026-10-07. source와 실제 등록/합성 receipt 및 현재 파일 hash만 검토했다. competition query·model·score·verifier를 실행하지 않았다.

## 등록 보완 수용

CR01/CR02는 닫혔다. runner3은 Python3.12와 -O 금지를 확인하고 실제 context2/base_context/reference_loader/prefix module.__file__를 own folder 절대경로와 비교한다. 이 파일들은 등록의 현재 source pins에 포함된다. context2는 expected ID 전체의 존재·block/farm/time·RAW-only schema를 선검사하고 별도 loop에서만 strip/float를 호출한다.

등록3의94개 현재 SHA mismatch0을 독립 PowerShell로 확인했다. 후보 v8은202행/202고유 ID다. 기존 v7의 처음196행이 v8과 정확히 같다. 새6행은 등록3을 참조하는 새 버전이다. 기존 CSV·등록·runner는 보존됐다.

실제 context 합성 receipt의3개 source SHA mismatch0, 저장 count8과 checklist8이 일치한다. 마지막행 badschema/missingrow를 앞선 유효행의 ExplodesOnStrip보다 먼저 거부하는 사례는 CR02를 직접 검사한다. 서로 다른 farm/같은farm 다른block/앞뒤query 전체 forbidden값 sentinel, blank→None 및 nonfinite 거부도 포함된다. constructor를 호출하지 않는 synthetic fixture이므로 competition CSV loader 전체의 실제 검증으로 확대할 수 없다.

## verifier2 준비 검토

아직 미실행인 verifier2는 기존 baseline gate fresh 검사, 등록/current94pins, pred/audit/runner/registration SHA, reference fresh replay를 연결한다.1440 unique ID 및 정확6scope-seed×3method 출력집합·길이·finite를 강제한다.360boundary pair 집합·개수·True 확인은 중복으로 누락 pair를 채우는 우회를 막는다.

소비 로그는 원 자료 사용 기록의 expected ID를 독립적으로 다시 구성해 검사하며 각 rid의 호출 수를 probe2회/나머지1회로 정확히 제한한다. 따라서 예상1560회 로그가 모두 허용 same-block prefix여야 한다. fresh context가1440packet을 다시 만들고 prefix SHA를 확인하는 검사는 저장 로그 문자열만 믿는 검사보다 강하다.

선택/diagnostic은 같은 choose 구현의 재실행이다. **선택 거리·component ranking의 독립 구현 검산**은 아니며 그 표현을 쓰면 안 된다. endpoint 산술은 다른 표현인 양끝 거리 dl/dr로1440×3×2×3=25920개를 계산하여 독립 대조한다. fallback이면 baseline bit equality, 활성일 때 .8/.2·clip·finite와1e-12 미만 오차를 검사한다. 이것은 정답 없는 조합 검산이며 실제 성능 측정이 아니다.

## verifier/gate 전에 남은 연결 조건

- 현재94pins에는 `verify_ch2_BLK_v2.py`가 없다. inference 실행을 막는 후보 정책 결함은 아니지만, gate/score 파이프라인 source를 별도 등록으로 봉인하여 verifier의 판정 기준도 실행 전 고정해야 한다. verifier가 출력한 자기 SHA만으로 사전등록을 대신하지 않는다. scorer는 gate/등록·pred/audit·verifier/currentsource를 truth read 전에 검증해야 한다.
- verifier2는 runner main을 호출하지 않으므로 runner main의 Python3.12/base_context 실제 경로 검사를 자동 실행하지 않는다. verifier에서도 같은 runtime/base_context 경계를 직접 확인하거나 실제 runner receipt의 runtime 증거를 확인하는 것이 명확하다. 현재 다른 runtime/import가 사용됐다는 증거는 아니다.
- 현재 gate는 audit의360sample boundary flag와 fresh 전수prefix 재생을 확인한다. 모든1440×18 최종 예측을 미래 교란한 감사라는 표현은 여전히 부적절하다. source와 sample+fullprefix/산술의 결합 검증이라는 실제 범위를 유지한다.

새 core inference blocker는 발견하지 못했다. 등록의 누적60/.025/60·고정3방법·.2blend·원검증기/모든seed 의무·미사용확정 판정 미대체는 유지한다. actual inference/gate/score0이므로 진단 채점이나 개선 보고를 허용하지 않는다. 이 정적 검토는 차단된 domain assembly를 대신 실행할 근거가 아니다.
