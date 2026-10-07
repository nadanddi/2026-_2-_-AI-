# CH2 prefix 경계 보완 독립 재비평

2026-10-07. `ch2_prefix_sources_v3.py`, `audit_ch2_prefix_synthetic_v2.py`, 실제 `CH2_prefix_synthetic_audit_v2.json`을 source/receipt로 검토했다. 모델·합성 코드·competition inference·채점은 실행하지 않았다.

## CP01/CP02 판정: 닫힘

**CP01 same-block 접근 경계:** `_packet`이 현재 block의 expected ID 집합을 먼저 만들고 `set(prefix)==expected`를 검사한다. 이후 모든 행의 query membership, block index, farm, 시각, RAW-only schema를 검사한 뒤 별도 두 번째 loop에서 float 변환한다. 따라서 다른 block 과거 행이나 다른 farm/future/gap 행은 숫자 변환 전에 거부된다. 값 변환과 identity/schema 확인이 행별로 섞이는 경로도 제거됐다.

**CP02 complete prefix:** expected는 해당 block의 모든 이전 query day를 각각24행 포함하고, 현재 day는0..h를 포함한다. 행 누락과 추가가 모두 거부된다. 원자료 값 결측은 완전한 행의 None으로 계속 허용되므로 행 누락과 혼동하지 않는다. 세 방법 모두 같은 packet 규칙을 통과해야 하며 current-only 방법도 완전 packet을 받되 ranking에서는 현재 day만 쓴다.

실제 저장48checks와 check_count48가 일치한다. 새12개는 세 방법 각각 complete prefix 수용1개, 이전 day 중간행 누락 거부1개, 현재 hour 누락 거부1개, 같은 farm의 이전 다른 block 거부1개다. 거부 packet에 `ExplodesOnFloat`를 사용하여 AssertionError로 거부되는 동안 숫자 해석이 발생하지 않았는지 검사한다. 이 추가 사례는 기존36개에서 빠졌던 CP01/02의 경계를 직접 겨냥한다.

receipt의 두 source SHA256을 현재 파일과 별도 PowerShell `Get-FileHash`로 다시 비교했고 mismatch0이다. v2→v3 source 텍스트 비교에서 constructor까지와 `_distance` 이후 ranking/choose/predict 전체는 동일하다. 따라서 이번 경계 수정과 함께 비용식·순환 component fallback·불변 scale·고정 .2 blend가 바뀌었다는 근거는 없다. 기존 파일은 보존됐다.

## 남은 범위

새 core packet blocker는 발견하지 못했다. 다만 48은 저장 검사 항목 수이며 세 immutability 항목 이름은 여전히 같아 서로 다른 이름의48가설이라는 표현은 부적절하다. source로 전체 schema 선검사 경로는 확인했지만, 유효한 첫 행에 float sentinel을 놓고 마지막 행의 schema만 잘못된 별도 사례는 저장 검사에 없다. 현재 구현 수용을 막는 문제는 아니며 향후 경계 회귀 검사에 추가할 수 있다.

합성용 `complete_packet`은 사람이 만든 fixture의 누락 관측을 복사해서 채운다. 이는 테스트 데이터 생성 함수이고 실제 runner가 관측되지 않은 입력을 복사·합성해 packet을 완성해도 된다는 허가가 아니다. 실제 자료에서는 허용된 원행과 그 원결측을 사용해야 한다.

앞서 기록한 N01 입력 거리 의미·N02 recordhour 보간 가정은 유지한다. N03의 실제 train-only scale/graph 생성 provenance·loader 소비 ID 감사, N04의 후보 등록·조합 순서·검증기·seed·비교 수·최초 미사용 판정은 아직 완료되지 않았다. 실제 competition inference/fit/score0 및 `adoption_permitted=False` 표시는 타당하다. 이번 닫힘은 CP01/02와 합성 기능 경계에 한정되며, 실제 인과 gate·성능 PASS·바닐라 데이터 전수 완결·차단된 도메인 실행의 우회 근거가 아니다.
