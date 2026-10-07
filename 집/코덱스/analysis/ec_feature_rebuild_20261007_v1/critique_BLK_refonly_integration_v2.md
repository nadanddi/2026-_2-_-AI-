# BLK 통합 IN01 보완 독립 재검토

2026-10-07. `verify_BLK_baseline_v4.py`와 `blk_score_v6.py` 소스만 읽었다. 모델/코드 실행·보류 정답 열람·채점·GPU 사용 0. 실제 cached context8 또는 assembly/gate 완료 여부는 확인하지 않았다.

## IN01 판정: 소스상 핵심 결함 닫힘, 실제 gate 실행 대기

gate v4의 transitive_paths는 method registration, R3/PFN registration, assembly audit/predictions, cached receipt, layout/context, weight를 직접 넣고, method_code 목록·R3 dependencies/sources·PFN library/runtime source·13member·assembler source·cached 감사/출력/complete·verifier lineage를 합친다. resolve한 절대경로와 현재 SHA를 저장하고 >=40개의 양성 범위를 강제한다.

scorer v6는 보류 정답 CSV의 값 변환 전에 해당 전체 dictionary의 모든 경로가 절대경로이며 현재 SHA와 일치함을 검사한다. 주요7경로 membership도 검사한다. 기존 gate/code/receipt/등록/예측 및6variant 고정 규칙 검사도 유지한다. 따라서 이전 IN01에서 지적한 gate 생성 이후 method 코드/R3 등록·원자료/PFN library·weight/layout/context/assembly audit 변경이 채점 직전 검증을 빠져나가던 핵심 틈은 소스상 닫혔다.

이는 새 gate를 실행해 유효한 transitive receipt를 실제 만든 상태와는 다르다. 4cached context 실제 PASS→receipt v2 fresh PASS→assembly v4→gate v4→scorer v6의 순서를 지켜야 한다. 기존 gate/scorer를 덮어쓰거나 옛 raw를 대체하는 경로는 없다. 본 재검토에서 채점을 막아야 할 새로운 명백한 입력 누수는 발견하지 못했다.

## 남는 좁은 기록 한계

`BLK_R3_full_future_audit_v1.json`이 참조하는 `blk_r3_future_audit_v1.py`는 gate 생성 시 code SHA를 검사하지만 transitive 집합에 명시적으로 추가되지 않는다(method 등록·dependencies 다른 목록에 포함됐다면 간접 포함). 이 audit producer도 집합에 명시하면 '모든 감사 생성 소스 현재 SHA'라는 표현이 정확해진다. scorer가 읽어 비교하는 이전 method registration v3와 기록하는 parent scorer source도 재현 기록용으로 포함하면 좋다. 이 항목들은 현 고정 예측의 계산에 새로 실행되는 코드가 아니며 이전 IN01 핵심 차단 결함이 다시 열렸다는 판정은 아니다.

>=40과7필수경로는 정확한 전체 경로집합을 독립 재구성하는 검사는 아니지만, 고정 gate v4 source가 등록 dictionary들을 합치는 경로를 검토했다. 의도적 receipt/source 동시 위조를 방어하는 서명 체계로 확대 해석하지 않는다. SHA 검사 직후 동시 파일 편집을 피하고 실행 소스와 자료를 immutable하게 유지해야 한다.

## 기존 판정 범위 유지

조합 순서·shrink1회·clip/SG2/endpoint prefix·guard fallback·전체1440행 손실 및6variant 고정 통계에 대한 이전 source 판단은 변하지 않는다. cached 정책은 uncached query-stat 처리를 수정한 새 baseline이며 이전6.373/374와 numerical FAIL은 보존된다. BLK6개 진단 선별은 전체196후보 탐색, 원 TM/P2LOO/EL1/DIAG 검증기, 미사용 seed/layout 최초1회 확정을 대체하지 않는다. 실제4문맥/gate 및 성능 결과에 관한 결론은 유보한다.
