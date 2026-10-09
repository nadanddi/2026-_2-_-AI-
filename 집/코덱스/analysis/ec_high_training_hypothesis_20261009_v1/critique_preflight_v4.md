# 독립 source preflight v4 — 등록 JSON 정규화
2026-10-09 · run_v3.py → run_v4.py diff 검토

판정: **실행 승인, 새 차단 문제 없음.** 후보 fit 전에 발생한 Python tuple/JSON list 구조 차이 수정이다.

확인: reg==old를 json.loads(json.dumps(reg,ensure_ascii=False,allow_nan=False))==old로 바꿨다. sorted tuple 삭제집합은 JSON 저장 시 list로 변환되므로 Python 원객체와 JSON 로드 객체의 직접 비교는 값이 같아도 실패한다. 양측을 JSON형태로 정규화하면 저장된 동일값 계약을 검사한다. dict 키/순서·숫자값을 느슨하게 비교하거나 정답/행/특징/seed/통계 차이를 무시하는 코드는 추가되지 않았다. allow_nan=False도 유지된다.

나머지는 registration/replay/preflight 이름을 v4로 변경한 것뿐. v1~v3 검토의 누수·캐시/메타 대체증거·조건부 ET개입·통계 및 주장 제한은 그대로 적용. 검산 v4도 새 등록/재현명과 independent_v4 output만 변경한다. 첫 baseline 재현과 최종 cached baseline maxdiff PASS가 여전히 필수.
