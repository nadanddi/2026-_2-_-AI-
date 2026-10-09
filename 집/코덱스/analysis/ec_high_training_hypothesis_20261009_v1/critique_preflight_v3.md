# 독립 source preflight v3 — 손상 메타 대체증거
2026-10-09 · run_v2.py → run_v3.py diff 및 원 runner_v4.py 준비코드 검토

판정: **이번 고정 PFN을 그대로 사용하는 ET 경로 진단에는 대체증거 충분, 실행 승인.** 원 PFN 재학습 재현성 전체를 새로 입증한 것은 아니다. 원 메타 복구/수정 없음.

대체 검증 근거:
- v3의 스킵 조건은 정확히 fold9/seed3이며 3422바이트 전체 NUL인 파일만 허용. 다른 파일의 새 손상은 중단된다.
- git tracked preparation_v4.json의 script_sha와 현재 원 runner_v4.py SHA가 일치함을 독립 실행으로 확인했다. 원 prepare는 PFN 40개 각각에서 train/validation 특징해시, NPZ prediction SHA, ordered row ID와 seed RNG2000 context를 assert한 뒤에만 preparation_v4를 기록한다.
- 새 prepare는 원 preparation의 PFN_sources SHA와 실제 NPZ SHA를 대조하고, 현재 fresh train/query의 FULL_R3 해시를 원 기록과 대조한다. FULL_PFN은 FULL_R3의 부분집합이다. 직접 row ID 및 RNG context도 다시 대조하므로 원 증거에 기록된 동일 NPZ/행/특징맥락의 재사용이라는 chain이 성립한다.
- 손상 메타 SHA 자체도 cachepins에 들어가므로 이후 무검토 변경/복구는 drift 중단. 첫 ET replay와 fold마다 최종 baseline 재현1e-10도 추가 보호.

한계: 손상 파일에서 provenance 문서를 직접 읽어 확인하지 못했다. 이는 이전의 정상 준비 기록과 동일 생성코드/캐시 SHA/현재 행 및 특징 동일성으로 대체한 것으로, 전체 PFN 훈련 새 재현과 같지 않다. 최종 보고에 메타1개 손상과 대체증거 방식을 명시하고 원 NPZ 로딩·finite 예측·baseline 전수 gap PASS를 확인할 것.

새 등록/재현/preflight 참조만 v3. 후보 선정/seed/가중치/통계 변경 없음. v1/v2 비평의 주장범위 제한은 유지.
