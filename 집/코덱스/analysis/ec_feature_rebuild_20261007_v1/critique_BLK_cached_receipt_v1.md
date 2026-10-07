# BLK cached PFN strict receipt와 context5 독립 검토 v1

2026-10-07. `blk_cached_pfn_receipt_v1.py`는 읽기만 했다. 실행 모델/보류 정답/채점/GPU 사용 0. context5 저장 출력·감사·등록을 별도 PowerShell JSON 및 Get-FileHash로 대조했다. 모델 수치를 새로 예측한 독립 재현은 아니다.

## context5 저장 증거 교차확인

| 항목 | 독립 대조 |
| --- | --- |
| audit status / 18개 검사 최대 차이 | PASS / 0.0, 정확한 18키 차집합0, 비영값0 |
| trace | fit4 / predict104, fit2000행·2000라벨·cacheFalse 오류0, predict0라벨·cacheTrue 오류0 |
| mask/group fit | 8 / 4 |
| cache 전후 내용 | 일치, 4캐시 모두 KV12 layer 기록 |
| 예측 저장 길이 | 1440 |
| prediction file / runner / registration SHA | audit와 모두 일치 |
| library / 실제 runtime module source SHA | 7 / 7개 모두 불일치0 |
| 기록 실행 시간 | 308.9379999998491초 |
| 등록환경 | Python3.12.10 / Torch2.14.0+cpu / TabPFN9.0.0 |
| 기록 runtime | ExplicitKVCache / TabPFNV2 / fit_with_cache / auto cache precision / torch.float32 / thread4·interop1 |

부모가 작성한 `BLK_cached_context5_crosscheck_v1.json`의 핵심 항목을 독립 재확인했다. 기존 파일을 고치지 않았다. 확인 시 complete.json 및 BLK_cached_PFN_receipt_v1.json은 존재하지 않았다. 나머지 문맥 및 전체 PFN receipt는 완료로 판정하지 않는다.

## strict helper의 소스상 판단

이 helper는 complete 문자열만 믿지 않는다. R3/runner/adapter/source/weight pin, query 및 2000 context ID, 고정1e-6·rtol0·정확18키·finite/nonnegative 값, trace 각 항목, 원래 등록 shape와 matrix SHA 재계산, runtime 설정, 4cache 전후 기록을 다시 확인한다. 이후 4개 prediction+audit와 registration+complete의 실제 파일 SHA를 receipt로 묶는다. RCV302 핵심 지적은 이 검증이 실제 실행 PASS하고 새 gate가 이 receipt에 연결된다는 조건으로 닫을 수 있다.

CPU-only torch wheel(version의 +cpu 및 torch.version.cuda is None), 명시 CPU 모델 설정은 GPU0에 대한 별도 강한 근거다. helper는 실행 시 두 조건을 assert한다. 이 리뷰는 torch를 import하여 cuda build를 직접 확인하지 않았으며 현재는 등록 기록과 소스만 확인했다. `.cpu()` 이전 원 tensor device가 기록되지 않은 RCV301 한계는 helper limits에 정확히 남아 있다. 이를 이유로 이미 진행 중인 CPU fit을 중단해야 한다고 판단하지 않는다.

## 남은 제한 및 작은 강화 권고

1. library_sha256/runtime_module_sources는 있는 항목만 검사하여 빈 dict·누락항목에 대한 양성 검사가 없다. 정상 producer의 현재 등록은 독립 확인한 7개로 충분하다. 새 gate에서 정확7개 module/상대파일 키와 path==ROOT/pinnedfile을 강제하면 strict verifier 자체의 범위가 명확해진다. 실제 현재 import __file__ 역시 저장 기록과 별도로 확인할 수 있다.
2. KV 기록은 nonempty만 강제한다. context5는 12 layer 모두 있었음을 독립 확인했다. 기대 architecture config layer 개수/정확키와 train_shape 전체값, hash record shape의 양의 정수 여부를 확인하면 구조 변조·불완전 snapshot도 차단한다. 기존 모델 재fit을 요구하지 않는다.
3. cache SHA는 producer가 실제 tensor에서 만든 snapshot 기록의 전후 일치다. helper는 저장된 hash 형식/동등성만 확인하며 실제 메모리 tensor를 재생성하지 않는다. 이를 독립 모델 재현 또는 전체 input domain 증명으로 확대하면 안 된다.
4. helper receipt에는 자신의 code SHA가 포함된다. 새 assembly/gate가 receipt 파일 SHA와 helper 현재 code SHA를 실제로 검사하고 전체4문맥 명단을 강제해야 한다. write_text는 최종 JSON을 atomic 저장하지 않아 중간 종료시 불완전 파일이 남을 수 있다. 불완전 JSON은 gate에서 실패하게 하고 기존 파일을 덮어쓰지 않는 정책을 유지한다.

새로운 명백한 보류 정답 누수나 후보 채택을 허용하는 우회는 발견하지 못했다. source+현재 context5 기록의 통과 범위와 4문맥/전체 후처리 gate 범위는 구별되어 있다. 전체 목표·기존 검증기·후보 다중비교·최초 확정1회는 이 receipt가 대체하지 않는다. 정답 채점 및 개선 판단은 아직 허용하지 않는다.
