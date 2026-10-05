# 실제 A 중첩 OOF v4 최종 사전 독립 비평

2026-10-06 · 집 코덱스 독립 비평 담당. run_v4.py, verify_v4.py 및 준비 완료된 preparation_v4.json을 읽었다. 실제 학습·source 수정은 하지 않았다.

## 판정

**명백한 잔여 실행 차단 결함은 발견하지 않았다. 사전검토 차원의 실행 허용 의견이다.** 기존 지적의 핵심 문제는 코드에 반영됐다. source·준비 해시 등록이 완료된 뒤 그대로 실행하며 첫 실제 재적합 감사 실패 시 중단해야 한다. 이 판정은 실제80문맥 적합 성공, 결과 검산 성공, 성능 개선 또는 규정 원문 적합을 증명하지 않는다.

## 모델·분할 불변성 재확인

AST 독립 파싱에서 v1과 v4의 models/r3/pfn/splits/prep/runtime/ordered/ids/ar 함수가 정확히 동일했다. preparation_v1과 preparation_v4의 records/runtime/inputs/full/base/core_params/seeds/pfn_seeds/pfn_config/threads를 재대조해 모두 일치했다. 외부20/내부80의 ID와 순서, target/feature SHA, season notes 및 clip bounds도 records 완전 일치에 포함된다. 이전 보고서의 예정370,800시간행·61,800사례행 출현 수는 유지되며 고유 독립 표본 수가 아니다.

## 문제별 최종조치 피드백

| 지적 | v4의 최종조치 | 독립 확인 의견 |
|---|---|---|
| 정답/flags 자기검산 | 등록 query target SHA와 원 공개 OOF y 대조 유지 | 적절 |
| NaN 누락 | CSV 핵심 실수열 finite 전수검사 유지 | 적절 |
| 정상 재검산/최종화 중단 | staging→promotion, compare-or-create, check-only 유지 | 적절 |
| 시드 합의 산출물 | seed consensus CSV 생성과 세 시드/투표/범위 독립 재계산 유지 | 적절 |
| raw_r3 부수열 누락 | CSV raw_r3와 해당 구성원 NPZ raw 직접 대조를 gaps.raw_R3에 추가 | 지적 수용 |
| 행 ID/문맥 누락 | row_id 파싱 farm/day/hour 및 파일 문맥 v/k/s 직접 대조 추가 | 지적 수용 |
| 검산 현재 runtime 미검사 | 실제 버전·5개 module-init SHA·TabPFN regressor module SHA를 prep와 대조 | 지적 수용 |
| check-only ZIP 미검사 | bundle 지정 ZIP SHA 대조 및 CRC testzip 추가 | 핵심 지적 수용 |

첫 원 A 입력 source 동일성과 v1/v4 모델·분할 불변성을 확인했으므로 위 검산 보완을 성능 튜닝이나 새 모델 변형으로 해석할 근거는 없다. 자동 검산 문서는 사용자 확정 결과와 구분하며, 최종 전달 전에 독립 결과 비평을 요구하는 문구를 유지한다.

## 남은 범위 한계 — 실행 차단 아님

- check-only는 ZIP 전체 SHA와 CRC를 확인하지만 원 finalize처럼 entry별 현재 파일 SHA까지 다시 비교하지 않는다. 최초 정상 ZIP 생성/기존 ZIP 검산은 별도 경로에서 entry 집합/CRC를 확인하고, 기존 ZIP 재사용 시 entry SHA를 확인한다. fresh 산술 검산과 bundle 전체 SHA라는 실제 감사 범위로 표현한다.
- receipt.source_sha/count/file-list의 정확한 예상 집합·중복 금지, 부수 train_raw 및 구성원 배열 전체의 명시 shape/finite 전수검사는 아직 별도 강화항목이다. 필수560구성원/60OOF 열기와 raw·후처리·public y·합의 독립검산이 핵심 오류를 방어하므로 이번 학습의 차단 사유로 판단하지 않았다. 완료 독립 결과 감사에서 이 범위를 보충할 수 있다.
- 현재 runtime 검사에 checkpoint 실파일 SHA 재대조는 추가되지 않았다. 실제 run의 prep 재계산은 checkpoint SHA를 검증하고 metadata에 서명을 저장한다. checkpoint 없이 캐시만 재검산하는 경우와 실제 재적합의 재현 범위를 구분한다.
- 최초 등록 JSON·최초 fit·정상 완료·check-only 두 번의 실제 실행을 이 사전 코드 검토에서 대신 검증한 것은 아니다. 실제 수행 결과를 별도 로그와 결과로 확인해야 한다.
- prelaunch v4의 외부 참조 파일명 오류가 v5에서 수정됐다는 전달 내용은 모델 코드 변경과 분리해 다룬다. 이 독립 검토는 run_v4/verify_v4와 preparation_v4의 실제 모델·분할 불변성을 재계산했고, 오류 로그의 보존·v5의 실제 종료 PASS는 실행 담당의 사전 게이트에서 확인할 사항이다.

## 실행 및 완료 게이트

1. source/plan/preparation/검산 코드 등록과 사전 실행 PASS를 첫 fit 전에 확보한다.
2. 첫 실제 R3/PFN 재적합 및 PFN8행 배치 검산을 수행한다. 불일치 시 결과를 채택하거나 허용오차를 사후 완화하지 않는다.
3. 전체80문맥 후560구성원·60OOF·예상행 포괄성, 원 public y, 산술, seed consensus, receipt 및 ZIP을 새로 검산한다.
4. 독립 결과 비평에서 누수·학습/query 분리·성능 주장 부재·재현·다음 구별기용 fold별 짝짓기를 다시 평가하고 문제별 개선 피드백을 최종 전달한다.
