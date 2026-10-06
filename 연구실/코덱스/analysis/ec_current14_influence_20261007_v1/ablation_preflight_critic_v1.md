# ET 삭제 진단 코드 사전 비평

2026-10-07. `ablation_v1.py`를 읽은 사전 검토다. 새 fit, baseline 채점, 삭제안 실행은 하지 않았다. baseline 완료와 1단계 독립 감사 이후 실행한다는 순서는 유지한다.

확정적인 누수나 구성원 고정 실패는 발견하지 않았다. D1/D2에서 ET의 학습 집합만 `tt`로 바뀐다. LGB·MLP·PFN 출력과 season 변환, bounds, SG2 structure/calendar/label ref는 원 `t` 기준으로 고정된다. 삭제행 없는 fold는 baseline ET를 재사용한다. seed별 출력 및 실제 raw 평균 ensemble의 후처리 순서도 맞다. fold1/seed7 BASE 추가1fit은 전체 query 원캐시와 tol1e-10 대조한 뒤 F47_161의24시간 fullforest를 저장하므로 단순 재현용 추가 학습이다.

실행 전 명시할 한 가지가 있다. PLAN2의 판정용 “ET 하루평균편향”은 raw24시간평균인지, prefix 평활 후24시간평균인지 아직 문구만으로 확정되지 않는다. 현재 코드는 두 값을 모두 저장한다. **결과를 보기 전에 판정용 척도 하나를 고정하고 다른 값은 보조진단이라고 적어야 한다.** 두 값 중 유리한 것을 사후 선택하면 사전 고정 판정이 아니다. 이는 판정 정의의 명확화이며 모델/문턱 변경이 아니다.

구현·감사상 남은 사항은 다음과 같다.

- 삭제안 캐시는 현재 재개 기능이 없다. 기존 NPZ가 있으면 assert로 중단한다. 중간 실패 후 완료된 학습물을 재사용하려면 source/prep/NPZ SHA와 train/query/removed IDs를 검증하는 새 버전이 필요하다. 부분 산출물과 실패 로그를 보존하며 캐시를 덮어쓰지 않는다.
- BASE 재현1fit과 실제 삭제 fit 수를 분리해 기록한다. `new_ET_fits`는 삭제 fit만 세며 추가 replay-fit은 별도 항목이다.
- trace 평활 가중치는 전 행의 prefix를 계산하지만 현재 인수 `qt`는 정렬된 F47_161 단일24시간이라 정확하다. 향후 여러 날을 넘기는 재사용에는 farm/day별 prefix가 필요하다. 현재 실험의 결함은 아니다.
- 완료 후 60 arm×fold×seed 캐시의 source/prep SHA, ID·삭제행수·유한값·no-op exact equality, 69120 최종행의 유일성과 coverage를 직접 검사한다. sidecar와 receipt의 고정 숫자만으로 완료 판정하지 않는다.
- fullforest3개는 저장된 float32 train/query 경로, node counts/weighted counts/mean y, leaves·weights 및 원캐시 rawET를 재구성해 확인한다. 원본47열 순서·imputer median을 준비서명에 연결해 검증한다.

삭제 효과는 ET median imputer와 forest 전체를 다시 학습한 효과다. 고정 forest에서 특정 날이 갖던 가중치를 빼는 계산과 동일하지 않다. 최종 손실에는 SG2 gate와 clipping의 상호작용도 포함하므로 raw ET→혼합/평활/clip→SG2 delta→최종의 각 단계 변화가 필요하다. 선택 날의 고정 forest 기여나 삭제 효과를 물리 인과로 확대하지 않는다. 사전 선택된 F47_161 개선과 전체/고EC/나머지 날의 손익, 공개 pass2 지원46일의 한계를 함께 보고한다.

baseline 및 삭제 결과에 대한 수치 감사·효용 평가는 아직 미완료다.
