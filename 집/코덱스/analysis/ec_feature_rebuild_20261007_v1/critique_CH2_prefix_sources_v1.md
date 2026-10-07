# CH2 prefix 초안·합성 감사 독립 비평

2026-10-07. 대상: `ch2_prefix_sources_v2.py`, `audit_ch2_prefix_synthetic_v1.py`, 실제 합성 receipt 및 PF 출력 메타데이터 v2. 코드 실행·모델 fit/predict·정답 채점·GPU 없이 source를 읽고 저장 SHA만 재확인했다. 신규 CH2 세 방법은 실제 데이터 실행·등록 전이다.

## 실행 전 필수 보완

**CP01 — same-block 경계를 수치 변환 전에 강제할 것.** `_packet`(92행 이하)은 같은 farm의 과거 query라면 다른 block이어도 float 변환한다. `_rank`가 현재 block에 없는 day를 나중에 무시하므로 현재 결과에 직접 사용되지는 않지만, 선언한 ‘같은 block 앞 query만’ 접근 규칙과 불일치한다. `query_to_block[other] == query_to_block[rid]`를 float 전에 강제하고, 다른 block의 과거 행에 `ExplodesOnFloat`를 넣은 합성 거부 검사를 추가해야 한다. 기존 36검사는 farm/future/gap 거부만 확인하여 이 경계를 시험하지 않았다.

**CP02 — complete prefix 또는 희소 packet 정책을 고정할 것.** `_packet`은 현재 rid 존재만 요구한다. 현재 h>0에서 0..h 중간 행을 생략해도 선택이 가능하다. 과거 query 기록도 일부 시각만 선택적으로 전달할 수 있다. ‘현재0..h+이미 관측된 앞 query’의 전체 상태를 실험하는 계획이면 runner에서 정확 ID 집합을 강제해야 한다. 원자료 결측 값은 행을 유지하고 None으로 표현하는 것과 행 누락을 구분한다. complete prefix 및 과거 상태 정책을 등록하고, 중간 시각 생략·이전 날 일부 전달을 거부하거나 명시 fallback하는 검사를 추가한다. 합성 replay는 같은 sparse packet을 반복한 일치일 뿐 전체 prefix 완전성 검사는 아니다.

두 항목은 현재 알려진 규정·계획 경계 결함이며, 아직 실행하지 않은 후보를 막는 조건이다. 실제 미래값 사용이나 성능 누수를 이미 관측했다는 뜻은 아니다.

## N01~04 이행 상태

- **N01 부분 충족:** query 거리에는 EC와 sub_temp가 없고 RAW 키만 허용한다. 고정 weather/indoor/control 비용, 공통 reference 가용 열, 관측 term 평균, component별 side 평균과 절대 tie tolerance 1e-12가 source에 구체화됐다. 이 비용은 원 CH2의 midnight slope 비용과 다른 **동일 hour 입력 level 유사도**다. weather increment 표준편차를 level 차이의 단위로 쓴 고정 가설이며 ‘원 CH2 query cost 재현’으로 부르면 안 된다. current/history 선택은 EC residual 없이 비교된다.
- **N02 부분 충족:** union root를 연결된 component 식별자로만 쓰고, 해당 component에 cycle이 있으면 선택 결과 전체를 baseline fallback한다. 다른 acyclic component를 cycle 때문에 재라벨하지 않는다. 양쪽 flank 각3개만 검색한다. 보간은 CH2 연결 순서를 복원하지 않고 `24*recordday+hour`의 고정 가정으로 한다. 실제 시간 간격·물리적 원기록 복원으로 확대할 수 없다.
- **N03 부분 충족:** 앞선 CP01/02가 남는다. current h0에는 reference 동일 h0만 비교하며 query B1에 접근하는 식이 없다. nested mapping과 외부 입력 복사로 reference/scale를 동결하고 predict에 scale fit 경로가 없다. 단, constructor가 받은 scale이 정말 해당 fold의 train weather increments에서 계산됐는지는 이 클래스와 synthetic receipt가 입증하지 않는다. 실제 loader에서 학습 ID·scale 생성 source·matrix/graph digest를 봉인해야 한다.
- **N04 미충족:** source의 METHODS 세 개와 .8 baseline+.2 anchor, train bounds clip은 고정됐다. 기존 blend 뒤에 shrink/SG2를 다시 적용하지 않는 full runner 순서, RAW_PASS/QUERY_ROLE 범위, seed, 실제 비교 수 및 누적 본페로니, 전체 원검증기와 최초 미사용 판정은 아직 등록되지 않았다. 코드 상수는 이 전체 사전등록을 대체하지 않는다.

`PAST_QUERY_PREFIX_STATE`는 과거 query 상태를 새로 fit하는 모델이 아니라 current 선택과 입력 history 선택의 component 일치를 요구하는 guard다. history는 day별 평균을 다시 평균하므로 현재 짧은 prefix와 과거 완전 기록에 day 단위 동일 가중을 준다. 이 설계를 그대로 등록하거나 결과 전에 변경해야 한다. missing/tie/overflow/zero scale는 fallback하지만, 단일 component만 남을 때는 경쟁 margin이 없다는 선택 정책도 등록 설명에 포함해야 한다.

## 합성 PASS의 범위

저장 receipt의 source 2개 SHA를 현재 파일과 재확인했고 mismatch 0이다. 저장 check_count 36은 코드의 체크 항목 수와 맞는다. h0 anchor 산술·exact tie·전결측·overflow·query label 키 거부·future/farm/gap 값 변환 전 거부·반복/역순·immutability·cycle·zero scale·common support 사례가 포함된다. 세 immutability 검사는 저장 이름이 동일하므로 36개를 서로 다른 이름의 독립 가설 36개로 표현하지 않는다.

이 감사는 사람이 만든 graph와 scale에 대한 기능 검사다. 실제 17cycle topology, BLK packet 소비 ID, graph/scale provenance, 전체1440 인과 교란, 기존 검증기 전 fold, 모델 baseline 및 성능을 검증하지 않았다. 현재 receipt의 `adoption_permitted=False`는 적절하다.

## PF 메타데이터와 바닐라 데이터 범위

PF metadata receipt의 11개 SHA를 현재 파일과 재확인했고 mismatch 0이다. 저장 PF1은10848행/66fold, PF2는6648행/34fold로4200행이 빠졌다. 공통6648키의 공유 예측/경계 차이 최대4.440892098500626e-16은 재사용 정합성 근거이며, PF2의 추가 pfnE 효과를 입증하지 않는다. PF2 partial 허용은 검사 목적을 가용 현황으로 좁힌 것이고 v1 complete assertion 실패를 무효화하거나 완료로 바꾸지 않는다.

PF reference train purge, fold 밖 weather scaling, TabPFN query 통계/cache, runtime 및 최종 SG2 조합이 미감사이므로 metadata PASS는 causal/model gate가 아니다. 기존 PF1 일반 이웃 검색·PF2 anchor 차이와 CH2 fixed6flank+label-conditioned component는 구별되지만, 검색 기반 파생 방법이라는 점에서는 중첩된다. 계보 비교 후 별도 가설임을 등록해야 하며 공유 파일만 보고 generic neighbor 재실험이나 PF2 개선을 주장하지 않는다.

사용자의 원래 ‘바닐라 데이터 전수’는 외부·파생·합성 제외다. 공개 train EC/sub_temp, 허용 raw 입력·row_id·현재/과거 및 허용 이웃 **원기록**의 가용성 목록과, CH2 graph/거리/anchor 보간/PF 출력 같은 파생 실험 목록을 구분해야 한다. 후자는 새 바닐라 데이터 발견 개수에 넣을 수 없다. sub_temp 공개 원정답의 존재 확인과 온도 변환/모델링도 별도이다. 이번 source 검토는 원자료 전수 목록의 완전성 검사가 아니므로 ‘이제 누락 없음’이라고 답할 근거를 제공하지 않는다.

도메인 조립 실행의 approval credits 차단은 그대로 유지한다. 이 정적 감사는 우회 실행이나 실제 CH2 후보 시험의 근거가 아니다.
