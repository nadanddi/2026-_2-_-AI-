# 원 고정 후처리 커널 독립 비평

2026-10-07. kernel/audit 소스와 저장 합성 receipt/current SHA만 읽었다. 실제 모델·SG2 선택·보류 정답·성능·GPU 실행0이다.

**순수 산술 순서에서 새 핵심 blocker는 없다.** raw등록3 recipe대로 .6×(.6ET+.3LGB+.1MLP)+.4×PFN5..8 평균, 같은farm/day raw_mix 0..h 평균과 현재값의 .5/.5 shrink한번, fold trainbounds clip, RAW_PASS SG2 callback, 마지막clip이다. h0와 day/farm 변경에 prefix를 초기화하고 day<179 identity를 사후 강제한다. 현재 receipt는 Decimal60 384stage/max4.440892098500626e-16, future/otherfarm poison10, invalid15를 기록하며 두 source SHA가 현 파일과 일치한다. 이 리뷰에서 합성 코드를 재실행하지 않았다.

## 수용한 경계

ids unique/canonical/sorted, 모든 queryday24h, 모델 vector 길이/유한 scalar·bool 거부, 정확PFN4 key 집합, 유한bounds 및 lower≤upper/callable을 검사한다. callback에는 현재day의 pre_SG2_clip0..h dict 복사만 넘기고 callback의 사후 변경을 거부한다. final 값도 유한scalar 및 bounds를 검사한다. API에 정답·원입력·모델·파일·gate가 없다.

## 남은 주장 한계와 강화 권고

- callback이 closure/global에 full query/정답/다른farm을 갖는지, 실제SG2 ref selection이 학습전용인지 커널은 알 수 없다. dict 복사 및 mutation 검사는 외부 상태 접근을 차단하지 않는다. 실제 callback source/runtime/prefix 소비와 future-input 교란 검사는 caller fullgate의 필수다.
- assemble은 전체 et/lgb/mlp/PFN vector를 시작에 finite 검사하고 차례로 모든 day를 처리한다. **현재prefix만 숫자 읽는 streaming API는 아니다.** poison10은 유한 future predictions를 바꿀 때 현재 출력 불변이라는 검사다. future NaN이면 전체호출이 reject되는 것도 설계대로이며 '미래 어떤 값도 읽지 않는다'고 확대하면 안 된다. 원 query input 경계와 raw prediction vector의 offline 산술을 구분한다.
- 현재 bounds가 그 fold train label min/max인지, 전달된 배열이 raw/unshrunk인지, context5..8의 모델/matrix lineage가 맞는지 커널은 검증하지 않는다. already-shrunk 입력을 넘기면 double shrink가 된다. caller는 producer stage 이름·원 rawreceipt/weights/rows/sourceSHA와 bounds proof를 조립 등록에 고정해야 한다.
- day<179에도 callback을 호출한 뒤 identity를 검사한다. 반환값만 보면 정책에 맞지만 skip 정책의 계산 자체가 없다는 보증은 아니다. 실제 callback의 skip 분기까지 source/consumption 감사로 확인한다. 새 threshold나 synthetic day/queryrole 변경은 허용하지 않는다.
- key집합만 비교하므로 숫자상 같은 float key5.0..8.0도 dict 조건을 만족한다. 등록된 integer context ID를 엄격히 요구하려면 type(key) is int를 추가하고 합성negative에 넣는다. 이것은 현정상등록 출력의 산술 결함 증거는 아니다.
- PFN mean은 sum, shrink는 fsum이며 Decimal/string 변환과 machine float의 미세차이를 허용한 검산이다. 숫자 동등성은 고정 허용오차 수준으로만 주장하고 bitwise sourceidentity를 주장하지 않는다.

합성audit은 stage별 독립Decimal 대조·h0/h23/day178→179/두farmreset·prefixschema·복사mutation을 포함한다. 추가권고는 callback NaN/Inf/boolean 반환, noncallable, 비문자 ID/foreignfarm, float contextkeys, day범위caller registry, callback prefix 각단계의 clip값 independent 검산 및 미래 비유한값 rejection의 명시적 범위 테스트다. syntheticSG2 +.08은 실제SG2 방식 동등성이 아니라 단계 연결 검사다.

커널과 합성PASS를 원전체 baseline gate로 사용할 수 없다. raw5346/PFN264 completion 및 엄격receipt, 원행 전수 causal·postprocess·독립 currentlineage gate가 선행하고 그 뒤 정답parse·고정 통계다. 현재source 초안을 새조립등록에 pin해야 하며 existing producer source는 그대로 보존한다.
