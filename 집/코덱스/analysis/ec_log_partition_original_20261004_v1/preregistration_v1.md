# log ratio 분할 + 원 EC leaf 평균 · 실행 전 고정

2026-10-04 집 코덱스, 가족수18의 새 단일안. 기존 LOG_PARTITION_MEAN은 DIAG 전체−2.9~−3.5%이나 A/B 악화·통계 미달로 기각했다. 앞선 POWER partition도 기각됐으며 이 후속을 완전히 새로운 원리라고 부르지 않는다. 같은 log ratio 분할과 원단위 평균·계절 복원 제거 조합은 확인한 기록에서 발견하지 못했다.

**변경은 실제계절v2의 ET 구성원 계산 하나다.** 외부학습만으로 기존21일 계절중앙값 b를 만들고, FULL38(day→season 마지막 열) ET600/leaf1/seed7,101,2024를 log(y/b) 목표로 적합한다. 각tree가 실제로 적용한 훈련leaf에서 **원 y의 산술평균**을 저장한다. query에는 600tree의 원unit leaf평균을 평균해 raw를 출력한다. bquery를 곱하지 않고 log 역변환도 하지 않는다. b는 train partition을 만드는 데만 사용한다. 이는 각tree의 고정상수leaf 원단위 훈련SSE 최적값이며 forest 전체/일반화/최종혼합 SSE 최적이라는 주장은 하지 않는다.

기존실제계절v2+.48×(인과shrink(newraw)−기존shrink된ET),마지막outertrain min/maxclip. PFN/LGB/MLP/비중/다른입력 그대로다. query 미래/다른온실/검증정답에 의존하는 학습·정규화 없음. 학습은 기존공개8640행MASK안전 입력을 재사용하며 원시train_y/잠금/EL1/test/외부자료/제출0.

## 검사 및 판정

22fold×3seed=66fit. 전fold에서 기존 LOG 산술raw와동일partition 재현1e−9를 요구한다. 첫fold 원ET 재현·각leaf 원y평균수동합·재학습/배치·미래/다른온실6대조, allrow raw의outertrain min/max내위치 확인. 현재자료가 아니라 합성반례로 최적성 설명을 대체하지 않는다. 전체점수/혼합식/클러스터bootstrap독립검산, high/ordinary/late/farm 손익 보고.

모든15칸(DIAG10/A/B/EXT10/EXT12×3seed) 개선,DIAG10 farm층화5기록일20,000bootstrap p_worse<.025/18 및CI상한<0. 후반2%악화도경고. 내부최적값/게이트/마지막clip/혼합비를좋은결과로사후조절하지않는다. 공개통과만으로최종후보/제출물생성없음. 기존모든안의기각유지.

근거: 고정첫fold log partition에서b²가중ratio식과산술ratio식의queryraw차1.05e−15이나, bquery를제거한원y leaf평균은현재raw와최대.221821차이(`../ec_log_partition_mean_20261004_v1/leaf_diagnostic_v1.json`). 후자는계산이달라진다는증거일뿐개선증거가아니다. `../ec_harsh_review_20261004_v1/비평_v6.md` 비평에따른단일사전안이다.
