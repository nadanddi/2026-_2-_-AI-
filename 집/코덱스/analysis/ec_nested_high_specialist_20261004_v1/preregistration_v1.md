# 실제 계절v2의 R3 고EC 전문가 비중을 내부검증으로 학습 · 실행 전 고정

2026-10-04 집 코덱스. 가족수17의 새 단일안이다. HM1 공개 최적비중을 선택하지 않고, 각 외부 학습집합 안에서만 비중을 학습한다. 기존 HM1은 R3S 기준 고정50% 혼합이고 이번은 실제 계절v2의 R3 부분에 내부학습 비중을 적용한다. PFN20%는 유지한다.

## 모델과 계산

- 기존 공개22fold×시드7/101/2024, FULL38의 day→season 마지막 열, ±1buffer와 기존 제외행을 그대로 사용한다. 새 원시정답/잠금/EL1/test/외부자료를 읽지 않는다.
- 각outer의 내부학습 a/내부검증 b는 현재 matched CPU의 같은 row_id·선택 규칙·buffer를 쓴다. a만으로 계절 변환을 적합한다. 내부 R3raw .6ET+.3TweedieLGB+.1MLP, PFN4문맥×2000행/4estimators/CPU float32/기존V2를 그대로 사용한다.
- R3S=clip(shrink(R3raw),fulltrain 최소/최대), actualbase=clip(shrink(.8R3raw+.2PFNbag),fulltrain 최소/최대). shrink는 .5현재+.5당일0..h평균이다. gate는 이미 shrink된 R3S를 다시0..h 평균한 값≥.9인 행이다(원HM1 표지 방식).
- 전문모델 sp는 학습 일평균EC≥.8인 날짜만의 ET600/leaf1/FULL38다. sp=clip(shrink(spraw),hightrain 최소/최대). inner와outer 각각 새로 학습한다. hightrain이2일 미만이면 해당 단계의 sp를R3S로 대체해 방향0으로 둔다.
- d=.8×gate×(sp−R3S). w는outer×seed당 하나이며 두farm·모든hour의 **전체 innerquery행(게이트0 포함)**으로 `clip(−mean((actualbase−y)*d)/(mean(d²)+.01),0,.5)`를 계산한다. 평균행수 분모·λ.01·상한.5를 바꾸지 않는다. 최종후보=clip(actualbase_outer+w*d_outer,outerfulltrain 범위).
- w식은 **최종clip 전** 평균제곱손실+λw²의 제한 최적해다. 최종clip 후 RMSE 정확최적이라는 주장은 하지 않는다. 상수0쪽 수축과범위 제한은 사전고정이며 좋은값을 사후 탐색하지 않는다.

## 실행 전 관문과 감사

matched source SHA256=`b1041713e093ca9b8f17437e05b194f284a7e44e8a681a06ce033bc6fc8a7e40`. source_sha,22CPU/88PFN,완료fit_audit1584검사와oof.csv 존재 및 기본검사,첫CPU/PFN 재현PASS를 요구한다. 미완료이면 --prepare는목록만저장하고 새학습0; 실제실행은실패종료한다. 기존작업이살아있으면재시작하지않는다.

모든내부 train/query/context ID와외부train 포함관계·buffer를검사한다. outer 원R3캐시는 예측SHA·train_row_id·query_row_id·정답·목표범위·raw합을 전수확인한다. HM1 cache의 개수·현재 hash만으로 과거train provenance를증명하지 않는다. **outer specialist도모두새fit**하고 HM1숫자와의일치는감사로만쓴다. 첫inner/outer 재학습·첫8행배치·prefix 수동합·가중치KKT·scalar독립계산·미래/다른온실변조를검사한다.

inner/outer hightrain일수·목표범위·출력분포,flag행수/독립일수,w=0/상한빈도,clip횟수를기록한다. 내부와외부의표본량/계절/범위 이동이 남는다는한계를유지한다. 공개 high/ordinary/late/farm 세그먼트는진단만하고정답기반gate를새로선택하지않는다.

## 판정

실제EC계절v2 기준모든15칸(DIAG10/A/B/EXT10/EXT12×3시드) 같은개선방향+DIAG10 farm층화5기록일20,000bootstrap p_worse<.025/17 및CI상한<0. 기존사용자직접규칙을따른다. 새EC규칙으로이미본후보소급채택없음. 후반2%악화는안전경고도기록한다. 공개통과만으로최종후보/제출을만들지않는다. 독립후속확인은별도단계다.

근거/비평: HM1 R3S손실작은혼합방향9/9와실제계절v2 6/9차이(사후진단), `../ec_harsh_review_20261004_v1/비평_v1.md`, `비평_v5.md`. 이시험은그방향이새내부학습에서도유효한지확인하는것이지이차식오라클최적점을제출하는것이아니다.
