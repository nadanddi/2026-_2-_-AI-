# 로그 비율 분할·산술 잎 평균 · 실행 전 고정

2026-10-04 집 코덱스. 단일 LOG_PARTITION_MEAN, 지속목표 family16. 기준 실제 EC계절v2, 공개22fold×seed7/101/2024, 입력·계절·buffer·혼합·최종clip 고정. 모든15칸 개선 및 DIAG farm층화5기록일20,000bootstrap p_worse<.025/16·동일수준 CI상한<0. 추가안 도입에 따라 이후 비교 family16, 이미기각한 안 소급채택없음. rawtrain_y/잠금/EL1/test/제출0.

수리 대상은 LR1(6.190)의 `bq*exp(forest_mean(log(y/btr)))`이다. LR1은 코드상 no-smearing을 명시했고, 목표는 로그평균의 지수이며 원단위 평균과 다르다. [Duan1983](https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478017)의 publisher 초록에서 역변환 평균 문제 확인(본문 미확인). 이번은 Duan 전역 residual-smearing 재현이 아닌 forest 잎을 원단위 ratio 평균으로 바꾸는 자체 계산이다. Jensen/가중 평균 항등식은 직접 검산한다. 과거 기각이 코드 버그나 판정 실수였다고 단정하지 않는다.

기존 POWER2_PARTITION_ET는 EC² 분할 후 각트리sqrt였고 이번은 동일 LR1 log(y/seasonbaseline) 분할 후 train ratio 잎 평균이다. 원단위 잎값/계절 정규화/구간 이동 가정이 다르지만 목표변환 계열 반복 탐색 한계는 유지한다.

1. 기존FULL38(day→season끝열), 학습 fold만 기존계절fit. LR1과같이 farm별 train-daymean을season순정렬·21일 centered rollingmedian(min5)·bfill/ffill 후 np.interp, btr/bq 산출. 검증정답으로 b를 fit하지 않는다.
2. ratio=y/btr 양수유한 확인. 기존 동일 ET600/leaf1/maxfeatures1.0/seed로 log(ratio) 학습.
3. 각tree.apply(Xtrain)의 leaf count/sum(ratio)로 모든 실제 잎의 **산술 ratio 평균**을 산출. query leaf는 반드시 count>0. newraw=bq*(600그루 산술leafmean 평균). 로그평균 지수나 트리간 분산을 posterior로 해석하지 않는다.
4. 후보=clip(실제계절v2+.48*(shrink(newraw)−기존oldETshrunk),trainEC최소최대). 동일인과shrink 한 번. baseline/oldET 공개83,160행 경계clip0으로전수동등성검산한 범위만 부품교체라고부른다.

고EC를 선택해 올리는 방식이 아니다. 양의ratio 같은분할이면 newraw≥기존LR1raw이며 일반일도 올라갈수있다. train b의farm변화/계절 앵커희소성/FS의farmID부재로 `bq E[ratio|X]`가 참EC조건평균이라는 보장도 없다. 전체 RMSE를 직접 평가하며 고EC 편향·일반/후반/농장 손실을 같이 보고한다.

첫fold LR1 etL 캐시(동일tree logratio) 재현, 원ET 캐시재현, train-ratio leaf값과 tree.logvalue 직접대조, leaf count/empty 검사, raw Jensen, 독립 fsum 평활/혼합, scalar toy(1,3→geometric√3/arithmetic2), 재학습/배치/미래·다른온실 변조. 학습/분할/공개OOF만으로수치검증. 성능보기전에코드커밋고정, 다른AI파일편집없음.
