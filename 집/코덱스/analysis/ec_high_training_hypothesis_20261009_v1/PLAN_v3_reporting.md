# 보고 보완 v3 — preflight 소스비평 반영 (후보fit0)
- bootstrap day//5는 5기록번호 고정구간이며 실제5연속달력일과 같다고 주장하지 않는다. 양농장층화로 같은draw를세비교에사용.
- ci95_delta는 서술용95% CI, Bonferroni alpha .025/3 통과 여부는p값으로만판정(해당95%구간을조정구간이라고하지않음).
- 독립검산보고에서 rawET RMSE/각농장×pass×일반고EC/fold일반RMSE 분산 표를보충. 생성CSV에raw_et/일반날판정재료/농장/구간/fold 모두포함.
- CONTROL은표본수만맞춘일반일제거단회. target분포/고EC제거자체와discordance특이성을분리하지못한다. 선택적제거성공을설명안되는EC만의유해성인과증거로표시하지않음.
- 준비 첫실행PID32524 유지. 중복된prepare(fit0) 재시도PID11368은명령줄검증후종료,모델학습중복0. 원run_v1/계획v1/v2보존.