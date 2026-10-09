# 계획 보완 v2 — 계획 비평 반영, 아직 fit 0
PLAN_v1과 함께 적용하며 아래 항목이 우선한다.
- 세 확률 비교 ALL−BASE/DISCORD−BASE/DISCORD−CONTROL를 사전 고정, 각 alpha=.025/3. DISCORD 진단 지지는 두 비교에서 일반일 모든seed 감소와 각 블록bootstrap p<alpha를 모두 요구. ALL은 해당 baseline 비교에 같은 조건. ‘입증’/채택은 아님.
- 고유DIAG 8640행/360일의 비중복을 강제. bootstrap은 seed를표본으로pool하지않고 seed평균예측의farm층화5기록블록pairedSSE, 같은RNGdraw를모든비교에사용. 한고정random CONTROL의운을제거하지못하며 인과주장금지.
- 각training/query날24시간완전·유일ID·finiteEC 강제. samefarm ±1 이웃제외 후5개없는날은오류. CONTROL farm×pass 일반후보부족이면해당실험중단·다른층대체없음. DISCORD 0날이면해당fold no-op로baseline재사용·개선아님.
- 선정 거리대치/표준화는원outertraining47일평균에서한번fit후고정,seed/arm동일. ET imputer는각삭제학습에서재적합(현재pipeline기본); 학습삭제+훈련전처리반응의합이며원imputer고정개입이아님.
- ET raw/평활단독 및actualEC14후처리 각각평가. 각farm/pass·pass2일반/고EC·F47_161제외·fold분산 필수. 전체 또는pass2일반>=2%손해면 유해영향일부완화라도실제효용보류.
- baseline30cached provenance·prep/currentSHA·orderedtrain/query특징hash/PFN40/SG2ref 확인. 첫foldseed7원ET재현1fit PASS뒤후보. 전체학습모델삭제가아니며SG2/LGB/MLP/PFN에서고EC는유지. 평가고EC없음가설은이번실험으로확정불가.