# 정제 EC 모델 첫 폴드 독립 중간 비평 v1
2026-10-09 · 새 실험 ec_discord_clean_model · 최종판정 아님

판정: **새 차단 누수/산술 문제 없음. 사전 계획대로 나머지9폴드 및 요청한 최종 정제모델 제작 진행 승인.** 한폴드 성능으로 조기기각·임계수정·모델선택하지 않는다.

## 실제 독립 검산
verify_independent_v1.py는run/recipe모듈을import하지않고CSV/fsum·NPZ배열산술로검산한다. env는실행라이브러리경로부트스트랩만사용. data 및midpoint모드실제exit0, data_independent_v1.json/midpoint_independent_v1.json에PASS.
- public360자료는12일288행통삭제→8352행348일. global삭제목록은CVtrain에참조하지않는소스계약확인.
- 각10foldselector_snapshot의z표준화모양·유한값·training목표dailymean,5NN거리순서/tie/self·±1제외/중앙값/삭제조건/저장이웃을별도산술로전수재생. public및foldcleanX/Y/삭제IDs/原값·24행완전·원본행유지·등록전cache/sourcehash 확인. qsnapshot targetdailymean은공개360정답으로만재계산.
- fold0은학습10일통삭제6048행,검증38일912행/조건부삭제0일. 일반37일888행/고EC1일24행. pass2표본은첫fold없음. filtered=all이므로두공동비교를독립표본처럼해석금지.
- baseline3모델새factory재현gaps ET0/LGB0/MLP3.1086244689504383e-15. 후보fitreceipt9개·모든kind×seed/훈련IDs/모든행수·등록검사PASS.
- baselinecached각ET/LGB/MLP .6/.3/.1raw혼합 및ensemble을독립배열산술로재생. CLEAN저장mixraw의3seed평균, BASE/CLEAN모든raw→시간순prefix평균.5/.5→각trainingclipbounds전수재계산PASS. constant학습평균도fsum검산.
- 모든그룹RMSE/bias/상수RMSE를별도fsum검산. farmer×day//5(상대기록번호구간)pairedbootstrap20000회/두scope재생해p/CI모두일치.

## 중간 수치(성능 최종판정용 아님)
전체/filtered ensemble BASE RMSE.1360304656→CLEAN.1372819477,delta+.0012514821559,p_worse.64285,각alpha.0125. 두scope 모두사전screen부분미통과지만첫폴드이므로최종판정없음.
일반RMSE.1167292050→.1192550234, 고EC단1일.4461073380→.4358388667. 고EC1일이좋아졌다는사례만으로모든고EC일/실제평가에좋다고주장금지.
CLEAN학습ensembleRMSE(생산자receipt값).0342613938,원검증.1372819477,격차+.1030205539. trainraw예측은미저장하여학습RMSE값자체는독립재생이아니며receipt기반격차산술만독립확인했다. final모델9개는직접predict로보완예정.

## 저장 자료의 검증 한계
CV CLEAN의각kindraw예측·CV모델은저장하지않았다. .6/.3/.1개별mixture는source검토로한정하며원cachedbaseline에서는실제수치재생했다. CLEANmixraw이후seedmean/prefix/clip은전수검산완료. 최종9joblib은각구성원predict를다시계산하여최종실제mixture경로독립검산을보충할것.
selector의원47dailyfeature생성전체를처음부터별도재생한것은아니다. 저장z스냅샷의거리/target/삭제조건은전수독립재생했고원preprocessing/matrixhash는source계약검토한범위이다.
조건부검증모집단은heldoutdailyy로선정되어실제평가에서선택불가. 단순filteredscore좋아짐을실전성능주장하지않음. 전체all공동screen과원all/pass2all2%guard를최종결과에서그대로적용.

최종관문: 전체90모델receipt/훈련검증격차/농장×구간과일반·high/손실집중도/모든seed방향/두bootstrap·전체guard. CV완료후에만추가40일을400일finalfit에포함,40재채점0. 모델9artifactsha/reload/각farm×pass1/pass2및h5/h17미래/타farm변조삭제·실제predictionpath로검증. 성능과무관하게사용자요청모델까지완료,채택0/제출0.
