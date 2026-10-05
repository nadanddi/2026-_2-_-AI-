# 낮게 예측된 고EC와 일반날 오탐의 공동 학습
2026-10-05 집 코덱스. 사용자 진행 요청. 탐색 실험이며 제출·확정 구성·전문가 혼합 없음. 기존산출물 불변.

문제: 기존 내부b만으로 구별기/문턱을 학습해 고EC0분할 및 경계사례 부족. 이번에는각외부검증 tr전체에4fold 날짜OOF를 만들어 두종류사례를 함께 학습한다. 고정두안 UNIFORM/HARD는 학습가중치 하나만 다르다. 같은 데이터/피처/대리모델/시드/문턱.

외부=기존DIAG10/A/B×seed7/101/2024. 내부OOF=각farm 날짜정렬5기록일block의ordinal//5 %4,train은valid동일farm±1day purge;4fold전체coverage. outerq는어느proxy/classifier/ref/preprocess 학습에도사용안함. 다른fold의저장OOF를합치면outerq정답이원학습에들어갈수있으므로사용0. innerquery reference/EC proxy훈련에도같은해당foldtrain만.

대리일반EC=기존R3구성의LightGBM tweedie component(core.lg,800trees/31leaves/etc),BASE에서day제외+ref-onlyseason. 전체R3/TabPFN혼합재현 아님. proxy출력은기존shrink .5 및refEC범위clip. fulltr에fit한proxy로outerq예측. 대리는특징/경계사례정의용이며기존실제A가비교기준·GUARD1차판정이다. SG2최신합본 효과미시험.

분류기22열=currentproxyEC,prefixproxyEC,14prefix입력,h/23,5ALLreference(ECmean/highfraction/std/firstdistance/logn). 내부OOF피처모두같은innertrain만사용. LR C1 lbfgs/tol1e-8/maxiter3000/scalertrain-only. UNIFORM=양성·음성class균형가중치. HARD=같은가중치×4 (진짜고EC 하루mean≥1 & 해당시각proxy prefix<1.2, 또는일반날 & proxy prefix≥.9). 두class가중치를훈련평균1로최종normalize해penalty변화차이를억제. HARD는고EC과소와일반과대모두강조;하루전체proxy로이른시각행을선택하지않음. raw공개정답이아닌감사완료publiclabel사용;test/잠금0.

문턱은양쪽모두.5 고정(현재외부정답을보고정하지않음). DIRECT=분류기만,p≥.5. GUARD=기존A prefix≥.9 & p≥.5. 주평가GUARD:23시모든seed×DIAG/A/B TP감소0 및 FP증가0,DIAG매seed FP엄격감소. DIRECT는보조보고. 통과해도새seed/새배치/결합RMSE·원통계채택조건확인전채택0. 두안비교1요소가중치의진단,결과에따라추가문턱튜닝0.

훈련기록에lowpredhigh/falsehigh가각≥3개의독립날짜존재하는지23시기준으로보고;부족하면'공동학습정보충분'주장0. 내부고EC최소3일/일반10일없는OOFfold는중단·기록(상수고EC0모델학습0). fullOOF전체는분류학습에쓰이므로학습집계는성능검증아님. classifier훈련행적합값/외부gap보고. 관측된일부공개날짜를재사용한탐색,새홀드아웃아님.

실험예상300개proxy(20outer×3seed×(4inner+1full))+120LR(2안). proxy구성/특징/가중치/문턱사전hash. inverse-scaled LR scalar/gradient/임계값·rawcounts재計算,LightGBM dumped-tree scalar재생·context ID/±1purge·参照날짜검증·2proxy&2LR재학습/기준분모검算. 0/6/12/23時/farm×pass/시드spread 및bootstrap오탐차이진단;고EC미탐증가시기각. A/B重複日期출현명시. 평가미래입력/타온실評価入力/testfit0;MASK기존featurepipeline 유지.
