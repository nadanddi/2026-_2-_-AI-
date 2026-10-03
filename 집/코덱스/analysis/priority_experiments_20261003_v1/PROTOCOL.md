# 중요도 순 실험 사전 고정 — 2026-10-03 집 코덱스

사용자: 오늘 얻은 정보에 따른 목록을 중요도 순으로 실행. 온도 실험도 직접 요청했으므로 이번 범위에 포함하며 모든 새 파일은 집/코덱스에만 저장한다. 새 제출물·확정 구성 없음.

## 순서와 구현
0. 저장 BASE/CODEX/PFN으로 W40G를 두 BASE 시드(7/101), 두 PFN 문맥(1–8/17–24), 기존 DIAG10/EXT10/EXT12에서 재구성. 이는 검증용 시드별 W40G이며 실제 제출의 BASE 3시드 평균과 동일하지 않다. W30G 실패16일 변화 별도 기록.
1. EC_DAY20: 각 예측시각 h에 최신 관측 구간끝 c∈{0,3,6,12,18,23}, c≤h 사용. 구간끝별 모델은 학습일마다 정확히 한 행. 입력10종의 0..c 평균/표준편차/0시/현재/차이, season·farm_id. ET160/leaf3. 목표=학습일 전체 EC 평균. 기존 계절v2 + .2*(하루수준예측 − 기존예측0..h평균), 학습범위clip. 학습일 전체평균은 정답으로만 사용, 평가일 미래입력·정답 사용0.
2. EC_STATE20: 같은 하루한행 표현, 고EC=학습일평균≥1. ET분류160/leaf3(확률), 낮음/높음 ET회귀160/leaf3(크기), 확률혼합. 동일 .2 수준보정. 한 상태만 있는 학습분할은 단일 수준회귀로 대체.
3. T_DIRECT20/T_GAP20/T_GAPSPLIT20: W40G80% + 별도 LGB20%. DIRECT는온도정답, GAP은sub_temp−현재실내(결측시fold훈련중앙값), GAP SPLIT은학습일평균gap의부호를분류한확률로두gap회귀혼합. LGB220,lr.035,leaves12,minchild100,lambda15,시드7/101,기존CODEX89개 MASK특징·원가중치. DIRECT는목표변환의효과를구분하는대조군. 분류는 ET160/leaf10, 하루총가중치가같도록1/24가중. 구분은훈련라벨로학습하고검증에서정답부호사용0.
4. EC_RARE_ET/T_RARE_GAP20: EC훈련일평균≥1,온도훈련일평균|gap|≥2인학습일가중4배. EC는원ET와동일600trees/leaf1로재학습,원계절ET를대체(.48×변화에원인과shrink),baseline에추가후clip. 온도는GAP모델만가중변경. EC원ET는공개DI1 etS 캐시와대조한다.
5. EC_KNN20/T_KNN20: 같은하루prefix표현으로train-only중앙값/표준화,7최근접(역거리). EC는하루평균EC,온도는하루평균gap을학습. EC는DAY와동일 .2 보정,온도는현재실내+gap 모델20%혼합. 평가일을이웃에포함하지않는다.
6. 별도전문가제안은1·3의목표/학습단위가다른모델로실행. 동일실험을다른이름으로세지않음.
7. EC_H0CHANGE: 0시10특징을각현재값−0시값으로치환(현재입력유지). 동일ET600재학습및원ET대체. 제거실험과구분.
8. TEMP/EC 실패위험은기존nested inner 캐시의독립예측잔차로분류라벨을만듦(TEMP일RMSE>.5,EC>.1). 입력은같은prefix하루표현·현재까지예측평균,ET160/leaf5.outer 검증 라벨로학습하지않음. 모델채택용보정은하지않고AUC/Brier/상하위위험군RMSE 진단만기록. 클래스비율·표본/기존학습일지원함께기록. 

## 검증·채택
- EC: 기존계절v2 publicOOF의A/B/DIAG10/EXT10/EXT12×7/101/2024, 동일fold·평가/비공개제외일±1buffer. EC라벨은저장된공개DIAG10의360일만읽음. 잠금파일·원시train_y EC·test입력읽기0. EL1 추가검증은공개통과후별도로사전고정하며EL1이없으면최종채택불가. 
- TEMP: 기존MASK400일·DIAG10/EXT10/EXT12×2시드×2문맥,동일split_mask. 평가입력은전부NaN,정답은train_y row_id/sub_temp만읽음. 가중치·전처리는fold훈련만. 최종채택은추가EL1 guard필수. 
- 모델 arm10개(DAY,STATE,DIRECT,GAP,GAPSPLIT,EC_RARE,T_RARE,EC_KNN,T_KNN,H0CHANGE)、family_k=10、DIAG10 5일block/farm 층화bootstrap20,000회, p_worse<.025/10=.0025 및상한<0. 모든 시드×검증기×문맥RMSE개선필수. 좋은slice만선별하지않음. 
- 사전고정된전체queue완주. 2~3불합격후현재오차구조/분포를중간확인하되후속설정변경없음. 결과후가중치/피처재선택0. 
- CPU동시최대4thread、GPU새학습없음. 모델체크포인트/OOF는새local、표/로그/프로토콜은새analysis. 원파일덮어쓰기0. 

## 누수 점검
| 입력/절차 | 현재시점가능 | 조치 |
|---|---|---|
| prefix입력평균·std·current/h0/delta | 예 | 각온실·날별0..c,c≤h만 |
| season | 학습입력으로추정 | 기존fold훈련일날씨만mappingfit,평가날씨사용0 |
| 하루 평균EC/temperature gap/고값·부호 | 예측시점불가 | 훈련목표/가중치에만사용,검증feature0 |
| imputer/scaler/nearest neighbor | train-only | fold마다 fit |
| 위험분류학습잔차 | nestedOOF이면 사용 가능 | 기존inner fit/query분리확인 |
| 기존예측prefix평균 | 예 | 같은 온실현재이전행만 |

원본메타데이터로구분한날은실제달력날짜로단정하지않음. 기존분할의동일날씨중복/인접상관한계를유지하여보고. 
