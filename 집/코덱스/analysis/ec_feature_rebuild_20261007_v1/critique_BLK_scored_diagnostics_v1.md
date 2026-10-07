# BLK 실제 진단 점수 독립 비평 v1

2026-10-07. 부모가 공개 query EC를 사용한 산술 검산을 이번 단계에 허용했다. 저장 predictions/results/cells/scorer spec/Decimal 검산 소스와 receipt를 읽고 PowerShell로 공개 query1440행의 RMSE·block 손실을 별도 계산했다. 새 모델 fit/GPU/가중·문턱 선택은 하지 않았다. analysis-verification의 별도 계산·반론·한계 원칙을 적용했다.

## 수치·등록 확인

scorer v6 source SHA, spec→prediction SHA, result→spec SHA, Decimal checker receipt→result SHA 및 SG2 supplemental preflight→현재 원 SG2 SHA가 모두 일치했다. CSV108행=6variant×3seed×6segment와 결과 JSON의 구조가 맞는다. 공개 query1440행/60기록일/고EC3기록일72행을 새로 집계했다.

PowerShell 별도 squared-error 누적으로 모든 cell RMSE를 다시 계산하여 저장값과 최대5.551115123125783e-17 차이였다. seed평균 row SE delta를8block으로 다시 합쳐 저장 block SSE와 최대7.105427357601002e-15 차이였다. checker는 별도 Decimal60/명시randrange20k bootstrap을 사용하며6variant 모두 PASS, p/CI/선별 결과를 대조한다. 본 리뷰가 bootstrap20k를 새로 재실행한 것은 아니며 해당 checker 소스와 current-result SHA를 확인했다. 모델 재fit 독립재현과 구별한다.

## 선별 판정

| 범위 | 방법 | 평균 seed ΔRMSE | p_worse | 선별 |
| --- | --- | ---: | ---: | --- |
| QUERY_ROLE | BOTH | +0.002566859 | 0.575271236 | FAIL |
| QUERY_ROLE | PAST | +0.022142501 | 0.804959752 | FAIL |
| QUERY_ROLE | GUARD | 0 | 1 | FAIL |
| RAW_PASS | BOTH | +0.000227465 | 0.482375881 | FAIL |
| RAW_PASS | PAST | +0.018842964 | 0.729263537 | FAIL |
| RAW_PASS | GUARD | 0 | 1 | FAIL |

양수는 악화다. BOTH/PAST의 전체 RMSE는 두 scope에서 각각 세 seed 모두 악화했다. 그 차이가 작다는 이유로 BOTH를 채택할 수 없다. guard는 activation0으로 baseline과 같았으며 효능 검증이 아니라 no-op 확인이다. '현재 고정6안은 이 BLK에서 선별실패'의 신뢰도는 높다. 'endpoint 정보 전체가 쓸모없다'의 신뢰도는 낮으며 이 결과로 주장할 수 없다.

RAW BOTH의 평균 ΔRMSE>0인데 bootstrap p_worse<.5인 것은 오류 증거가 아니다. RMSE seed평균과 bootstrap의 seed평균 MSE 차이·블록 재표집 분포는 다른 양이다. CI도 네 endpoint 안에서0을 포함한다. 점수표에 근접한 안을 승자로 정하는 해석은 금지한다.

## 이질성: 관찰로 남기고 적용 규칙을 만들지 말 것

일반1368행에서는 BOTH/PAST 모두 악화했다. QUERY_ROLE 평균 ΔRMSE는 BOTH+0.012277167, PAST+0.034531649; RAW_PASS는 +0.008576274/+0.029527802다. 고EC72행에서는 BOTH−0.049150318, PAST−0.040954675로 두 scope 모두 개선됐다. 고EC는 공개 정답으로 정의한 단3기록일이며 예측 시 실제 하루 EC 평균은 알 수 없다. 이 segment 개선을 근거로 고EC에서만 endpoint를 켜거나 threshold를 새로 고르면 label 기반 재튜닝이다.

위치별 행수는 앞576/가운데480/뒤384다. BOTH는 가운데에서 QUERY−0.009943605/RAW−0.012396362로 개선하고 앞에서+0.017890207/+0.014056668, 뒤에서+0.001860950/+0.000539654로 악화했다. PAST는 세 위치 모두 악화했다. 균등3분할이 아니므로 위치별 평균을 같은 가중치로 합쳐 전체 효과라고 하면 안 된다. 가운데만 적용하는 후속은 이 정답을 보고 만든 새로운 가설이다.

seed평균 block 손실 부호는 QUERY BOTH 개선4/악화4, PAST 개선3/악화5; RAW BOTH 개선6/악화2, PAST 개선5/악화3이다. 마지막 block의 양의 SSE 차이가 BOTH 약2.7, PAST 약7.3~7.6으로 전체에 큰 영향을 준다. 이를 제외해 개선으로 바꾸면 모집단 선택이다. seed3개는 같은 query와 공유 학습자료를 사용하므로 독립 검증3번으로 세면 안 된다.

8block만으로 시간 의존·공유 reference·source 불확실성을 해결하지 못한다. bootstrap p는 이 고정 설계의 진단량이고 새 외부 표본 확률이 아니다.6variant Bonferroni는 이 가족의 선별 기준이며 전체196후보/앞선 탐색을 대신하지 않는다.

## CH2 후속 사전등록 확인 및 다음 순서

파일 검색과 PROGRESS/기존 chain critique/CH2 reference diagnostics에서 CH2_REFONLY_GUARD·FLANK_SOURCE_MATCH·PAST_QUERY_PREFIX_STATE의 결과 전 제안은 확인했다. 그러나 별도의 예측후보 공식 registration은 찾지 못했다. CH2 reference artifact는201links/46components/양끝 same-component1의 train-only 구조 진단이며 `No additional predictive candidate registered ... yet`로 명시돼 있다. 이 수치는 label 연속성 비용으로 만든 그래프의 일부 특성이므로 물리적 source 정확도 또는 query assignment accuracy의 정답 대조가 아니다.

원 CH2의 EC trend+weather/indoor 비용 Hungarian과 이번 global mutual EC guard는 다르다. 현재 guard0 결과로 CH2를 기각하면 안 된다. 반대로 원 CH2 그래프가 연결된다는 이유로 유효한 예측 후보라고 승격할 수도 없다.

권고는 **현재6안의 BLK 선별을 종료하고 도메인24 단계로 이동**하는 것이다. BOTH/PAST 계수·위치·고EC 조건의 즉석 변형은 하지 않는다. 기존 CH2 source matching 제안은 '미시험 후속'으로 장부에 남긴다. 결과 전 제안은 있어도 예측 규칙·cycle/missing/fallback·누적 multiplicity·원검증기 조건이 아직 봉인되지 않아, 오늘 같은1440정답에서 확증 실험으로 취급할 수 없다.

사용자의 학습정답사슬 목표를 위해 CH2를 후속 실행하려면 별도의 bounded family를 새로 등록하고, 원source 고정의 차이만 시험한다. 참고 BLK 정답은 이미 공개됐으므로 이 layout의 점수는 탐색진단으로만 표시한다. 새 threshold/weight/source 선택은 train 내 pseudo-block 또는 nested 검증으로만 고정하며, 원검증기와 미사용 최종 seed/layout을 성능선별에 먼저 쓰지 않는다. 최대후속안·전체 variant장부·중단조건을 등록하고 끝나면 도메인24로 돌아와야 한다. CH2 후속의 정당성은 이번 segment 점수의 개선 방향이 아니라 결과 전 source 기반 가설에서 나온다.

## 목표 범위

새 reference-only cached PFN baseline은 원 제출14(.2PFN/다른 seed/context)·역사적 GPU OOF와 동일하지 않다. 이 baseline의6안 실패로 원 제출 endpoint 효과를 직접 설명할 수 없다. 기존 원 TM111/P2LOO/EL1/DIAG 검증,196후보 전체단계,미사용 seed/layout 최초1회는 미완료로 유지한다. 이번 리뷰는 산술·선별·해석의 검증이며 최종 모델/전체 목표 완료를 보고하지 않는다.
