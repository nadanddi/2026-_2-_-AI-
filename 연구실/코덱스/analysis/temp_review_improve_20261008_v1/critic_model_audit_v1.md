# 9회차 온도 모델·패키지 독립 정적 감사

2026-10-08 · 연구실 · 코덱스 critic. 사용자 새 온도 감사·개선 예외 요청에 따른 Phase1. 새 fit/채점0, 원패키지 수정0. 자료: 9회차 ZIP 새 추출본 v13/temp_mask/train_flags/v3/v7/resid_reset/feature_v4/fp/env/season/temp_v13_checks 및 README/manifest, 카탈로그6.41/47/222/430/442, 집 코덱스 TK world.py·종합보고서와 집 클로드 VW1 소스. 전체51파일의 재현 SHA 전수 검증과 개별 모델 경로 정적 감사는 구별한다.

## 요약 판정

모델 생성 경로는 완결되어 실제 BASE/CODEX/PFN8 학습과 1440행 출력이 가능했다. 누락한 자체 학습 가중치 파일 때문에 계산이 불가능한 모델은 아니다. 다만 **정확 재현 보장, 등록 문서, 독립 시드 증거, 검증 가중치의 fold 격리**에 결함·제약이 있다. 미래 입력/다른 온실 격리 위반은 지금 정적 감사에서 발견하지 않았으나 전체 모델 교란검사 재실행 전 완료 PASS를 주장하지 않는다. 이미 수행한 6.442는 정확 재현 실패 그대로다.

## 1. 구성·경로·누락

- env는 AGRI_DATA 또는 package/data를 common.DATA에 연결한다. v13의 BASE/CODEX/TabPFN 함수와 resid_reset89열, fixed season 두 표, 원본 reference CSV가 동봉되어 실행에 필요한 프로젝트 코드가 있다. 오래된 `../analysis/codex_independent/2차` sys.path 삽입은 무효경로이나 동봉 code/resid_reset_features로 실행되며 실제 import경로 검증도 통과했다.
- 온도 배포폴더의 CSV3개 미동봉은 README에 명시돼 있다. 공개 PFN checkpoint는 합본 EC 폴더에 있으며 SHA를 온도 README/manifest에서 지정한다. 온도만 단독 오프라인 실행할 때 checkpoint 배치 방법은 설명이 부족하다. 원zip 합본+공식CSV+동일SHA checkpoint 연결로 실행 가능함은 이미 검증됐다.
- 자체 모델 이진파일을 보관하지 않는 설계는 재학습 자체에는 문제없으나, 제출 당시 구성원 원시NPZ가 패키지에 없어 0.004595℃ 차이를 구성원별로 분해하지 못한다. 최초 학습의 문맥 rowID/float32 특징 해시·batch/thread/전체 환경 fingerprint도 미등록이다. 원제출member NPZ는 local 검색 범위에서 미발견이며 영구 부재라고 단정하지 않는다.
- 온도 README submission_12 제목·manifest_12 이름과 내용 v13의 불일치, README의 현재 EC 설명이 오래된 submission04인 점은 문서 결함이다. 상위 README의 최종합본 안내와 v13 소스가 실제 대상이다. 온도CSV의 EC는 복사 열이며 9회차EC 재현 결과가 아니다.
- manifest의 BASE params는 주요LGB설정만 담고 Ridge alpha100/Nystroem gamma.005/500개·alpha1, CODEX residual LGB220/.035/leaves12/minchild100/lambda15 등은 소스에만 있다. 코드가 있으므로 계산 누락은 아니지만 설정 문서로 완전하지 않다. 저장 runtime에는 torch/tabpfn 버전과 선택 checkpoint/문맥 설정을 함께 보존해야 한다.

## 2. 실제 학습 파이프라인과 가중치

- BASE는 물리14열 median imputer→가중 LinearRegression→135열 residual LGB(.65), 별도135열 median imputer/StandardScaler/Ridge100(.25), Nystroem RBF gamma.005/500→Ridge1(.10). 가중치는 regression loss에 전달되고 imputer/StandardScaler/Nystroem 자체의 분포 계산은 비가중이다. 이 선택은 일관된 기존 구현이며 자동 bug 판정은 불가하나 “가중치가 전 처리 모두에 적용된다”는 설명은 틀리다.
- CODEX는19물리열 median/StandardScaler/Ridge100 기준선과89열 LGB residual의 합. 과학적인 물리 정답식이 아니라 학습된 선형 기준선이다. BASE·CODEX 물리함수의 훈련 preprocessing을 평가 행에 fit하는 경로는 없다.
- **가중치 설명 정정 필요**: restored 행에 .2를 적용하고 noisy 날 행에 다시 .2를 곱하므로 교집합은 **.04**이다. noisy 플래그는 .2단독과 다르다. 2341은 w<1 총수이며 전부 .2라는 뜻이 아니다. 실제0.04행 수는 별도 무학습 계산으로 확인하면 된다.
- **fold 격리 문제**: TF.row_weights(labsubset)는 subset에 정렬하지만 restored/noisy 계산은 common.load_raw 전체train_X를 읽는다. noisy day rank 및 상위25% threshold는 전체400학습날의 CO₂차분을 사용하며 cold 예외는 전체훈련시계열EWMA를 사용한다. 평가test_X는 읽더라도 가중치에 쓰지 않아 최종학습의 평가누수와 다르다. 그러나 CV의 검증 입력을 학습 가중치 결정에 쓰므로 fold-local 훈련통계를 요구하는 새 실험에서는 각fold 훈련부분으로 재계산하거나 이 관행을 명확히 분리해야 한다. TK 종합보고서도 기존전체rank유지를 이미 한계로 공개했다. 이 결함 때문에 기존CV숫자 자체가 조작됐다고 말할 수는 없다.

## 3. 시드와 근거의 독립성

- BASE Ridge factory(s)는 s를 쓰지 않는다. 7/101/2024 세 fit은 독립 확률 반복이 아니며 동일Ridge를3번평균한다. BASE LGB는 subsample.8/colsample.6으로 시드에 따른 변화가 있고 Nystroem500 landmark도 random_state=s로 변화한다. “전체 BASE3seed가 독립”은 구성원별로 구분해야 한다.
- **CODEX726/727 독립성 없음**: 소스는 LightGBM subsample=1·subsample_freq=0·colsample=1 기본과 동일Ridge를 사용한다. 두 seed가 훈련 다양성을 만들어야 할 설정이 없다. 실제동일성은 TK 종합보고서에도 이미 확인돼 있다. 두시드평균은2개독립증거라고 쓰면 안 된다.
- PFN1..8은 문맥 비복원2000행 표집과 모델seed를 같은 번호로 설정한 반복이므로 변화는 문맥과 모델ensemble난수가 함께 바뀐다. 독립seed의 효과를 분리하지는 못하나 문맥묶음A1..8/B17..24는 별도강건성 비교에 활용됐다. 추가 규정검사는1문맥만이라 전체8조건 확인으로 확대하면 안 된다.
- **W40G-S는 “단일seed 증거밖에 없음”으로 끝내면 오래된 설명**이다. 원README는 단일seed 탐색/LB판정용이며 채택기준 통과 후보가 아니었다. 후속6.222 VW1은BASE7/101×PFNA/B4조합, DIAG10/EXT10/EXT12+EL1 방향16/16개선이지만 DIAGp_worse.16~.31로 통계미달을 기록했다. 이 후속증거를 반영해야 한다. 원README6.92 인용은 현재EC H2 항목이며 오류이다. root 새후보는LB0.5127로 가중치/문턱을 고르면 안 된다.

## 4. 시간·온실·MASK·season

- v13.build_frames는 학습세계의 test 입력14열을NaN으로 만든 뒤 BASE피처를 생성하고 CODEX 학습도mask feature를 사용한다. 평가세계는 실입력으로 생성한다. 레이블은공식train_y만. gate는현재in_temp로 clip((t−8)/2);NaN이면1이라PFN최대비중이 된다. NaN기본값선택은 명시된 설계이며 이득근거는 별도 필요하다.
- resid_reset89는farm/day 그룹 expanding mean/std·diff·당일EWMA와hour0입력만 사용한다. h0는00시를ffill하며 뒤시간값을앞시간으로가져오지 않는다. fp_features h0도00시만reindex한다. features_v4 seg의groupmax는00시외NaN으로가려진단일값의broadcast이므로 하루최대실내온도를쓴다는 해석은 틀리다. phys filter는farm별EWMA다. 이 정적경로에 다른온실현재입력/미래입력의직접경로를 발견하지 못했다.
- **fixed season 최종학습과CV는다름**: 원season.mapping은훈련1차두온실날씨전체로표준화/검색reference를 만들고2차훈련날외기를매칭한PAV표를만든다. 평가day는farm/day로표를보간하며평가외기/미래평가입력을읽지않는다. 허용훈련자료로구축한global모델이므로다른온실훈련자료활용과다른온실평가입력활용을혼동하지않는다.
- fulltrain season표를그대로CV에서읽으면검증날외기까지mapping fit에들어가는입력누수가된다. 단, TK world.season_fold는foldtrainkeys만vectors를전달하고validationkey불교집합assert,query는day보간으로만계산한다. 6.222가사용한TK2자료의season은이방식을쓴다. fullfixedtable누수를TK2기존결과의확정결함으로단정하면안된다.
- temp_v13_checks는determinism1쌍/future4cut/isolation2farm을 end-to-end검사하지만 PFN sample1로줄인다. 학습feature불변을확인하고weights는전체훈련입력고정이라평가교란에불변. 표본수를늘려도구조가같다는주장은소스설계추론이지PFN8실증검사아니다. 기존6.41/47은5행순서/배치위치관련증거이며전체시드·모든배치설정확정증거가아니다. 이번0.004595차이원인도그기록만으로확정하지않는다.

## 5. 재현·품질 판정과 다음 실험 조건

6.430은 **EC package partial/fullquery assert 실패**이며온도run이assert로중단됐다는근거가아니다. 온도6.442는원소스학습완료1594초·소수6자리1223행불일치/max.004595/RMS.00072017이다. 지금환경PFN문맥1반복raw동일은전체정확재현성/원인확정과다르다.

root가실행할featurecausality는무학습feature감사범위이며전체모델규정검사를대체하지않는다. 새실험전고정해야할것: 동일기준·동일fold·seedfamily, fold-local weights와fold-local season의정의, 검증dayMASK를학습피처에적용하는방식, 사전bonferroni와전칸방향+DIAGp기준, 공개/잠금/EL1범위명시. 가중치관행수정은현재모델개선후보와분리한감사조건비교로취급하여비교기준까지몰래바꾸지않는다. 새제출패키지/확정구성은만들지않는다.
