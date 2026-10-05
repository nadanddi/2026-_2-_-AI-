# 고EC 여부 구별 모델: 공개 학습 참조 범위 대조
2026-10-05 집 코덱스. 사용자 새 구별 모델 요청. 추가자료 확인 질문은 병행하되 현재 확인된10/5안내(카탈로그6.309/메모리)를 기준으로 진행. 공식 PDF원문은 local미확보; Downloads 현장지식조사 문서는 운영위 문서로 확인되지 않았고 그 주장을 검증된 사실로 사용하지 않음. 새 자료가 추가 확인되면 별도 버전으로 반영.

목표=고EC **여부**(정답 하루평균≥1). 이전family28~30의 B승리분류/최근family31회귀잔차와 다른 목표. 고EC 크기 예측/혼합/제출 없음. 규정허용성은현재사용자메모리근거이며원문검토완료라고표현하지않음.

고정 모델=LogisticRegression C1,class_weight balanced,lbfgs,maxiter3000,tol1e-8. StandardScaler fit은각훈련행만. 두비교 PAST/ALL은참조허용범위만 다름. querysamefarm/samepass,참조훈련일제외조건은기존outer/inner purge 그대로. PAST는기록순서이전 train_y,ALL은훈련전체(뒤날짜 public train_y포함);평가행의미래입력/다른온실/test통계사용0. queryprefixsignature0~h와참조훈련기록signature0~h,14열표준화/결측대체는ref-only. 거리최근접5일stabletie day.

특징17기존(currentA,prefixA,14prefix입력,h/23)+5참조(EC평균,고EC비율,ECstd,최소거리,log1p참조수)=22. 특징강화가아닌reference범위한요소대조. 참고baseline=현재까지A예측평균≥.9,31/30의full-daymean 기준과각h시점구분.

검증=공개DIAG10 10fold×3기존seed,meta=각innerb5기록3분할±1일purge/시드·fold·mode90모델. 외부threshold는해당fold의metaOOF **23시** 고EC점수로95%recall을목표로가장높은문턱선정(상수 fallback포함). 외부y로문턱변경0. meta예측으로문턱을정했으므로meta문턱후성능은최종검증점수아님. 시점별0/6/12/23시와전체시간 AUC/TP/FN/FP/precision/recall 보고. 두모델120fit씩240 총,outer60CSV/각meta60CSV/라벨사용캘리브레이션한계명시.

선행단서 기준=ALL이PAST보다외부23시3seed에서recall낮아짐없이FP감소,또prefixA≥.9보다3seed에서recall낮아짐없이FP감소. 이기준미통과시고EC구별이개선됐다고주장하지않음. 통과해도새배치/새seed/분류기의혼합RMSE검증전채택아님. 일반날손해/고EC전문가품질은이실험으로확정못함. hyperparameter/threshold tuning0.

원A캐시·public라벨·prefix입력provenance기존감사로재확인,rawtrain_y/test값/EL1/잠금읽기0. 원시train_X참조는공개360기록만,학습행은test입력NaN MASK 기존경로(미래eval입력차단)유지. reference特徴및label대리값은innera/outertr만으로계산. meta-heldout일 라벨은classifier/reference/scaler fit에서제외.
