# 고EC 구별기 희소 분할과 문턱 보완
2026-10-05 집코덱스. 사용자 진행 승인. 이전 모델/자료 불변. 실험 및 진단만; 채택/전문가혼합/제출 없음.

같은 ALL22특징/LR C1 balanced 고정, 기존 공개 DIAG10/A/B·3seed7/101/2024 그대로. 평가 입력은 같은온실 현재까지만, 전체 공개학습 정답 참조는 기존허용범위. 원PDF 확인했다고 주장하지 않음. 같은 공개 날짜를 여러번 쓴 탐색 실험이며 신규홀드아웃 아님.

한요소 대조 두 단계: SPLIT=기존분류기의 메타분할만 보완. GUARD=SPLIT 점수로 기존A prefix≥.9 판정의 일부를 거르는 안전형. GUARD는 새일을 고EC로 추가하지 못하므로 기존미탐을 회복하지 못함. 고EC 크기예측/RMSE개선 평가 아님.

메타분할은 날짜 단위. 고EC/일반 × farm/pass 별 날짜정렬을 고정 순환 배정(고EC는 1일씩,일반은5기록일block), 고EC 글로벌offset으로 전체fold 배분. 동일farm±1day purge. 3분할 먼저, 부족하면2분할. 각 train 고EC≥2일/일반≥10일,각valid고EC≥1일; 충족안하면 해당 전체 외부fold를 기존A로 fallback. fold검색에서 점수사용0/분할조건만 점검. 학습규모/양성일수를 별도 기록. 기존 A/prefix 특징의 fullcrossfit 재학습은 하지 않음(전달 문맥 차이 한계).

SPLIT 문턱은 메타23시 고EC 점수의 하위5%(lower), 기존규칙 유지. GUARD 문턱은 메타23시 기존A≥.9가 잡은 고EC 중 가장 작은 분류점수, 이표본≥3일이어야 함; 부족하거나 문턱≤0이면GUARD 전체fold fallback. 메타에서는 기존포착을 유지하도록 고정하지만 외부에서 유지 보장아님. 후보/기준은 외부정답 보기전에 고정. 외부오탐을 보고 문턱변경0.

primary 선행기준: 각mode마다 3seed×DIAG10/A/B 외부23시에서 기존A보다 TP감소0 및 FP증가0,각seed DIAG에서FP엄격감소. 미충족이면 채택/진전 주장0. 통과해도새seed/새배치·결합RMSE 및 원채택조건 전 채택안함. 0/6/12/23시·farm/pass 별 counts/AUC/precision/recall·meta calibration 숫자보고(메타 성능은 최종검증아님). 2안 동시검토이며 새RMSE후보검정 아님.

누수: q/tr & b/a 분리 기존감사 재생; 참조정답은a/tr만,메타훈련과valid날짜/purge 분리; scaler각학습만;문턱학습meta만. 고EC라벨은훈련 분할구성에만,평가배정에 사용0. 입력NaN MASK기존경로/평가미래·다른온실·testfit0. split/특징/캐시hash 학습전고정,독립확률/학습gradient/분할·문턱재생/2재학습/stdlib counts쌍AUC교차검산.
