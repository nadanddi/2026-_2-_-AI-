# LGB 잎계산 후속 점검 v1 · 2026-10-04 · 집 코덱스

family21 기각 상태는 유지한다. 이번 작업은 원 코드·공식 4.7.0 소스·논문 수식 대조와 합성 스칼라 수식 검산뿐이다. 모델/입력/캐시 수정, 새 학습, custom objective 실행, 공개 점수, 제출은 모두 0이다. 현재 모델에서 실제 A/B 또는 고EC 잎을 추출하지 않았다. 따라서 고EC 과소의 원인이나 개선 가능성을 실증한 결과가 아니다.

## 실제 설정과 공식 구현

로컬 `집/코덱스/analysis/codex_independent/rl_ec_v1/run.py`의 core.lg(seed,'tweedie')를 직접 읽었다. SHA `057ff4d8da6f3af29105b049251498f79f7fcd6b88d980a8222e5629eb5ce3b2`.

| 설정 | 현재 값 |
|---|---|
| objective / tweedie_variance_power | tweedie / 1.5 |
| n_estimators / learning_rate | 800 / .03 |
| num_leaves / min_child_samples | 31 / 40 |
| subsample / subsample_freq | .8 / 1 |
| colsample_bytree | .8 |
| reg_lambda / reg_alpha | 1.0 / wrapper 기본0 |
| deterministic / force_col_wise / n_jobs | True / True / 4 |
| verbose / random_state | −1 / 해당 시드 |

원 fit은 sample_weight를 전달하지 않으므로 관측 가중치는 1이다. 운영특징 변경은 이 설정을 유지했으며 원 구성원 BASE14 또는 후보23 특징에서 학습한다. 최종 raw에서 LGB 계수는 .24이고 그 후 shrink .5·train min/max clip이다.

공식 objective는 raw score F에 대해 g=−y exp((1−ρ)F)+exp((2−ρ)F), h=−(1−ρ)y exp((1−ρ)F)+(2−ρ)exp((2−ρ)F)를 계산한다. 출력은 exp(F), 초기점은 log(가중 y 평균)을 Poisson 부모로부터 상속한다. Tweedie 클래스는 정확 잎 renewal을 override하지 않는다. 기본 IsRenewTreeOutput=false와 serial learner 조건 때문에 log(A/B) 갱신은 수행되지 않는다. [objective 4.7.0](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/objective/regression_objective.hpp#L676), [기본 interface](https://github.com/microsoft/LightGBM/blob/v4.7.0/include/LightGBM/objective_function.h#L51), [renew 조건](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/treelearner/serial_tree_learner.cpp#L874).

numeric leaf는 L1=0에서 −Σg/(Σh+λ2)를 사용한다. split gain도 같은 quadratic 근사를 사용하므로 leaf만 교체하는 것과 분할 기준까지 교체하는 것은 다른 실험이다. 구현의 epsilon·score_t 정밀도 때문에 아래 실수 수식과 비트 단위 동일성을 주장하지 않는다. [leaf 공식](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/treelearner/feature_histogram.hpp#L677).

새 tree에 learning rate가 한 번 곱해지고 raw score에 더해진다. 따라서 우리 설정은 F←F+.03·ηNewton, 최대 요청 반복800이다. split 가능한 잎이 없으면 먼저 중단될 수 있어 모든 fold가 정확800tree라고 소스만으로 단정하지 않는다. 최종 exp는 score 누적 후 적용된다. [GBDT 갱신](https://github.com/microsoft/LightGBM/blob/v4.7.0/src/boosting/gbdt.cpp#L384).

## 직접 미분: 부호와 정확 해

이하 유도는 위 g/h와 같은 손실 정규화에서 독립적으로 수행했다. 고정 tree 잎에 속한 **현재 iteration의 in-bag 관측**에 대해 w≥0, F는 갱신 전 raw score라 둔다.

L(F)=Σw[−y exp((1−ρ)F)/(1−ρ)+exp((2−ρ)F)/(2−ρ)].

ρ=1.5에서 A=Σw y exp(−F/2), B=Σw exp(F/2). 상수 increment t를 이 잎에 더하면

L(t)=2A exp(−t/2)+2B exp(t/2),
L′(0)=B−A=G, L″(0)=(A+B)/2=H.

따라서 현재 λ=1의 Newton잎은 ηN=(A−B)/[(A+B)/2+1]이다. A>B면 양의 score 갱신이며 exp(F)가 증가한다. gradient를 negative gradient로 착각해 이 부호를 뒤집으면 안 된다.

λ=0에서 정확 정상점은 −A exp(−t/2)+B exp(t/2)=0이므로 t*=log(A/B). A,B>0이면 strict convex이고 유일하다. 이 식은 [Yang·Qian·Zou 논문 Eq17–19, PDF page index10](https://arxiv.org/pdf/1508.06378)의 고정 잎 line search 및 νt* 갱신과 일치한다. 논문 Eq14의 negative-gradient 명칭/표기 부호는 직접 미분으로 판단한다. 해당 논문의 ν=.005와 우리 .03은 다르며 논문 그대로의 재현이라고 부를 수 없다.

**λ=1을 유지하는 정확 해는 log(A/B)가 아니다.** 현재 Newton denominator와 같은 sum-loss 정규화를 보존하려면 Lλ(t)=L(t)+λt²/2를 두고

−A exp(−t/2)+B exp(t/2)+λt=0

의 유일한 근을 찾아야 한다. 미분은 .5A exp(−t/2)+.5B exp(t/2)+λ>0이다. 근을 구한 후 .03을 한 번 곱한다. 실제 적용 increment .03t에 penalty를 먼저 걸거나 근과 적용 때 .03을 두 번 쓰는 것은 다른 최적화이다. A=0 또는 B=0의 경계에서는 log비가 유한하지 않으므로 λ0 공식을 맹목적으로 쓰지 않는다.

가중치를 모두 c배 하면 λ0 log비는 같지만 λ1은 effective λ=1/c가 된다. sum을 mean으로 정규화하고 λ1을 유지하면 정규화 강도가 바뀐다. Hessian/gradient loss 전체를 다른 상수배로 쓸 때도 λ를 함께 바꿔야 같은 식이다. 이번 EC에 보험 exposure 가중치나 φ 추정값을 임의로 도입하지 않는다. bagging80% 대신 모든 train 관측으로 A/B를 구하면 기존 sampling까지 바뀌므로 잎 계산 단일 대조가 아니다.

## 무엇을 말할 수 있는가

λ0에서는 ηN=2(A−B)/(A+B)=2 tanh(log(A/B)/2). log비가 0 부근이면 정확 해와 가까우나, 먼 경우 |ηN|<2이고 |log비|는 제한되지 않는다. 양·음 방향 모두의 갱신 크기가 다를 수 있으므로 고EC를 올리는 편향 개선 공식으로 해석하지 않는다. λ1은 Newton 크기를 추가로 줄이며 exactλ1과의 차이는 A/B뿐 아니라 A+B 크기에도 좌우된다.

합성 A=100,B=1에서 λ1 Newton=1.9223300971, exactλ1=4.1892496313이며 .03 갱신은 .0576699029 대 .1256774889다. A=1,B=100이면 두 값의 부호가 반대다. A=3,B=2에서는 .03 갱신 차이가 약 .00006766에 불과하다. 이는 단일 고정 잎 스칼라 예시이며 EC 개선 수치가 아니다. 유한차분 gradient, 이분법 exact근 residual<1e−10, λ0 closed form<1e−12, objective감소를 8합성 설정에서 대조했다. 근거 `LGB_잎계산_수식검산_v1.json`, 실제 데이터 읽기/fit/predict/score0.

boosting은 iteration마다 F·분할·in-bag가 달라지고 후속 tree가 보상할 수 있다. 현재800/.03이 수치갱신 부족인지, 입력/분할 제약인지, 통계적 정규화인지 아직 확인하지 않았다. 정확 training잎 최적화가 일반화 RMSE·혼합후 RMSE를 개선한다는 보장도 없다.

## 이전 ET/LOG partition 실험과 구별

기존 `ec_log_partition_mean_20261004_v1/run.py`는 ET를 log(y/b) 목표로 fit한 forest에서 각 tree의 고정 leaf에 ratio 산술평균을 넣고 bquery를 곱해 tree를 평균했다. `ec_log_partition_original_20261004_v1/run.py`는 같은 log ratio ET 분할에서 원y leaf 산술평균을 출력하고 bquery 복원을 제거했다. 원단위 고정상수 SSE 최적화, forest tree간 평균·역변환 순서를 다루는 실험이었다. 이번 점검은 **LGB의 sequential raw score increment와 Tweedie loss**이며 ET 원y평균이나 Jensen 역변환과 동일한 조작이 아니다. LGB leaf의 log(A/B)를 원EC 평균으로 치환하는 식도 성립하지 않는다.

원 ET 두 후속의 기각을 소급 변경하지 않는다. family21 운영특징 기각도 leaf loss 차이를 검증한 실험은 아니다. 후속 학습이 필요하다면 sampling/분할/regularization/learning-rate/멤버입력·혼합·후처리를 먼저 단일안으로 고정하고 새 사전등록해야 한다. 이번 파일은 후보 채택/제출 제안이 아니다.

공식 tag 소스를 웹에서 확인한 것이며 로컬 lib_lightgbm.dll의 build provenance를 역검증한 것은 아니다. 위 소스 대응 범위와 synthetic 수식만 확인했다.
