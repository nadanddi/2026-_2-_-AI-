# -*- coding: utf-8 -*-
import sys,json,hashlib,math
sys.dont_write_bytecode=True
from evaluate_task1 import H,ROOT,np,pd
import sklearn,lightgbm
r=json.loads((H/'validation_results.json').read_text(encoding='utf-8'))
a=json.loads((H/'causality_results.json').read_text(encoding='utf-8'))
d=json.loads((H/'f47_diagnosis.json').read_text(encoding='utf-8'))
labels={'unweighted':'무가중','weighted':'지정 가중','weighted_foldlocal':'폴드 내부 가중','weighted_vs_unweighted':'지정 가중 − 무가중','weighted_foldlocal_vs_unweighted':'폴드 내부 가중 − 무가중'}
groups={'all':'전체','clean':'복원 흔적 ±3시간 제외','clean_low_noise':'깨끗함+잡음점수 하위군'}
def ci(m):
    if m['days']<2:return '1일: 추론 구간 산출 불가'
    return f"[{m['ci95'][0]:+.5f}, {m['ci95'][1]:+.5f}]"
def row(tag,m):
    tag=tag.replace('|',' / ')
    if not m['n']:return f'| {tag} | 0 | 0 | — | — | — | 표본 없음 |'
    return f"| {tag} | {m['n']} | {m['days']} | {m['base']:.5f} | {m['new']:.5f} | {m['delta']:+.5f} | {ci(m)} |"
def table(entries):return '\n'.join(['| 비교·부분집합 | 행 | 온실-날 | 기준 RMSE | 혼합 RMSE | 차이 | 차이 95% 구간 |','|---|---:|---:|---:|---:|---:|---|']+[row(k,m) for k,m in entries])
# 각 폴드의 모든 요청 분할과 두 혼합 간 비교를 빠짐없이 보존.
full=['# 과제 1 상세 검증표','',f'재현 스크립트: `{H / "evaluate_task1.py"}`. 음수 차이는 개선. 혼합은 항상 기존 80%+독립 모델 20%.', '', '키: all=전체, clean=복원 흔적 ±3시간 제외, clean_low_noise=clean이고 noise_score<1.445. part1은 일차<179, part2는 일차≥179. cold10은 해당 날 ph_in_temp_3 최솟값<10. 기본 비교의 기준은 F60ND; vs_unweighted 비교의 기준은 무가중 20% 혼합이다.','', '1일 블록의 부트스트랩은 퇴화하므로 아래 표에는 추론 구간을 표시하지 않는다. JSON에는 기계적으로 계산된 동일한 양 끝 값이 남지만 유효한 신뢰구간으로 해석하지 않는다.']
metric_cells=0
for name,obj in r['evaluations'].items():
    views=[('전체 검증행 합산',obj['aggregate'])]+[(f"폴드 {f['fold']} (학습 {f['train_rows']}행 / 검증 {f['val_rows']}행)",f['metrics']) for f in obj['folds']]
    for desc,met in views:
        full.extend(['',f'## {name} — {desc}'])
        for v,scores in met.items():
            full.extend(['',f'### {labels[v]}','',table(list(scores.items()))])
            metric_cells+=len(scores)
(H/'검증표_전체.md').write_text('\n'.join(full),encoding='utf-8')
mainentries=[]
for name,obj in r['evaluations'].items():
    for g in groups:
        for v in ['unweighted','weighted','weighted_foldlocal']:
            mainentries.append((name+' / '+groups[g]+' / '+labels[v],obj['aggregate'][v][g+'|all']))
strat=[]
for g in groups:
    for sub in ['F13_part1','F13_part2','F47_part1','F47_part2','hour00_03','hour04_23']:
        for v in ['unweighted','weighted']:
            strat.append((groups[g]+' / '+sub+' / '+labels[v],r['evaluations']['DIAG10']['aggregate'][v][g+'|'+sub]))
pairentries=[(n+' / '+groups[g],o['aggregate']['weighted_vs_unweighted'][g+'|all']) for n,o in r['evaluations'].items() for g in groups]
u=d['variants']['unweighted']; w=d['variants']['weighted']
dw={a['day']:a for a in w['days']}
daylines=['| F47 일차 | 기준 | 무가중 혼합 | 지정 가중 혼합 | 무가중 ΔSSE | 0~3시 ΔSSE | 4~23시 ΔSSE | 일평균 실내℃ | 일최소 실내℃ | ph3 최솟값 | 자정 점프℃ | 난방 평균 | 잡음 표지 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for z in u['days']:
    j='관측 불가' if not math.isfinite(z['jump']) else f"{z['jump']:+.2f}"
    daylines.append(f"| {z['day']} | {z['rmse_base']:.4f} | {z['rmse_mix']:.4f} | {dw[z['day']]['rmse_mix']:.4f} | {z['delta_sse']:+.3f} | {z['early_delta_sse']:+.3f} | {z['late_delta_sse']:+.3f} | {z['in_temp_mean']:.2f} | {z['in_temp_min']:.2f} | {z['ph3_min']:.2f} | {j} | {z['heat_mean']:.2f} | {'있음' if z['noisy'] else '없음'} |")
hourentries=[(f"{h:02d}시 / {labels[v]}",d['variants'][v]['hours'][str(h)]) for h in range(24) for v in ['unweighted','weighted']]
(H/'F47_날별_시간별.md').write_text('# F47 2차 구간 상세 진단\n\n무가중 혼합의 오차제곱합 증가 순서. 잡음 표지는 noisy_days()의 cold 예외를 포함한다. 모든 행은 학습·OOF 진단이며 평가 정답을 사용하지 않았다. 자정 점프는 정확히 1시간 전 입력이 존재하고 결측이 아닐 때만 산출한다.\n\n'+'\n'.join(daylines)+'\n\n## 시간별 비교\n\n'+table(hourentries)+f'\n\n재현: `{H / "diagnose_f47.py"}`\n',encoding='utf-8')
meta={'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,'feature_module':'resid_reset_features.py','feature_count':89,'physics_count':19,'mix':.2,'ridge':{'alpha':100,'imputer':'median on training fold','scaler':'StandardScaler on training fold','weights':'ridge__sample_weight'},'lightgbm':{'n_estimators':220,'learning_rate':.035,'num_leaves':12,'max_depth':-1,'min_child_samples':100,'reg_lambda':15,'n_jobs':4,'random_state':726,'objective':'regression (default)'},'bootstrap':{'reps':2000,'seed':726,'unit':'farm-day'}}
(H/'model_configuration.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
weighted=r['evaluations']['DIAG10']['aggregate']['weighted']['all|all']
f47entries=[]
for nm,alias in [('all','全体'.replace('全体','전체')),('cold10','추운 날'),('not_cold10','그 밖의 날'),('clean_low_noise','깨끗함+잡음 하위군')]:
    for v in ['unweighted','weighted']:f47entries.append((alias+' / '+labels[v],d['variants'][v]['subsets'][nm]))
foldwins={v:sum(f['metrics'][v]['all|all']['delta']<0 for f in r['evaluations']['DIAG10']['folds']) for v in ['unweighted','weighted','weighted_foldlocal']}
report=f'''# 과제 1 — resid_reset 20% 혼합 후속 검증

작성일: 2026-09-26. 요청서: `research/Codex_요청_2차_2026-09-26.md`. 모든 새 파일은 이 보고서와 같은 `analysis/codex_independent/2차/` 안에 저장했다.

## 1. 제출 후보 판단

**유력: 지정 가중치로 학습한 resid_reset을 20% 섞는 방법은 제출 후보로 검토할 근거가 있다.** DIAG10 전체 RMSE가 **0.69654 → 0.68684**, 상대 **{100*(1-weighted['new']/weighted['base']):.2f}% 개선**, 차이 95% 구간은 **{ci(weighted)}**였다. 깨끗하고 잡음 점수가 낮은 행에서는 **0.51825 → 0.50048**이다. EXT10/12에서도 개선이 이어졌다. 비중·트리 수·시드는 새로 고르지 않았다.

**확실: 무가중치 혼합을 먼저 교체할 이유가 전체 평균만으로 입증된 것은 아니다.** 무가중 0.68733에서 지정 가중 0.68684로의 추가 차이는 −0.00049, 95% 구간 [−0.00294, +0.00190]이다. 다만 잡음 하위군과 EXT12의 추가 개선, F47 2차 구간 악화 완화는 가중 버전을 검토할 이유다.

**유력: F47 2차 구간의 1차 악화는 추운 날 전반의 구조적 실패로 보이지 않는다.** 무가중 악화는 27일 중 18일에서 발생했고, 189·241·242일에 양의 ΔSSE의 51.4%가 모였다. 세 날은 지정 ph3<10℃ 기준의 추운 날에 속하지 않는다. 가중 적용 후 전체 F47 2차 구간은 **1.03779 → 1.03712**로 거의 같아졌으며, 구간 [−0.01128, +0.00986]은 0을 포함한다. 우연인지 특정 출처·관리 상태 때문인지 확정할 수는 없다.

**확실: 새 피처의 강화 감사 25건은 모두 통과했다.** 모델에 들어가는 89개 피처를 float64 비트로 비교했다. 실제 리더보드 개선을 검증한 것은 아니며, 이번에도 제출 CSV를 만들지 않았다.

## 2. 정확히 무엇을 비교했나

### 모델과 가중치

비교 기준은 제공된 `eval_v6_oof.npz`의 `F60ND__DIAG10`, `F60ND__EXT8/10/12`다. 기준 모델은 재학습하지 않았다. 모든 후보는 **기준 80% + 독립 예측 20%**이며, 시간·온실에 따른 혼합 비중 변경도 하지 않았다.

| 항목 | 고정 설정 |
|---|---|
| 입력 피처 | 당일 현재까지의 89개, 선형 기준선 19개 |
| Ridge 전처리 | 학습 폴드 중앙값 대체, StandardScaler; 각각 학습 폴드에서만 적합 |
| Ridge | alpha=100; 지정 버전은 `ridge__sample_weight` 적용 |
| LightGBM 목표 | 해당 학습 폴드의 정답 − 해당 Ridge의 학습 예측 |
| 트리·학습률 | 220, 0.035 |
| 잎 수·최소 잎 표본 | 12, 100 |
| 최대 깊이·L2 | −1, 15 |
| 난수 시드·스레드 | 726, 4 |
| 목표함수 | 기본 회귀 제곱오차 |
| 가중치 적용 | Ridge 목적함수와 LightGBM 양쪽. 대체값·스케일 계산 자체에는 가중치를 주지 않음 |
| 혼합 | 0.8×기준 + 0.2×독립 모델 |

세 버전을 고정 비교했다.

1. **무가중**: 1차와 같은 설정. 새 모듈과 1차 모델 입력은 비트 단위로 일치했고, 다시 학습한 DIAG10 예측과 1차 저장값의 최대 절댓값 차이는 **{r['unweighted_oof_max_abs_difference']:.1f}**이었다.
2. **지정 가중**: 원본 `train_flags_v6.row_weights(lab, 0.2, w_noisy=0.2)`를 그대로 호출. 전체 원본 학습 입력으로 계산한 현재 평가 코드와 같은 가중치 목록을 각 폴드 학습행에 적용했다.
3. **폴드 내부 가중(보조)**: 함수를 수정하지 않고 메모리에서 학습 입력 공급만 해당 폴드 학습행으로 제한하여 재계산했다. 검증일과 완충일의 입력도 잡음 순위 계산에서 제외한다. 적용 뒤 원래 함수를 복구했다.

지정 가중은 복원 흔적과 잡음이 겹치면 **0.2×0.2=0.04**다. 실제 9,600행에서 0.04가 23행, 0.2가 2,318행, 1이 7,259행이었다. 전부 0.2로 단순 처리하지 않았다.

지정 버전의 원본 학습 입력에는 CV 검증일 입력도 포함되므로 잡음 순위가 엄밀한 폴드 내부 전처리는 아니다. **실제 test_X 분포를 사용하는 것은 아니지만 CV 해석에는 한계**가 있어 세 번째 버전을 병기했다. 기준 OOF 자체는 같은 전역 학습 목록 방식이므로, 보조 비교 역시 전체 파이프라인을 완전히 새로 중첩 검증한 결과는 아니다. 두 가중 버전의 DIAG10·EXT10은 비슷했고, EXT12에서는 차이가 커졌다.

설정·버전 재현: `{H / 'model_configuration.json'}`. 실행 환경은 numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}, lightgbm {lightgbm.__version__}.

### 폴드와 채점 집합

- DIAG10: 온실별 학습일을 정렬하여 5일씩 10개 폴드에 순환 배치, 전후 1일 완충. 각 행은 한 번 검증된다.
- EXT8/10/12: 기존 `features_v4.phys_features()`의 ph_in_temp_3 일 최솟값이 각 온도 미만인 날 전체를 검증하고 전후 하루는 학습에서 제외. 저장된 기준 OOF의 유효 행과 새 예측의 유효 행이 정확히 같음을 확인했다.
- **전체**: DIAG10 기준 9,600행, 400일.
- **깨끗한 행**: `row_weights(lab, 0.0, radius=3)>=1` 그대로. 9,403행. 거리≤3시간을 제외하므로 실제 조건은 복원 흔적에서 3시간 초과 떨어진 행이다.
- **깨끗함+잡음 하위군**: 위 깨끗한 행이면서 `noisy_days().noise_score < 75% 분위수`인 날. 분위수는 **{r['noise_threshold']:.3f}**, 총 7,073행·299일이다. 경계 동점 때문에 상위군은 정확히 100일이 아니라 **101일**이다. `noisy`는 저온 예외를 적용해 96일이므로 단순 `~noisy`와 이 조건은 다르다.

잡음 하위군이 평가와 유사하다는 가설은 요청서의 해석이며, 이 부분집합 점수가 실제 평가 점수의 불편 추정량이라고 보장하지 않는다. 평가는 정답을 보지 않았고 평가 분포에 맞춘 조건 조정도 하지 않았다.

## 3. DIAG10·저온 외삽 결과

모든 표에서 차이=혼합−기준. 신뢰구간은 온실-날을 짝지어 2,000회 복원추출, 시드 726. 개별 행 부트스트랩을 사용하지 않았다.

{table(mainentries)}

**해석:** 지정 가중 버전은 네 검증 조건의 전체·깨끗함·잡음 하위군에서 모두 점 추정치가 개선되었다. EXT8은 전체 11일, 잡음 하위군 6일이어서 구간이 넓으며 개선을 확정하기 어렵다. EXT8/10/12는 날이 겹치므로 독립적인 세 번의 확증이 아니다.

### 가중치만의 추가 효과

아래 표에서 기준은 **무가중 20% 혼합**, 비교값은 **지정 가중 20% 혼합**이다.

{table(pairentries)}

### 온실×구간 및 시간대

DIAG10의 요구 분할을 아래에 모두 제시한다. EXT 각 조건과 DIAG10 **개별 폴드 1~10**의 동일 분할, 폴드 내부 가중과 두 혼합 간 차이는 `{H / '검증표_전체.md'}`에 빠짐없이 수록했다. 원시 수치는 `validation_results.json`이다.

{table(strat)}

전체행 기준 개선 폴드는 무가중 {foldwins['unweighted']}/10, 지정 가중 {foldwins['weighted']}/10, 폴드 내부 가중 {foldwins['weighted_foldlocal']}/10이다. 이 역시 이미 사용했던 DIAG10의 재검증이며 미사용 외부 홀드아웃으로 부르면 안 된다.

## 4. F47 2차 구간 악화의 위치와 원인

{table(f47entries)}

무가중 모델의 전체 ΔSSE는 **{u['all_delta_sse']:+.3f}**다. 시간대로 나누면 0~3시는 **{sum(z['early_delta_sse'] for z in u['days']):+.3f}**로 개선, 4~23시는 **{sum(z['late_delta_sse'] for z in u['days']):+.3f}**로 악화였다. 따라서 이 구간의 악화를 전체적으로 자정 오차 문제라고 설명하면 맞지 않는다. 가중 버전은 초기 ΔSSE **{sum(z['early_delta_sse'] for z in w['days']):+.3f}**, 그 밖의 시간 **{sum(z['late_delta_sse'] for z in w['days']):+.3f}**였다.

무가중 악화 상위 세 날:

- **189일**: 실내 평균 18.91℃, ph3 최솟값 14.60℃, 잡음 표지 있음. 기준 RMSE 2.1509 → 2.2002. 음의 편향이 −2.1259 → −2.1790으로 커졌다. ΔSSE +5.149 중 초기 +3.196, 이후 +1.953. 자정 점프는 결측으로 판단 불가다.
- **241일**: 실내 평균 14.78℃, ph3 최솟값 11.15℃, 자정 +3.70℃, 잡음 표지 있음. RMSE 1.8588 → 1.8926. 양의 편향이 더 커져 189일과 오차 방향은 반대다.
- **242일**: 실내 평균 13.63℃, 최소 9.10℃지만 ph3 최솟값 10.009℃로 cold10 조건에는 조금 못 미침. 자정 −1.90℃, 잡음 표지 있음. ΔSSE +2.167 중 +2.036이 4~23시에 발생했다.

전체 27일의 무가중 양의 ΔSSE 중 이 세 날의 비중은 **51.4%**다. 그러나 **209일은 잡음 하위군**인데도 0.1973 → 0.3427로 나빠졌고, 대부분이 4~23시의 양의 편향 증가였다. 즉 “잡음 날만의 실패”도 아니다. 가중 적용으로 잡음 하위군의 평균 악화는 사라졌지만 12일 표본의 불확실성은 크다.

상위 3일을 사후 제외하면 무가중 차이는 −0.00305 [−0.01463, +0.01311]로 부호가 바뀐다. **이는 영향도 진단이지 제출 성능 추정이 아니다.** 실제 검증에서 그 날들을 제거해 좋은 점수를 선택하지 않았다.

### 추운 2차 구간에서 반복되나

- DIAG10 F47 2차 cold10의 8일에서는 무가중 −0.00222, 지정 가중 −0.00085로 모두 거의 같고 구간은 0을 포함한다.
- EXT10의 F47 2차 8일에서는 기준 0.91342 → 무가중 0.88614 / 지정 가중 0.89599. 두 구간은 0을 포함한다.
- EXT12의 F47 2차 17일은 무가중 0.99000으로 악화하지만, 지정 가중은 **0.98056 → 0.94308**, 차이 [−0.07139, −0.01153]이다. 폴드 내부 가중은 0.96513, 차이 [−0.04569, +0.00678]로 개선 점 추정이나 불확실하다.
- EXT8에는 F47 2차 행이 없어 이 조건으로 판단할 수 없다. EXT10의 F47 2차·깨끗함·잡음 하위군은 **1일**뿐이므로 신뢰구간 추론이 불가능하다.

**원인 판정: 확실한 관측은 소수 큰 오차 날, 낮 시간 악화, 잡음 가중치에 대한 민감도다. 출처별 관리·센서 또는 온도 반응 차이라는 구조적 설명은 추측이다. “F47의 추운 2차 구간에서 항상 나빠진다”는 설명은 이번 결과와 맞지 않는다.**

날별 27일 전체 표와 시각 0~23시의 paired CI: `{H / 'F47_날별_시간별.md'}`. 입력 특징·ΔSSE·상관의 기계 판독값: `f47_diagnosis.json`. 상관은 3버전×5개 변수=15개의 사후 탐색이며 인과 효과가 아니다.

## 5. 강화된 인과성 검사

`resid_reset_features.py`의 모든 모델 피처 89개와 행 식별·메타데이터를 검사했다.

| 검사 | 횟수 | 결과 |
|---|---:|---|
| F13/F47 × 30/220일 × 0/5/7/13/23시 이후 입력 ×10+잡음+약33% 행 결측 | 20 | 모두 비트 일치 |
| 다른 온실 입력 전체 교란 | 2 | 해당 온실 전체 행 비트 일치 |
| train_y 두 목표값 무작위 재배열 | 1 | 전체 피처 비트 일치 |
| 실제 CSV 경로 입력과 DataFrame 입력 일치 | 1 | 전체 피처 비트 일치 |
| test_X 및 train_y 교란 후 지정 학습 가중치 | 1 | 가중치 비트 일치 |

소수점 허용오차 비교가 아니라 float64를 uint64로 재해석하여 **NaN 비트 패턴과 +0/−0까지 동일**한지 비교했다. 미래 행은 그대로 두고 값만 ×10, 평균0·표준편차5의 잡음, 약33% 행 결측을 함께 적용했다. train_y 호환 인자는 모듈에서 아예 읽지 않으므로 독립성이 구조적으로도 보장된다.

평가행을 포함한 반환 행은 {a['feature_rows']:,}개, 실제 test_X {a['test_rows']:,}개 모두 포함된다. 일 누적 통계는 같은 온실·같은 날 현재까지이며, 전날 입력·이웃 정답·전체 평가 분포를 독립 모델 피처에 넣지 않는다. 학습 가중치의 전역 순위와 채점용 사후 그룹은 예측 피처와 분리했다. 감사가 모든 가능한 버그의 증명은 아니지만 요청한 교란 시험은 모두 통과했다.

재현: `{H / 'audit_causality.py'}`. 세부 절단점·검사 행 수: `causality_results.json`.

## 6. Claude가 가져갈 피처 함수

**함수 모듈:** `{H / 'resid_reset_features.py'}`

- `build_features(train_X_csv, test_X_csv, train_y_csv=None)`를 제공한다.
- 첫 두 인자는 원본 CSV 경로 또는 DataFrame이다. 세 번째 인자는 감사 호환용이며 읽지 않는다.
- 반환값은 row_id별 DataFrame. F13/F47만 포함하고 온실·시각순으로 정렬한다.
- `FEATURE_COLUMNS`는 독립 잔차 모델 89개, `PHYSICS_COLUMNS`는 선형 기준선 19개다. 메타데이터까지 통째로 모델에 넣지 않는다.
- 0시 입력 결측은 나중 시각 값으로 채우지 않는다. 누적 표준편차의 초기 결측은 정상이며 Ridge는 학습 중앙값, LightGBM은 자체 결측 처리를 쓴다.
- 입력 파일·폴더에 어떤 파일도 쓰지 않는다. pandas/numpy만 필요하며 모델 학습이나 제출 저장을 자동 실행하지 않는다.

```python
# 기존 파이프라인의 env 초기화 및 이 모듈 폴더 경로 등록 후
from resid_reset_features import build_features, FEATURE_COLUMNS, PHYSICS_COLUMNS
features = build_features(train_X_path, test_X_path)
# row_id로 학습 정답 또는 원래 test_X 순서에 정렬
# 선형 기준선과 잔차 모델 모두 동일한 학습 가중치 적용
# 최종 온도 = 0.8 * 기존_온도예측 + 0.2 * 새_온도예측
```

가중 회귀의 완전한 재현 구현은 `evaluate_task1.py`의 `model_predict`를 참고한다. 피처 모듈은 동일 코드이며 제출 통합시 가중치와 두 모델의 학습 단계까지 동일해야 검증 결과와 비교할 수 있다. 패키지 버전이 달라지면 비트 동일 재현을 별도로 확인해야 한다.

## 7. 다중비교·실패·남은 불확실성

- 2개의 요청 버전과 1개의 가중치 계산 민감도 버전, 총 **3개 설정**을 비교했다. 13개 학습/검증 분할에서 각각 학습해 Ridge 39회, LightGBM 39회다. 비중은 20% 하나, 시드도 726 하나이며 최적화 탐색은 하지 않았다.
- 집합별·폴드별·부분집합별·혼합 간 차이까지 상세표는 **{metric_cells}개 셀**을 포함한다. 중복 합산·빈 셀도 들어 있으며 독립적인 실험 수가 아니다. 다중비교 보정은 하지 않았다. 특히 1차에서 선택한 모델을 같은 DIAG10으로 다시 평가했으므로 선택 편향은 남는다.
- CI는 저장된 예측을 고정한 날 부트스트랩이다. 모델 재학습 변동, 같은 달력 날짜를 공유하는 출처, 인접 날 의존성을 모두 반영하지 않는다. 1일 표본의 기계적 구간은 상세표에서 추론 불가로 표시했다.
- 실패/미입증: 무가중 F47 잡음 하위군 악화, EXT8의 작은 표본, 지정 가중치의 DIAG10 전체 추가 이득 미확정. 가중 버전도 F47 2차 일부 날에서는 여전히 악화한다.
- 잡음 가중치는 관측 입력에 대한 휴리스틱이며 진짜 잡음 정답이 아니다. 저온 예외와 75% 경계 동점을 반영했지만 오류가 있을 수 있다.
- 과제 2의 가설을 과제 1 모델에 추가하지 않았다. EC 모델도 바꾸지 않았다. 제출 CSV·커밋·푸시는 없다.

## 8. 파일·재현·보존 확인

수치 생성 스크립트는 다음과 같다.

| 스크립트 | 생성 결과 |
|---|---|
| `{H / 'evaluate_task1.py'}` | validation_results.json, validation_predictions.npz, validation_rows.json, original_hashes.json |
| `{H / 'diagnose_f47.py'}` | f47_diagnosis.json |
| `{H / 'audit_causality.py'}` | causality_results.json |
| `{H / 'write_task1_report.py'}` | 이 보고서, 검증표_전체.md, F47_날별_시간별.md, model_configuration.json |

제공된 Python 실행파일로 `-B` 옵션을 붙여 평가 → 진단 → 감사 → 보고서 순서로 실행한다. 기존 research 모듈의 바이트코드 생성을 막았다. 주요 원본 CSV, 가중치·감사·평가 코드, 1차 분석 코드·보고서, .gitignore의 SHA-256을 작업 전후 대조했다. 원본 파일과 1차 결과는 수정하지 않았다. NPZ는 학습 검증행 예측으로 제출용 파일이 아니다.
'''
(H/'보고서_과제1.md').write_text(report,encoding='utf-8')
hashes=json.loads((H/'original_hashes.json').read_text(encoding='utf-8'))
checks={p:hashlib.sha256(__import__('pathlib').Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items()}
assert all(checks.values())
(H/'preservation_check.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('reports saved',len(report),'metric cells',metric_cells,'original files unchanged',len(checks))

