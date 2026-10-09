"""Report fixed-model diagnostics without tuning or model selection."""
import runtime_v1
import json, math
import pandas as pd
import numpy as np
import joblib
import run_v1 as P
import recipe_v1 as C
s=json.loads((P.H/'final_score_v1.json').read_text(encoding='utf8'))
m=json.loads((P.L/'model_full_v1/model_manifest_v1.json').read_text(encoding='utf8'))
g=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted((P.L/'cv_predictions').glob('*_v1.csv'))],ignore_index=True)
segments=[]
for (v,farm,p2),z in g[g.hour==15].assign(pass2=g[g.hour==15].day>=179).groupby(['validator','farm','pass2']):
 segments.append(dict(validator=v,farm=farm,pass2=bool(p2),**P.metrics(z.high,z.ensemble)))
gap=[]
for v,z in g.groupby('validator'):
 values=[]
 for p in sorted((P.L/'cv_predictions').glob(v+'*_v1.json')):
  rr=json.loads(p.read_text(encoding='utf8'))
  values.extend(r['training_brier'] for r in rr['receipts'] if r['kind']=='et')
 gap.append(dict(validator=v,mean_training_brier_over_fold_members=float(np.mean(values)),validation_brier_all_hours=float(np.mean((z.ensemble-z.high)**2)),caution='train is member mean, validation is ensemble; all 24 prefixes repeated within each day, not independent samples'))
importance=[]
for rec in m['models']:
 model=joblib.load(P.L/'model_full_v1'/rec['file']);importance.append(model.named_steps['clf'].feature_importances_)
imp=pd.DataFrame({'feature':C.COLS,'importance':np.mean(importance,axis=0)}).sort_values('importance',ascending=False)
P.csvnew(P.H/'feature_importance_v1.csv',imp)
P.save(P.H/'diagnostics_v1.json',dict(segments=segments,overfit=gap,importance=imp.head(10).to_dict('records'),reliability=s['reliability'],seed_brier=[t for t in s['groups'] if t['hour']==15 and t['scope']=='all']))
lines=['# 일반일·고EC일 분류 모델 실험 결과', '', '고EC 기준은 사용자가 확정한 일평균 sub_ec ≥ 1.2다. 해당 시간의 순간 EC가 아니라 당일 24시간 평균의 고EC 여부를 예측한다. 원자료의 고EC일을 학습·검증에서 삭제하지 않았다.', '', '## 생성 모델', '', f"ExtraTrees 3시드 평균 분류기. 최종 학습 {m['training_days']}일/{m['training_rows']}행, 일반 {m['ordinary_days']}일·고EC {m['high_days']}일. 같은 농장의 당일 0..현재 h 입력에서 73특징을 계산한다. 현재/h0/누적평균/관측0비율과 운전 일정, 팬·차광·보온 낮밤대비를 포함하며 정답·기록번호 자체·미래·다른 온실 입력은 제외한다. .5 분류 임계값과 보조 .2 임계값은 실행 전 고정했다. 점수는 보정된 확률이 아니다.", '', '## 고정 검증 결과', '', '공개 360일은 일반334일·고EC26일이다. 뒤 구간46일은 일반41일·고EC5일이다. DIAG10 전체360일, FRESH7과 EL1은 동일한 뒤 구간46일의 다른 묶음이며 독립 복제라고 해석하지 않는다. 농장×일별24행을 함께 제외하고 같은 농장±1/다른 농장±3 및 소비40일 주변을 학습에서 제외했다. imputer와 모델은 fold 학습에서만 fit했다.', '', '|검증|입력 시각|일수|고EC|탐지 TP|놓침 FN|오탐 FP|정밀도|재현율|ROC-AUC|AP|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for t in s['groups']:
 if t['arm']=='ensemble' and t['scope']=='all':
  lines.append(f"|{t['validator']}|{t['hour']}시|{t['days']}|{t['high_days']}|{t['tp']}|{t['fn']}|{t['fp']}|{t['precision']:.1%}|{t['recall']:.1%}|{t['roc_auc']:.4f}|{t['average_precision']:.4f}|")
lines += ['', '## 사전 기준과 해석', '', f"15시 prior 대비 유용성 조건 지원: {s['prior_usefulness_supported']}. 이 조건은 고EC 비율만 내놓는 기준보다 Brier가 나은지에 관한 진단이다. EC 회귀 오차 개선이나 기존 회귀 후보 채택을 뜻하지 않는다.", '', '|15시 검증|상수 prior Brier|로지스틱 Brier|ET 평균 Brier|', '|---|---:|---:|---:|']
for v in ['DIAG10','FRESH7','EL1']:
 d={t['arm']:t['brier'] for t in s['groups'] if (t['validator'],t['hour'],t['scope'])==(v,15,'all')}
 lines.append(f"|{v}|{d['prior']:.6f}|{d['logit']:.6f}|{d['ensemble']:.6f}|")
lines += ['', '농장×기록번호5일 블록 20,000회 bootstrap(기존 공개자료, 교환가능성 근사):']
for b in s['bootstrap']:lines.append(f"- DIAG 15시 {b['comparator']} 대비 Brier 차이 CI95={b['delta_brier_ci95']}, p_worse={b['p_worse']:.5f}.")
lines += ['', '고정 .2 민감도 점검(최적 임계값으로 선택하지 않음):']
for t in s['groups']:
 if t['arm']=='ensemble' and t['scope']=='all' and t['hour']==15:
  z=t['threshold02'];lines.append(f"- {t['validator']}: TP{z['tp']}/FN{z['fn']}/FP{z['fp']}, 정밀도{z['precision']:.1%}, 재현율{z['recall']:.1%}.")
lines += ['', '## 과적합·구간 차이·점수 보정', '', 'training과 validation의 아래 Brier는 모든24시각 기준이다. training은 각 ET 구성원 평균, validation은 3구성원 평균이므로 완전히 같은 추정량은 아니다. 일 내24행을 독립일로 보지 않는다. 작은 training 오차는 미래 일반화의 증거가 아니다.']
for t in gap:lines.append(f"- {t['validator']}: 학습 Brier 평균 {t['mean_training_brier_over_fold_members']:.6f}, 검증 {t['validation_brier_all_hours']:.6f}.")
for t in segments:lines.append(f"- {t['validator']} {t['farm']} 뒤구간={t['pass2']}: {t['days']}일/고EC{t['high_days']}, TP{t['tp']} FN{t['fn']} FP{t['fp']}.")
lines += ['', '고정 점수 bin별 실제 고EC 비율은 final_score_v1.json/diagnostics_v1.json에 저장했다. calibration fit이나 사후 threshold 튜닝은 하지 않았다. 나무 불순도 중요도 상위: '+', '.join(imp.head(5).feature)+'. 상관된 특징끼리 중요도가 분산되고 학습자료를 사용한 지표라 물리 원인이나 독립 추가정보로 해석하지 않는다.', '', '## 최종 검증과 재현', '', '계획→첫2fold 중간→전체/저장모델의 독립 비평. 원자료 기반73특징/일평균라벨/분할을 독립 재현하고, 혼동행렬·AUC·AP·Brier·logloss·bootstrap을 별도 구현으로 재계산했다. 최종 causality_verification_v1.json은 미래·다른 온실 변조와 허용 prefix만 남기는 8사례, 입력행 순서변경, 재로딩, 코드·모델 SHA를 점검한다. 최종 independent critic 결과는 critique_final_v1.md를 함께 확인한다.', '', '모든27fold 점수 저장 후에만 공식400일 정답을 최종 학습에 읽었다. 소비40일은 최종 fit에만 쓰고 예측 성능을 다시 측정하지 않았다. 검증은 이미 여러 탐색에 노출된 자료이며 새 홀드아웃 성능은 없다. 뒤 구간 고EC5일이라는 작은 분모 때문에 성능 확신은 제한된다. 설명 가능/불가능 고EC 구분이나 진짜/가짜 판정 모델이 아니다. 수치 EC 보정·회귀 채택·제출물은 만들지 않았다.', '', '재현: 새 버전 폴더를 만들어 PLAN/코드를 복사한 뒤 run_v1.py prepare → first → 독립중간검토 → rest → finalfit → causality_verification_v1.py. 기존 등록/산출물은 덮어쓰기를 차단하므로 같은 폴더 재실행은 불가하다. predict_v1.py --model <model_full_v1> --input <row_id+RAW14 CSV> --output <새 CSV>로 추론한다. 각 날은0시부터현재까지연속입력이 필요하다. 출력: row_id, high_ec_score, high_ec_class. 공식test는 실행하지 않았다.']
p=P.H/'report_v1.md';assert not p.exists();p.write_text('\n'.join(lines)+'\n',encoding='utf8')
print('REPORT_AND_DIAGNOSTICS_CREATED',flush=True)
