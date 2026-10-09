"""Fresh counts and a reproducibility check for the frozen one-class CV."""
from pathlib import Path
import sys,json,csv,math
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.path.insert(0,str(H));import run_v2 as R
P=R.P;L=R.L
s=json.loads((H/'final_score_v1.json').read_text(encoding='utf8'));assert s['status']=='FULL_CV'
with (L/'final_h15_v1.csv').open(encoding='utf8') as f:rows=list(csv.DictReader(f))
fresh=[]
for v in ['DIAG10','FRESH7','EL1']:
 for scope in ['all','pass2']:
  q=[r for r in rows if r['validator']==v and (scope=='all' or int(r['day'])>=179)]
  for arm,col in [('oneclass','oneclass_high'),('et','et_high')]:
   tp=sum(int(r['high'])==1 and int(r[col])==1 for r in q);fn=sum(int(r['high'])==1 and int(r[col])==0 for r in q);fp=sum(int(r['high'])==0 and int(r[col])==1 for r in q);tn=sum(int(r['high'])==0 and int(r[col])==0 for r in q)
   m=next(m for m in s['groups'] if (m['validator'],m['scope'],m['hour'],m['arm'])==(v,scope,15,arm));assert [tp,fn,fp,tn]==[m[k] for k in ['tp','fn','fp','tn']]
   recall=tp/(tp+fn);fpr=fp/(fp+tn);precision=tp/(tp+fp) if tp+fp else 0.;assert abs(recall-m['recall'])<1e-12 and abs(fpr-m['ordinary_false_positive_rate'])<1e-12 and abs(precision-m['precision'])<1e-12
   fresh.append(dict(validator=v,scope=scope,arm=arm,days=len(q),tp=tp,fn=fn,fp=fp,tn=tn,recall=recall,fpr=fpr,precision=precision))
reg=R.check();meta=R.pd.read_csv(P.L/'metadata_public_v1.csv');X=R.pd.read_csv(P.L/'features_public_v1.csv',float_precision='round_trip')[R.C.COLS];r=reg['folds'][0]
model=R.Pipeline([('imputer',R.SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',R.StandardScaler()),('svm',R.OneClassSVM(**R.PARAM))])
with R.threadpool_limits(limits=2):model.fit(X.iloc[r['train_indices']]);score=model.decision_function(X.iloc[r['query_indices']]);pred=(model.predict(X.iloc[r['query_indices']])==1).astype(int)
saved=R.pd.read_csv(L/'predictions'/f"{r['key']}_v1.csv",float_precision='round_trip');assert saved.row_id.tolist()==meta.iloc[r['query_indices']].row_id.tolist();gap=float(R.np.max(abs(score-saved.oneclass_score.to_numpy())));assert gap<1e-12 and R.np.array_equal(pred,saved.oneclass_high)
P.save(H/'fresh_checks_v1.json',dict(status='PASS',h15_scalar_checks=fresh,refitted_repro_fold=r['key'],score_maxdiff=gap,prediction_match=True,extra40_rescored=False))
allrows=R.pd.concat([R.pd.read_csv(p,float_precision='round_trip') for p in sorted((L/'predictions').glob('*_v1.csv'))],ignore_index=True);overfit=[]
for v,z in allrows.groupby('validator'):
 receipts=[json.loads(p.read_text(encoding='utf8')) for p in sorted((L/'predictions').glob(v+'*_v1.json'))];train_fraction=sum(m['training_inlier_fraction_all24h']*m['train_rows'] for m in receipts)/sum(m['train_rows'] for m in receipts);high=z[z.high==1]
 overfit.append(dict(validator=v,training_acceptance_all24h=train_fraction,validation_high_acceptance_all24h=float(high.oneclass_high.mean()),validation_high_days=high[['farm','day']].drop_duplicates().shape[0],caution='repeated training days and hourly prefixes; descriptive rates, not independent samples'))
P.save(H/'acceptance_gap_v1.json',overfit)
lines=['# 고EC 전용 단일클래스 탐지의 일반일 오탐 검증', '', '고EC날만 학습한 OneClassSVM으로 일반일을 포함해 검증했다. 고EC는 당일24시간평균sub_ec>=1.2이며,현재시간순간EC기준이아니다. 입력은 같은농장당일0..현재h의73특징이다. imputer/scaler/SVM은해당fold고EC행만fit,일반일학습0. RBF nu=.1/gamma=scale 고정, native+1/decision>0를고EC로판정하고튜닝하지않았다. 경계점수는확률이아니다.', '', '## 15시 기준 탐지와 오탐', '', '|검증|모델|고EC수|탐지|놓침|일반일수|오탐|오탐률|정밀도|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for m in s['groups']:
 if m['hour']==15 and m['scope']=='all' and m['arm'] in ['oneclass','et']:
  lines.append(f"|{m['validator']}|{m['arm']}|{m['high_days']}|{m['tp']}|{m['fn']}|{m['ordinary_days']}|{m['fp']}|{m['ordinary_false_positive_rate']:.1%}|{m['precision']:.1%}|")
lines+=['', 'DIAG10은전체360일이고,FRESH7·EL1은같은뒤구간46일의다른묶음이다. 세검증을독립새자료로합산하지않는다. ET는기존양class학습분류기의같은fold저장3seed평균/.5경계로,알고리즘과학습범위가함께달라원인분리비교가아니다. always-high 기준은고EC탐지100%/일반오탐100%,always-normal은둘다0%다.', '', '## 시각별 단일클래스 결과', '', '|검증|시각|탐지/고EC|오탐/일반일|재현율|오탐률|', '|---|---:|---|---|---:|---:|']
for m in s['groups']:
 if m['arm']=='oneclass' and m['scope']=='all':lines.append(f"|{m['validator']}|{m['hour']}|{m['tp']}/{m['high_days']}|{m['fp']}/{m['ordinary_days']}|{m['recall']:.1%}|{m['ordinary_false_positive_rate']:.1%}|")
lines+=['', '## 농장과 구간', '']
for m in s['segments']:lines.append(f"- {m['validator']} {m['farm']} 뒤구간={m['pass2']}:고EC{m['high_days']}일중{m['tp']}탐지/{m['fn']}놓침,일반{m['ordinary_days']}일중{m['fp']}오탐.")
lines+=['', '## 학습과 검증 수용률', '', '다음은모든24시각의고EC인라이어수용률이다. 학습은fold train행가중평균/검증은query행가중평균으로분포·가중치차이와날중복이있고독립표본수가아니다. 주h15성과와직접동일시각비교하지않는다.']
for m in overfit:lines.append(f"- {m['validator']}:학습{m['training_acceptance_all24h']:.1%},검증고EC{m['validation_high_acceptance_all24h']:.1%}({m['validation_high_days']}일).")
lines+=['', '## 검증·한계', '', f'산술신뢰도높음:원등록27fold/소스SHA/학습라벨high1뿐/일반행0/무교집합/같은농장±1·다른농장±3·소비40주변제외를점검했다. CSVloop로h15의12개model/scope그룹confusion·재현율·오탐률·정밀도를다른산술로확인했고첫fold를고정설정재fit해score최대차이{gap:.3g}/판정일치를확인했다. 독립비평은계획→첫2fold중간→전체27fold최종이며근거critique_*_v1.md/critic검산파일을참조한다.', '', '초기비평의경계score>=0 정의는native+1과다를수있어v2에서>0로정정후실행했다. nu=.1은학습수용률이정확히90%라는뜻이아니다. 결정적SVM이라같은3seed반복을독립증거로제시하지않았다. 매fold고EC13~22일학습이며일반일은검증에유지했다. 기존ET와73입력은공유하지만새seed검증채택조건을통과한회귀후보가아니다.', '', '새날일반화신뢰도낮음:이미노출된공개360일/고EC26일·뒤46일고EC5일의작은표본,같은날의여러시각은독립날이아니다. 이번은고정nu/모델1개이므로단일클래스방법전체의가능성을부정하지않는다. EC수치예측/EC회귀개선·새400일model·채택·제출0,공식train_y값추가로드0/40성능재평가0/test0/이전대회자료0다.', '', '재현코드 run_v2.py prepare→first→중간독립검토→rest,report_checks_v1.py. 원폴더의등록과결과는덮어쓰기차단하므로새버전폴더로재등록해실행한다. 근거는PLAN_v2.md/registration_v1.json/final_score_v1.json/local final_h15_v1.csv 및predictions27개/fresh_checks_v1.json/acceptance_gap_v1.json이다.']
p=H/'report_v1.md';assert not p.exists();p.write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(dict(status='PASS',metrics=fresh,repro_gap=gap,overfit=overfit),ensure_ascii=True))
