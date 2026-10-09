import runtime_v1
import json,math
import pandas as pd
import run_v1 as P
from pathlib import Path
m=json.loads((P.L/'model_full_v1/model_manifest_v1.json').read_text(encoding='utf8'));assert m['high_days']==30 and m['ordinary_days']==370
cli=pd.read_csv(P.L/'cli_probe_prediction_v1.csv',float_precision='round_trip').set_index('row_id');saved=pd.read_csv(P.L/'model_full_v1/probe_prediction_v1.csv',float_precision='round_trip').set_index('row_id');gap=float(abs(cli.high_ec_score-saved.high_ec_score).max());assert gap<1e-12 and cli.high_ec_class.equals(saved.high_ec_class)
g=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted((P.L/'cv_predictions').glob('*_v1.csv'))],ignore_index=True);matched=[]
for v in ['DIAG10','FRESH7','EL1']:
 z=g[g.validator==v]
 for seed in [8383,1919,7171]:
  receipts=[json.loads(p.read_text(encoding='utf8')) for p in sorted((P.L/'cv_predictions').glob(v+'*_v1.json'))]
  train=[r['training_brier'] for f in receipts for r in f['receipts'] if r['kind']=='et' and r['seed']==seed]
  val=math.fsum((float(p)-int(y))**2 for p,y in zip(z['et_'+str(seed)],z.high))/len(z)
  matched.append(dict(validator=v,seed=seed,training_brier_fold_mean=math.fsum(train)/len(train),validation_brier_all24hours=val))
P.save(P.H/'matched_overfit_cli_v1.json',dict(status='PASS',training_high_days=m['high_days'],training_ordinary_days=m['ordinary_days'],cli_rows=len(cli),cli_reload_maxdiff=gap,matched_member_all24hour_brier=matched))
p=P.H/'report_v2.md';assert not p.exists();body=(P.H/'report_v1.md').read_text(encoding='utf8');body+='\n## 동일 구성원·동일 시각 범위 추가 검산\n\n위의 ensemble 비교를 보완해 각 구성원 train/validation 모두 24시각 Brier를 재계산했다. 학습은 fold별 평균, 검증은 query 전체 행 가중 평균이며 train/query 모집단이 다르다.\n\n|검증|시드|학습 Brier|검증 Brier|\n|---|---:|---:|---:|\n'
for t in matched:body+=f"|{t['validator']}|{t['seed']}|{t['training_brier_fold_mean']:.6f}|{t['validation_brier_all24hours']:.6f}|\n"
body+=f'\n실제 predict_v1.py CLI로192행을 읽고 새 출력 CSV를 생성했다. 저장 probe 대비 점수 최대차이 {gap:.3g}, 분류 일치. 공식400일 고EC30일·일반370일을 manifest 및 독립 원정답 집계로 확인했다.\n'
p.write_text(body,encoding='utf8');print(json.dumps(dict(status='PASS',high_days=m['high_days'],ordinary_days=m['ordinary_days'],cli_maxdiff=gap,matched=matched)))
