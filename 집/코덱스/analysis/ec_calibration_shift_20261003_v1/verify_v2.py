from pathlib import Path
import csv,json,math
H=Path(__file__).resolve().parent
rows=list(csv.DictReader((H/'distribution_v2.csv').open(encoding='utf-8')));saved=list(csv.DictReader((H/'summary_v2.csv').open(encoding='utf-8')))
checks=0
for s in saved:
 r=[a for a in rows if all(a[k]==s[k] for k in ['validator','sample','hour','group'])];n=sum(int(a['n']) for a in r);assert n==float(s['observations']);checks+=1
 if n:
  bias=math.fsum(float(a['bias'])*int(a['n']) for a in r if int(a['n']))/n;rmse=math.sqrt(math.fsum(float(a['rmse'])**2*int(a['n']) for a in r if int(a['n']))/n)
  assert abs(bias-float(s['weighted_bias']))<1e-12 and abs(rmse-float(s['pooled_rmse']))<1e-12;checks+=2
fits=json.loads((H.parent/'ec_nested_convex_calibration_20261003_v1/fit_audit_v1.json').read_text(encoding='utf-8'))['coefficients'];out=[]
for seed in [7,101,2024]:
 r=[a for a in rows if a['validator']=='DIAG10' and a['seed']==str(seed) and a['hour']=='0' and a['group']=='pred_high'];inside={int(a['fold']):a for a in r if a['sample']=='inner'};outside={int(a['fold']):a for a in r if a['sample']=='outer'}
 pair=[(inside[k],outside[k]) for k in inside if int(inside[k]['n']) and int(outside[k]['n'])]
 opposite=sum(float(a['bias'])*float(b['bias'])<0 for a,b in pair)
 selected=[a for a in fits if a['validator']=='DIAG10' and a['seed']==seed and a['hour']==0]
 corr=[]
 for a in selected:
  b=a['beta'];x=1.2;f=b[0]+b[1]*x+b[2]*max(x-.6,0)+b[3]*max(x-1,0);corr.append(.2*max(-.3,min(.3,f)))
 out.append(dict(seed=seed,paired_folds=len(pair),opposite_bias_folds=opposite,correction_at_1_2_min=min(corr),correction_at_1_2_max=max(corr),negative_corrections=sum(x<0 for x in corr)))
(H/'independent_verification_v2.json').write_text(json.dumps(dict(status='PASS',checks=checks,paired_DIAG0=out,warning='synthetic prediction level 1.2 is for reading fitted rule only; no tuning; day-fold-seed observations are dependent'),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2));print('CHECKS',checks)
