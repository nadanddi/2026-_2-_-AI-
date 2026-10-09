from pathlib import Path
import csv,json,math
H=Path(__file__).resolve().parent;R=H.parents[3];L=R/'집/코덱스/local'/H.name;data=[];receipts=[]
for p in sorted((L/'cv_predictions').glob('*_v1.csv')):
 qs=list(csv.DictReader(p.open(encoding='utf8')));data+=qs
 for t in json.loads(p.with_suffix('.json').read_text(encoding='utf8'))['receipts']:receipts.append(dict(validator=qs[0]['validator'],**t))
diagnostic=json.loads((H/'diagnostics_v1.json').read_text(encoding='utf8'));matched=json.loads((H/'matched_overfit_cli_v1.json').read_text(encoding='utf8'))
for q in matched['matched_member_all24hour_brier']:
 ts=[t for t in receipts if t['kind']=='et' and t['seed']==q['seed'] and t['validator']==q['validator']];train=math.fsum(t['training_brier'] for t in ts)/len(ts)
 qs=[t for t in data if t['validator']==q['validator']];valid=math.fsum((float(t[f"et_{q['seed']}"])-int(t['high']))**2 for t in qs)/len(qs)
 assert abs(train-q['training_brier_fold_mean'])<1e-12 and abs(valid-q['validation_brier_all24hours'])<1e-12
for q in diagnostic['segments']:
 qs=[t for t in data if t['validator']==q['validator'] and t['farm']==q['farm'] and int(t['hour'])==15 and (int(t['day'])>=179)==q['pass2']];ys=[int(t['high']) for t in qs];ps=[float(t['ensemble']) for t in qs]
 for key,cond in [('tp',lambda y,p:y==1 and p>=.5),('fp',lambda y,p:y==0 and p>=.5),('tn',lambda y,p:y==0 and p<.5),('fn',lambda y,p:y==1 and p<.5)]:assert sum(cond(y,p) for y,p in zip(ys,ps))==q[key]
 assert abs(math.fsum((p-y)**2 for y,p in zip(ys,ps))/len(qs)-q['brier'])<1e-12
 pos=[p for y,p in zip(ys,ps) if y];neg=[p for y,p in zip(ys,ps) if not y];auc=sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg));assert abs(auc-q['roc_auc'])<1e-12
for q in diagnostic['reliability']:
 text=q['bin'][1:-1];lo,hi=map(float,text.split(','));qs=[t for t in data if t['validator']==q['validator'] and int(t['hour'])==15 and lo<float(t['ensemble'])<=hi]
 assert len(qs)==q['days'];assert abs(math.fsum(float(t['ensemble']) for t in qs)/len(qs)-q['mean_score'])<1e-12;assert abs(sum(int(t['high']) for t in qs)/len(qs)-q['high_rate'])<1e-12
out={'status':'INDEPENDENT_SEGMENT_CALIBRATION_MATCHED_BRIER_PASS','matched_member_checks':len(matched['matched_member_all24hour_brier']),'segments_checked':len(diagnostic['segments']),'reliability_bins_checked':len(diagnostic['reliability']),'receipt_training_scores_independently_aggregated_not_refit':True,'validation_member_scores_from_saved_OOF_independently_recomputed':True,'matched_member_all24hour_brier':matched['matched_member_all24hour_brier']}
p=H/'critic_diagnostics_recheck_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k!='matched_member_all24hour_brier'}))
