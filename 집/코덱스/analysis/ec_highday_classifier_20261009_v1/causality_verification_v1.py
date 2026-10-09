"""Fresh final inference integrity, causality and independent metrics checks."""
import runtime_v1
import json, math
import pandas as pd
import recipe_v1 as C
from predict_v1 import predict
import run_v1 as P

def main():
 dest=P.L/'model_full_v1';m=json.loads((dest/'model_manifest_v1.json').read_text(encoding='utf8'))
 assert m['code_hashes']==P.codes()
 for member in m['models']:assert P.sha(dest/member['file'])==member['sha']
 x=pd.read_csv(dest/'probe_input_v1.csv',float_precision='round_trip');ids=C.identify(x);base=predict(dest,x).set_index('row_id')
 saved=pd.read_csv(dest/'probe_prediction_v1.csv',float_precision='round_trip').set_index('row_id');gap=float(abs(base.high_ec_score-saved.high_ec_score).max());assert gap<1e-12
 cases=[];days=ids[['farm','day']].drop_duplicates();days['pass2']=days.day>=179
 for r in days.groupby(['farm','pass2']).tail(1).itertuples():
  for h in [8,15]:
   row=ids[(ids.farm==r.farm)&(ids.day==r.day)&(ids.hour==h)].row_id.iloc[0]
   allowed=(ids.farm==r.farm)&((ids.day<r.day)|((ids.day==r.day)&(ids.hour<=h)))
   xx=x.copy();xx.loc[~allowed,C.RAW]=123456.789
   perturbed=predict(dest,xx).set_index('row_id').loc[row,'high_ec_score'];restricted=predict(dest,x[allowed].copy()).set_index('row_id').loc[row,'high_ec_score']
   a=float(abs(perturbed-base.loc[row,'high_ec_score']));b=float(abs(restricted-base.loc[row,'high_ec_score']));assert max(a,b)<1e-12
   cases.append(dict(row_id=row,allowed_rows=int(allowed.sum()),forbidden_rows=int((~allowed).sum()),perturbation_maxdiff=a,restricted_maxdiff=b))
 shuffled=predict(dest,x.sample(frac=1,random_state=20261009)).set_index('row_id').reindex(base.index);shuffle_gap=float(abs(shuffled.high_ec_score-base.high_ec_score).max());assert shuffle_gap<1e-12
 rejects=[]
 for name,bad in [('duplicate_id',pd.concat([x,x.iloc[:1]],ignore_index=True)),('unknown_farm',x.assign(row_id=x.row_id.str.replace('F13','F99',regex=False))),('missing_h0',x[ids.hour!=0].copy())]:
  try:predict(dest,bad)
  except (AssertionError,ValueError):rejects.append(name)
  else:raise AssertionError(name)
 oof=pd.read_csv(P.L/'classification_oof_h15_v1.csv',float_precision='round_trip');score=json.loads((P.H/'final_score_v1.json').read_text(encoding='utf8'));fresh=[]
 for v in ['DIAG10','FRESH7','EL1']:
  for scope in ['all','pass2']:
   z=oof[(oof.validator==v)&((oof.day>=179) if scope=='pass2' else True)];pairs=list(zip(z.high.astype(int),z.ensemble.astype(float)))
   tp=sum(y==1 and p>=.5 for y,p in pairs);fp=sum(y==0 and p>=.5 for y,p in pairs);fn=sum(y==1 and p<.5 for y,p in pairs);tn=sum(y==0 and p<.5 for y,p in pairs)
   brier=math.fsum((p-y)**2 for y,p in pairs)/len(pairs);pos=[p for y,p in pairs if y==1];neg=[p for y,p in pairs if y==0]
   auc=math.fsum(float(p>n)+.5*float(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
   mm=next(t for t in score['groups'] if (t['validator'],t['hour'],t['scope'],t['arm'])==(v,15,scope,'ensemble'))
   assert [tp,fp,tn,fn]==[mm[k] for k in ['tp','fp','tn','fn']];assert abs(brier-mm['brier'])<1e-12 and abs(auc-mm['roc_auc'])<1e-12
   fresh.append(dict(validator=v,scope=scope,days=len(pairs),high_days=len(pos),tp=tp,fp=fp,tn=tn,fn=fn,brier=brier,auc=auc))
 out=dict(status='PASS',code_and_model_hashes_match=True,probe_rows=len(x),reload_maxdiff=gap,shuffle_maxdiff=shuffle_gap,causal_cases=cases,rejected_inputs=rejects,fresh_independent_metrics=fresh,extra40_rescored=False)
 P.save(P.H/'causality_verification_v1.json',out);print(json.dumps(out,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
