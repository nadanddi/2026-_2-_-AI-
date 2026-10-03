from models_v3 import *
from run_models import metrics
import math
def mean(z):return math.fsum(float(v) for v in z)/len(z)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9),(a,b)
def main():
 world=joblib.load(O/'world.joblib');effects=pd.read_csv(H/'effects_v2.csv',float_precision='round_trip');agg=pd.read_csv(H/'group_effect_summary_v2.csv',float_precision='round_trip');checks=0;comparison=[]
 for key,q in effects.groupby(['target','farm','day','group']):
  row=agg[(agg.target==key[0])&(agg.farm==key[1])&(agg.day==key[2])&(agg.group==key[3])].iloc[0]
  for col,source,fn in [('mean_rmse_change','rmse_change',mean),('mean_prediction_shift','mean_shift',mean),('min_rmse_change','rmse_change',min),('max_rmse_change','rmse_change',max)]:close(row[col],fn(q[source]));checks+=1
  assert row['n']==len(q);checks+=1
 # Reconstruct perturbations from actual training worlds independently.
 for (target,fold,seed,farm,day),qeffects in effects.groupby(['target','fold','seed','farm','day']):
  core=S.loadcore()
  if target=='TEMP':
   lab=world['lab'];fd=next(fd for n,k,fd in world['folds'] if n=='DIAG10' and k==fold);tm,vm=S.common.split_mask(lab,fd);tr=lab[tm];va=lab[vm];model=joblib.load(O/f'TEMP_{fold}_{seed}_model.joblib');gm=json.loads((H/'TEMP_groups.json').read_text(encoding='utf-8'))
  else:
   lab=world['elab'];_,_,tm,vm=next(z for z in world['efolds'] if z[0]=='DIAG10' and z[1]==fold);tr,va=S.seasonal(lab[tm],lab[vm],world['wv']);model=joblib.load(O/f'ec_v3/EC_{fold}_{seed}_model.joblib');gm=json.loads((H/'ec_v3/EC_groups.json').read_text(encoding='utf-8'))
  query=va[(va.farm==farm)&(va.day==day)].sort_values('hour').reset_index(drop=True)
  for row in qeffects.itertuples():
   donor=tr[(tr.farm==farm)&(tr.day==row.donor_day)].sort_values('hour');assert len(donor)==24 and (donor.hour.to_numpy()==query.hour.to_numpy()).all();assert row.donor_day!=day;assert (row.donor_day>=179)==(day>=179);checks+=3
  e=qeffects.sort_values('rmse_change',ascending=False).iloc[0];donor=tr[(tr.farm==farm)&(tr.day==e.donor_day)].sort_values('hour');changed=query.copy();cs=gm[e.group];changed[cs]=donor[cs].to_numpy();y=query['sub_temp' if target=='TEMP' else 'sub_ec'].to_numpy()
  with S.threadpool_limits(limits=2):
   if target=='TEMP':
    # Independent component assembly, using fitted primitives rather than Temp.predict.
    bphys=model.phys.predict(model.imp.transform(changed[model.phc]));res=model.res.predict(changed[model.ct]);ridge=model.ridge.predict(changed[model.ct]);nys=model.nys.predict(changed[model.ct]);cp=model.lin.predict(changed[PHYSICS_COLUMNS])+model.tree.predict(changed[S.FEATURE_COLUMNS]);base=.65*(bphys+res)+.25*ridge+.1*nys
    ref=world['outer'];ref=ref[(ref.validator=='DIAG10')&(ref.base_seed==seed)&(ref.context=='1-8')&(ref.member=='PFN')].set_index('row_id').loc[query.row_id];g=np.where(query.in_temp.isna(),1,np.clip((query.in_temp.to_numpy()-8)/2,0,1));pred=(.4+.1*g)*base+(.6-.4*g)*cp+.3*g*ref.prediction.to_numpy()
   else:
    from ec_exact_finish import rawbag
    et=model.models[0].predict(changed[model.full]);lg=model.models[1].predict(changed[model.base]);mlp=model.models[2].predict(changed[model.base]);raw=.8*(.6*et+.3*lg+.1*mlp)+.2*rawbag(fold,query)
    # Manual causal shrink and bounds.
    prefix=[];shrunk=[]
    for z in raw:prefix.append(z);shrunk.append(.5*z+.5*mean(prefix))
    pred=np.clip(shrunk,tr.sub_ec.min(),tr.sub_ec.max())
  er=[float(p)-float(yv) for p,yv in zip(pred,y)];close(e['mean'],mean(pred));close(e['bias'],mean(er));close(e['rmse'],math.sqrt(mean([v*v for v in er])));checks+=3
 # Independent original-source fits for one fold per target.
 for target,fold in [('TEMP',2),('EC',3)]:
  if target=='TEMP':
   lab=world['lab'];fd=next(fd for n,k,fd in world['folds'] if n=='DIAG10' and k==fold);tm,vm=S.common.split_mask(lab,fd);tr=lab[tm];q=lab[vm].head(48);model=joblib.load(O/f'TEMP_{fold}_7_model.joblib');S.cold_v5.SEED=7
   with S.threadpool_limits(limits=2):m=S.temp_members(tr,q,world['ct'],world['phc'],world['w'][tm]);codex=S.TM.codex_fit_predict(tr,q,world['w'][tm],726);ref=model.predict(q)
   delta=max(float(np.max(np.abs(.65*m['res']+.25*m['ridge']+.1*m['nys']-ref['BASE']))),float(np.max(np.abs(codex-ref['CODEX']))))
  else:
   core=S.loadcore();lab=world['elab'];_,_,tm,vm=next(z for z in world['efolds'] if z[0]=='DIAG10' and z[1]==fold);tr,va=S.seasonal(lab[tm],lab[vm],world['wv']);q=va.head(48);core.FULL=[c for c in core.FULL if c!='day']+['season'];core.BASE=[c for c in core.BASE if c!='day']+['season'];model=joblib.load(O/f'ec_v3/EC_{fold}_7_model.joblib')
   with S.threadpool_limits(limits=2):m=core.members(tr,q,7);delta=float(np.max(np.abs(.6*m[0]+.3*m[1]+.1*m[2]-model.predict(q)['raw_r3'])))
  assert delta<1e-7;comparison.append(dict(target=target,fold=fold,source_refit_max_difference=delta));print('SOURCE_REFIT',target,delta,flush=True)
 savej(H/'verification.json',dict(status='PASS',scalar_and_split_checks=checks,independent_source_refits=comparison,replay=json.loads((H/'model_replay.json').read_text(encoding='utf-8')),limits=['Selected posthoc cases, no causal identification','CPU replacement: correlated derived groups can form hybrid states','Donors picked without labels; donors change different amounts across groups','PFN supplement requires separate native validation','All success cases close to severity threshold; temperature n4/EC n2','No new model adoption or reserved EC read']))
 print('VERIFY_PASS',checks,flush=True)
if __name__=='__main__':main()
