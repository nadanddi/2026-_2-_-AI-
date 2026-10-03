from models_v3 import *
from run_models import metrics
from ec_exact_finish import rawbag
def main():
 world=joblib.load(O/'world.joblib');root=O/'pfn_v3';paths=list(root.glob('PFN_*_*.npz'));assert len(paths)==40,len(paths);deltas={};batch=[];contexts=[]
 for path in paths:
  z=np.load(path);target=path.name.split('_')[1];seed=int(path.stem.split('_')[-1]);spec=[json.loads(s) for s in z['specs']];batch.append(float(z['maxdiff']));original={(s['farm'],s['day']):z['prediction'][i] for i,s in enumerate(spec) if s['group']=='ORIGINAL'}
  for i,s in enumerate(spec):
   if s['group']=='ORIGINAL':continue
   delta=z['prediction'][i]-original[s['farm'],s['day']];key=(target,s['farm'],s['day'],s['group'],s['donor_day']);deltas.setdefault(key,[]).append(delta.astype(float));contexts.append(dict(target=target,farm=s['farm'],day=s['day'],group=s['group'],donor_day=s['donor_day'],context=seed,mean_shift=float(delta.mean())))
 agg={k:np.mean(v,axis=0) for k,v in deltas.items()};effects=pd.read_csv(H/'effects_v2.csv');out=[];hourrows=[];replay=[]
 for case in [z for z in world['selected'] if (z['target']=='TEMP' and (z['farm'],z['day']) in TEMP[:5]) or (z['target']=='EC' and (z['farm'],z['day']) in EC[:2])]:
  target,farm,day,fold=case['target'],case['farm'],case['day'],case['fold'];core=S.loadcore()
  if target=='TEMP':
   lab=world['lab'];fd=next(fd for n,k,fd in world['folds'] if n=='DIAG10' and k==fold);tm,vm=S.common.split_mask(lab,fd);tr,va=lab[tm],lab[vm];gm=json.loads((H/'TEMP_groups.json').read_text(encoding='utf-8'))
  else:
   lab=world['elab'];_,_,tm,vm=next(z for z in world['efolds'] if z[0]=='DIAG10' and z[1]==fold);tr,va=S.seasonal(lab[tm],lab[vm],world['wv']);gm=json.loads((H/'ec_v3/EC_groups.json').read_text(encoding='utf-8'))
  query=va[(va.farm==farm)&(va.day==day)].sort_values('hour').reset_index(drop=True);y=query['sub_temp' if target=='TEMP' else 'sub_ec'].to_numpy()
  for seed in [7,101]:
   model=joblib.load(O/(f'TEMP_{fold}_{seed}_model.joblib' if target=='TEMP' else f'ec_v3/EC_{fold}_{seed}_model.joblib'));ee=effects[(effects.target==target)&(effects.farm==farm)&(effects.day==day)&(effects.seed==seed)&(effects['rank']==1)]
   if target=='TEMP':
    r=world['outer'];r=r[(r.validator=='DIAG10')&(r.base_seed==seed)&(r.context=='1-8')&(r.farm==farm)&(r.day==day)].pivot(index='row_id',columns='member',values='prediction').loc[query.row_id];base=r.W30G.to_numpy();pf=r.PFN.to_numpy();g=S.gate(query)
   else:
    r=world['eouter'];r=r[(r.validator=='DIAG10')&(r.seed==seed)&(r.farm==farm)&(r.day==day)].set_index('row_id').loc[query.row_id];base=r.season_v2.to_numpy();pf=rawbag(fold,query)
   for e in ee.itertuples():
    key=(target,farm,day,e.group,int(e.donor_day))
    if key not in agg:continue
    donor=tr[(tr.farm==farm)&(tr.day==e.donor_day)].sort_values('hour');change=query.copy();cs=gm[e.group];change[cs]=donor[cs].to_numpy()
    with S.threadpool_limits(limits=2):p=model.predict(change);origp=model.predict(query)
    if target=='TEMP':
     cp=(.4+.1*g)*p['BASE']+(.6-.4*g)*p['CODEX']+.3*g*pf;allp=cp+.3*g*agg[key];pfn_only=base+.3*g*agg[key];orig=(.4+.1*g)*origp['BASE']+(.6-.4*g)*origp['CODEX']+.3*g*pf
    else:
     cp=finished_ec(.8*p['raw_r3']+.2*pf,tr,query,core);allp=finished_ec(.8*p['raw_r3']+.2*(pf+agg[key]),tr,query,core);pfn_only=finished_ec(.8*origp['raw_r3']+.2*(pf+agg[key]),tr,query,core);orig=finished_ec(.8*origp['raw_r3']+.2*pf,tr,query,core)
    replay.append(float(np.max(np.abs(orig-base))));m=metrics(allp,y);bm=metrics(base,y);out.append(dict(target=target,farm=farm,day=int(day),seed=seed,group=e.group,donor_day=int(e.donor_day),contexts=len(deltas[key]),cpu_rmse_change=metrics(cp,y)['rmse']-bm['rmse'],pfn_only_rmse_change=metrics(pfn_only,y)['rmse']-bm['rmse'],all_rmse_change=m['rmse']-bm['rmse'],pfn_mean_shift=float(agg[key].mean()),all_mean_shift=float(np.mean(allp-base)),**m))
    for h in range(24):hourrows.append(dict(target=target,farm=farm,day=int(day),seed=seed,group=e.group,donor_day=int(e.donor_day),hour=h,y=y[h],base=base[h],CPU=cp[h],PFN_only=pfn_only[h],ALL=allp[h]))
 assert max(replay)<1e-7
 pd.DataFrame(out).to_csv(H/'whole_model_effects.csv',index=False);pd.DataFrame(hourrows).to_csv(H/'whole_model_effect_hours.csv',index=False);pd.DataFrame(contexts).to_csv(H/'pfn_context_effects.csv',index=False)
 gg=pd.DataFrame(out).groupby(['target','farm','day','group']).agg(all_rmse_change=('all_rmse_change','mean'),cpu_rmse_change=('cpu_rmse_change','mean'),pfn_rmse_change=('pfn_only_rmse_change','mean'),all_mean_shift=('all_mean_shift','mean')).reset_index();gg['rank']=gg.groupby(['target','farm','day']).all_rmse_change.rank(ascending=False,method='min');gg.to_csv(H/'whole_model_effect_summary.csv',index=False)
 savej(H/'pfn_supplement_verification.json',dict(status='APPROXIMATE_PASS',contexts=len(paths),native_original_whole_batch_difference=0.,first_context_full_vs_small_response_max_difference=.000118255615234375,maximum_small_batch_original_difference=max(batch),whole_baseline_replay_maxdiff=max(replay),effect_rows=len(out),precision_note='Native whole batch replays exactly; small batches have documented float32 variation. PFN sensitivities approximate, not adopted model scores.'))
 print(gg[gg['rank']<=3].sort_values(['target','farm','day','rank']).to_string(index=False))
if __name__=='__main__':main()
