from pfn_attribution import *
def main():
 torch.set_num_threads(4);assert torch.cuda.is_available();world=joblib.load(O/'world.joblib');start=time.time();destroot=O/'pfn_v3';destroot.mkdir(exist_ok=True)
 tx,_,sx=S.safeload();cf=S.build_features(tx,sx).set_index('row_id');temp=world['lab'].copy();temp[S.FEATURE_COLUMNS]=cf.loc[temp.row_id,S.FEATURE_COLUMNS].to_numpy()
 for target in ['TEMP','EC']:
  effects=pd.read_csv(H/'effects_v2.csv');effects=effects[(effects.target==target)&(effects.seed==7)&(effects['rank']==1)]
  cases=[z for z in world['selected'] if z['target']==target and ((z['farm'],z['day']) in (TEMP[:5] if target=='TEMP' else EC[:2]))];groupmap=json.loads((H/('TEMP_groups.json' if target=='TEMP' else 'ec_v3/EC_groups.json')).read_text(encoding='utf-8'))
  for fold in sorted({z['fold'] for z in cases}):
   if target=='TEMP':
    fd=next(fd for n,k,fd in world['folds'] if n=='DIAG10' and k==fold);tm,vm=S.common.split_mask(temp,fd);tr,va=temp[tm],temp[vm];cols=S.FEATURE_COLUMNS;seeds=range(1,9)
   else:
    core=S.loadcore();lab=world['elab'];_,_,tm,vm=next(z for z in world['efolds'] if z[0]=='DIAG10' and z[1]==fold);tr,va=S.seasonal(lab[tm],lab[vm],world['wv']);cols=[c for c in core.FULL if c!='day']+['season'];seeds=range(1,5)
   specs=[];queries=[]
   for case in [z for z in cases if z['fold']==fold]:
    f,day=case['farm'],case['day'];q=va[(va.farm==f)&(va.day==day)].sort_values('hour');specs.append(dict(farm=f,day=day,group='ORIGINAL',donor_day=-1));queries.append(q[cols].to_numpy(np.float32))
    for e in effects[(effects.farm==f)&(effects.day==day)].itertuples():
     replace=[c for c in groupmap[e.group] if c in cols]
     if not replace:continue
     donor=tr[(tr.farm==f)&(tr.day==e.donor_day)].sort_values('hour');changed=q.copy();changed[replace]=donor[replace].to_numpy();queries.append(changed[cols].to_numpy(np.float32));specs.append(dict(farm=f,day=day,group=e.group,donor_day=int(e.donor_day)))
   query=np.concatenate(queries);train=tr[cols].to_numpy(np.float32)
   for seed in seeds:
    dest=destroot/f'PFN_{target}_{fold}_{seed}.npz'
    if dest.exists():continue
    cache=R/(f'집/코덱스/local/temp_tk_season_20261003_v1/pfn_DIAG10_{fold}_{seed}.npz' if target=='TEMP' else f'집/코덱스/local/ec_dc4_integration_20261002_v1/DIAG10_{fold}_pfn_{seed}.npz');stored=np.load(cache);context=stored['context_row_id'].astype(str);idx=pd.Index(tr.row_id).get_indexer(context);assert (idx>=0).all() and not set(context)&set(va.row_id)
    model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cuda',n_estimators=4,random_state=seed,ignore_pretraining_limits=True,inference_precision=torch.float32)
    model.fit(train[idx],tr['sub_temp' if target=='TEMP' else 'sub_ec'].to_numpy()[idx]);pred=np.asarray(model.predict(query));old=stored['base' if target=='TEMP' else 'raw_pfn'];mp=dict(zip(stored['row_id'].astype(str),old));maxdiff=0.
    for i,spec in enumerate(specs):
     if spec['group']=='ORIGINAL':
      ids=[f"{spec['farm']}_{spec['day']:03d}_{h:02d}" for h in range(24)];maxdiff=max(maxdiff,float(np.max(np.abs(pred[24*i:24*(i+1)]-[mp[s] for s in ids]))))
    assert maxdiff<=.001,(target,fold,seed,maxdiff)
    np.savez_compressed(dest,prediction=pred.reshape(len(specs),24),specs=np.array([json.dumps(s) for s in specs]),maxdiff=maxdiff,context_row_id=context)
    print('PFN_APPROX_DONE',target,fold,seed,'batch_maxdiff',maxdiff,'seconds',round(time.time()-start),flush=True);del model;gc.collect();torch.cuda.empty_cache()
 print('PFN_APPROX_ALL_DONE',flush=True)
if __name__=='__main__':main()
