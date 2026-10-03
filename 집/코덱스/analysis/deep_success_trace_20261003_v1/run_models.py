from models import *
import argparse,time,warnings
warnings.filterwarnings('ignore',category=UserWarning)
def metrics(p,y):
 e=p-y;return dict(mean=float(np.mean(p)),bias=float(np.mean(e)),rmse=float(np.sqrt(np.mean(e*e))))
def main(target):
 world=joblib.load(O/'world.joblib');selected=[z for z in world['selected'] if z['target']==target];core=S.loadcore();start=time.time()
 groupset=groups(sorted(set(world['ct']+world['phc']+S.FEATURE_COLUMNS)),'TEMP') if target=='TEMP' else groups(['season' if c=='day' else c for c in core.FULL],'EC')
 savej(H/f'{target}_groups.json',groupset)
 for k in sorted({z['fold'] for z in selected}):
  if target=='TEMP':
   lab=world['lab'];fd=next(fd for name,kk,fd in world['folds'] if name=='DIAG10' and kk==k);tm,vm=S.common.split_mask(lab,fd);tr=lab[tm].copy();va=lab[vm].copy()
  else:
   lab=world['elab'];_,_,tm,vm=next(v for v in world['efolds'] if v[0]=='DIAG10' and v[1]==k);tr,va=S.seasonal(lab[tm],lab[vm],world['wv'])
  plan=[z for z in selected if z['fold']==k]
  for seed in [7,101]:
   dest=O/f'{target}_{k}_{seed}_done.json'
   if dest.exists():continue
   modelpath=O/f'{target}_{k}_{seed}_model.joblib'
   with S.threadpool_limits(limits=2):
    if modelpath.exists():model=joblib.load(modelpath)
    elif target=='TEMP':model=Temp().fit(tr,world['ct'],world['phc'],world['w'][tm],seed);joblib.dump(model,modelpath)
    else:model=ECModel().fit(tr,core,seed);joblib.dump(model,modelpath)
    replay=[];baseline=[];effects=[];provenance=[];parts=[]
    cols=sorted({c for cs in groupset.values() for c in cs});med=tr[cols].median();sd=tr[cols].fillna(med).std(ddof=0).replace(0,1).fillna(1)
    for case in plan:
     f,day=case['farm'],case['day'];q=va[(va.farm==f)&(va.day==day)].sort_values('hour').reset_index(drop=True);assert len(q)==24
     p=model.predict(q);y=q['sub_temp' if target=='TEMP' else 'sub_ec'].to_numpy()
     if target=='TEMP':
      ref=world['outer'];r=ref[(ref.validator=='DIAG10')&(ref.base_seed==seed)&(ref.context=='1-8')&(ref.farm==f)&(ref.day==day)].pivot(index='row_id',columns='member',values='prediction').loc[q.row_id]
      for member in ['BASE','CODEX']:replay.append(dict(farm=f,day=day,member=member,maxdiff=float(np.max(np.abs(p[member]-r[member].to_numpy())))))
      g=S.gate(q);predict=lambda pp:(.4+.1*g)*pp['BASE']+(.6-.4*g)*pp['CODEX']+.3*g*r.PFN.to_numpy()
      orig=predict(p);replay.append(dict(farm=f,day=day,member='W30G',maxdiff=float(np.max(np.abs(orig-r.W30G.to_numpy())))))
     else:
      ref=world['eouter'];r=ref[(ref.validator=='DIAG10')&(ref.seed==seed)&(ref.farm==f)&(ref.day==day)].set_index('row_id').loc[q.row_id]
      pfn=r.season_pfn.to_numpy();predict=lambda pp:.8*finished_ec(pp['raw_r3'],tr,q,core)+.2*pfn
      orig=predict(p);replay.append(dict(farm=f,day=day,member='R3',maxdiff=float(np.max(np.abs(finished_ec(p['raw_r3'],tr,q,core)-r.season_r3.to_numpy())))));replay.append(dict(farm=f,day=day,member='v2',maxdiff=float(np.max(np.abs(orig-r.season_v2.to_numpy())))))
     assert max(z['maxdiff'] for z in replay)<=1e-7,replay
     baseline.append(dict(target=target,fold=k,seed=seed,farm=f,day=day,**metrics(orig,y)))
     for name,pp in p.items():parts.append(dict(target=target,seed=seed,farm=f,day=day,component=name,**metrics(pp,y)))
     candidates=tr[(tr.farm==f)&((tr.day>=179)==(day>=179))].groupby('day')
     for group,replace in groupset.items():
      keep=[c for c in cols if c not in replace];query=q[keep].mean().fillna(med[keep]).fillna(0);daily=candidates[keep].mean().fillna(med[keep]).fillna(0);distance=np.sqrt((((daily-query)/sd[keep])**2).mean(axis=1));donors=distance.sort_values().head(5)
      for rank,(dd,dist) in enumerate(donors.items(),1):
       donor=tr[(tr.farm==f)&(tr.day==dd)].sort_values('hour');assert len(donor)==24 and set(donor.hour)==set(range(24));changed=q.copy();changed[replace]=donor[replace].to_numpy();unchanged=[c for c in cols if c not in replace];assert np.allclose(changed[unchanged],q[unchanged],equal_nan=True)
       cp=model.predict(changed);out=predict(cp);stats=metrics(out,y);effects.append(dict(target=target,fold=k,seed=seed,farm=f,day=day,group=group,donor_day=int(dd),rank=rank,distance=float(dist),mean_shift=float(np.mean(out-orig)),rmse_change=stats['rmse']-metrics(orig,y)['rmse'],**stats))
       provenance.append(dict(target=target,fold=k,seed=seed,farm=f,day=day,group=group,donor_day=int(dd),replace_columns=replace,unchanged_maxdiff=0.,donor_in_train=True))
    for name,items in [('replay',replay),('baseline',baseline),('effects',effects),('parts',parts)]:pd.DataFrame(items).to_csv(O/f'{target}_{k}_{seed}_{name}.csv',index=False)
    savej(O/f'{target}_{k}_{seed}_provenance.json',provenance);savej(dest,dict(status='PASS',elapsed=time.time()-start,replay_maxdiff=max(z['maxdiff'] for z in replay)))
   print('DONE',target,k,seed,'seconds',round(time.time()-start),flush=True)
 print('ALL_DONE',target,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('target',choices=['TEMP','EC']);args=ap.parse_args();main(args.target)
