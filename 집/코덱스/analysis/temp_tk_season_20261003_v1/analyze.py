from world import *
import math
def boot(d,a,b):
    q=pd.DataFrame({'block':d.farm+'_'+(d.day//5).astype(str),'delta':(b-d.sub_temp.to_numpy())**2-(a-d.sub_temp.to_numpy())**2})
    blocks=q.groupby('block').delta.agg(['sum','count']);rng=np.random.default_rng(20261003);idx=rng.integers(0,len(blocks),(20000,len(blocks)));dr=blocks['sum'].to_numpy()[idx].sum(1)/blocks['count'].to_numpy()[idx].sum(1);alpha=.025/18
    return dict(p_worse=float((dr>=0).mean()),mse_ci=np.quantile(dr,[alpha,1-alpha]).tolist())
def summary(o):
    rows=[]
    for keys,d in o.groupby(['variant','validator','seed','context'],sort=True):
        for seg,mask in {'all':np.ones(len(d),bool),'early':d.day<179,'late':d.day>=179,'F13_late':(d.farm=='F13')&(d.day>=179),'F47_late':(d.farm=='F47')&(d.day>=179)}.items():
            q=d[mask]
            if not len(q):continue
            a,b=rmse(q.base,q.sub_temp),rmse(q.candidate,q.sub_temp);fs=q.groupby('fold').apply(lambda f:rmse(f.candidate,f.sub_temp),include_groups=False)
            rec=dict(zip(['variant','validator','seed','context'],keys));rec.update(segment=seg,n=len(q),days=len(q[['farm','day']].drop_duplicates()),baseline_rmse=a,candidate_rmse=b,delta_pct=100*(b/a-1),fold_rmse_std=float(fs.std(ddof=0)))
            if keys[1]=='DIAG10' and seg=='all':rec.update(boot(q,q.base.to_numpy(),q.candidate.to_numpy()))
            rows.append(rec)
    return rows
def main():
    lab,pfn,ct,phc,wb,wp,wv,sets,z=worlds();frames=[];memberframes=[];cacheaudit=[];split=[]
    for name,folds in sets:
        for k,fd in enumerate(folds):
            tm,vm=common.split_mask(lab,fd);d=lab.loc[vm,['row_id','farm','day','hour','sub_temp','in_temp']].copy();d['validator']=name;d['fold']=k;mp=dict(np.load(OUT/f'members_{name}_{k}.npz',allow_pickle=True));assert np.array_equal(d.row_id,mp['row_id'])
            trdays=lab.loc[tm,['farm','day']].drop_duplicates();vd=d[['farm','day']].drop_duplicates();assert not set(trdays.itertuples(index=False,name=None))&set(vd.itertuples(index=False,name=None))
            gaps=[min(abs(int(dd)-int(day)) for ff,dd in trdays.itertuples(index=False,name=None) if ff==farm) for farm,day in vd.itertuples(index=False,name=None)]
            assert min(gaps)>=2;split.append(dict(validator=name,fold=k,train_rows=int(tm.sum()),validation_rows=int(vm.sum()),min_label_day_gap=min(gaps)))
            m=d.copy()
            for key,v in mp.items():
                if key!='row_id':m[key]=v
            memberframes.append(m)
            if name!='EL1':
                for seed in [7,101]:cacheaudit.append(dict(validator=name,fold=k,member=f'MASK_{seed}',maxdiff=float(np.max(np.abs(mp[f'mask_base_{seed}']-z[f'{name}__MASK__{seed}'][vm])))))
                cacheaudit.append(dict(validator=name,fold=k,member='CODEX_726',maxdiff=float(np.max(np.abs(mp['codex_base']-z[f'{name}__CODEX__726'][vm])))))
            g=np.where(d.in_temp.isna(),1,np.clip((d.in_temp.to_numpy()-8)/2,0,1))
            for context,seeds in [('1-8',list(range(1,9))),('17-24',list(range(17,25)))]:
                pp=[]
                for seed in seeds:
                    p=dict(np.load(OUT/f'pfn_{name}_{k}_{seed}.npz',allow_pickle=True));assert np.array_equal(d.row_id,p['row_id']);assert set(p['context_row_id'])<=set(lab.loc[tm,'row_id']);pp.append(p)
                pb=np.mean([p['base'] for p in pp],axis=0);ps=np.mean([p['season'] for p in pp],axis=0)
                if name!='EL1':
                    old=np.load(Path(env.LOCAL)/f'web_tabpfn_{"v2" if context=="1-8" else "v6"}_temp_{name}.npy');old=old.mean(0) if context=='1-8' else old[:8].mean(0)
                    cacheaudit.append(dict(validator=name,fold=k,member='PFN_'+context,maxdiff=float(np.max(np.abs(pb-old[vm]))),rmse_difference=rmse(pb,d.sub_temp)-rmse(old[vm],d.sub_temp)))
                f=d.copy();f['variant']='TK2_PFN';f['seed']=0;f['context']=context;f['base']=pb;f['candidate']=ps;frames.append(f)
                for bs in [7,101]:
                    b,c=mp[f'mask_base_{bs}'],mp['codex_base'];f=d.copy();f['variant']='TK2_W30G';f['seed']=bs;f['context']=context;f['base']=w30(b,c,pb,g);f['candidate']=w30(b,c,ps,g);frames.append(f)
                    f=d.copy();f['variant']='TK3_TC2_W30G';f['seed']=bs;f['context']=context;f['base']=w30(b,c,pb,g);f['candidate']=w30(mp[f'mask_season_{bs}'],mp['codex_season'],pb,g);frames.append(f)
    o=pd.concat(frames,ignore_index=True);m=pd.concat(memberframes,ignore_index=True);o.to_csv(OUT/'oof.csv',index=False);m.to_csv(OUT/'members.csv',index=False)
    rows=summary(o);pd.DataFrame(rows).to_csv(HERE/'model_scores.csv',index=False)
    repro=[]
    for name in ['TC1','TC2']:
        path=Path(env.LOCAL)/f'temp_{name}_oof.csv'
        if not path.exists():raise FileNotFoundError(path)
        old=pd.read_csv(path);joined=m.merge(old,on=['row_id','validator','fold'],suffixes=('_own','_claude'),validate='one_to_one')
        for c in (['mask_base','mask_seas','codex_base','codex_seas'] if name=='TC1' else ['mask_base_7','mask_seas_7','mask_base_101','mask_seas_101','codex_base','codex_seas']):
            own=c.replace('seas','season')
            if name=='TC1' and c.startswith('mask'):own+='_7'
            own=own+'_own' if own in old.columns else own
            theirs=c+'_claude' if c in m.columns else c
            dif=np.abs(joined[own]-joined[theirs]);repro.append(dict(source=name,column=c,n=len(joined),maxdiff=float(dif.max()),mean_diff=float(dif.mean())))
    def verdict(variant):
        q=[r for r in rows if r['variant']==variant and r['segment']=='all' and r['validator']!='EL1'];diag=[r for r in q if r['validator']=='DIAG10'];return 'PASS_REQUIRES_GUARD_CPU' if all(r['delta_pct']<0 for r in q) and all(r['p_worse']<.025/18 and r['mse_ci'][1]<0 for r in diag) else 'REJECT'
    result=dict(verdicts={v:verdict(v) for v in ['TK2_W30G','TK3_TC2_W30G']},summary=rows,cache_audit=cacheaudit,reproduction=repro,splits=split,k=18,alpha=.025/18,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')})
    (HERE/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result['verdicts']),flush=True)
    for r in rows:
        if r['segment']=='all':print(json.dumps(r),flush=True)
if __name__=='__main__':main()
