import run_v1 as T
import importlib.util,math
sp=importlib.util.spec_from_file_location('previous_verify',T.H.parent/'ec_high_classifier_20261005_v1/verify_v1.py');V=importlib.util.module_from_spec(sp);sp.loader.exec_module(V)
np,pd=T.np,T.pd

def metric(q,mode):
    y=q.high.to_numpy(int);base=q.prefix_A.to_numpy();picked=q[mode].to_numpy(bool);score=q.p.to_numpy();score=np.where(np.isfinite(score),score,base)
    got=V.metrics(y,score,picked);baseline=V.metrics(y,base,base>=.9)
    high=score[y==1];low=score[y==0]
    if len(high) and len(low):
        pair=math.fsum(1 if a>b else .5 if a==b else 0 for a in high for b in low)/(len(high)*len(low));assert abs(pair-got['auc'])<1e-12
    return dict(model=got,baseline=baseline)

def main():
    T.frozen();refs,bank,prep=T.prep();assert prep==T.load(T.H/'preparation_v1.json');rc=T.load(T.H/'receipt_v1.json');frames=[];done=set();err=0.;grad=0.;repeats=0;models=0;fallbacks=[]
    for rec in rc['manifest']:
        v,k,s,context=[rec[n] for n in ['v','k','s','context']];fp=T.OUT/rec['path'];mp=T.OUT/rec['model'];assert T.sha(fp)==rec['sha'] and T.sha(mp)==rec['model_sha'];b=T.load(mp);q,a,x,sources=bank[v,k,s,context=='meta'];iq,ia,ix,_=bank[v,k,s,True];f=pd.read_csv(fp,float_precision='round_trip');assert np.array_equal(q.row_id,f.row_id) and np.array_equal(T.P.target(q),f.high);T.close(T.R.prefix(q,a),f.prefix_A);T.close(a,f.A)
        if context=='outer':p=V.scalar(x,b['full']) if b['full'] else np.full(len(q),np.nan)
        else:
            p=np.full(len(q),np.nan);ss=T.splits(q);assert ss==b['split']
            for split,m in zip(ss,b['meta']):
                ti=np.array(split['ti']);vi=np.array(split['vi']);assert m['indices']==ti.tolist() and not set(ti)&set(vi)
                banned={(ff,int(dd)+o) for ff,dd in q.iloc[vi][['farm','day']].itertuples(index=False,name=None) for o in [-1,0,1]};assert not banned&set(q.iloc[ti][['farm','day']].itertuples(index=False,name=None))
                td=q.iloc[ti][['farm','day']].copy();td['high']=T.P.target(q)[ti];td=td.drop_duplicates();assert td.high.sum()>=2 and (td.high==0).sum()>=10
                p[vi]=V.scalar(x[vi],m);grad=max(grad,V.learning(q,x,m))
            if ss:
                assert np.isfinite(p).all();replayed=T.cuts(q,a,p);assert replayed['guard_support']==b['cuts']['guard_support']
                for key in ['SPLIT','GUARD']:
                    if replayed[key] is None:assert b['cuts'][key] is None
                    else:T.close([replayed[key]],[b['cuts'][key]])
                # Separate threshold calculation with sorted saved meta scores.
                ys=T.P.target(q);ii=[i for i in range(len(q)) if q.hour.iloc[i]==23 and ys[i]==1];high=sorted(float(p[i]) for i in ii);T.close([high[int(.05*(len(high)-1))]],[b['cuts']['SPLIT']])
                gh=[float(p[i]) for i in ii if T.R.prefix(q,a)[i]>=.9]
                if len(gh)>=3:T.close([min(gh)],[b['cuts']['GUARD']])
                else:assert b['cuts']['GUARD'] is None
        assert np.array_equal(np.isnan(p),f.p.isna());mask=np.isfinite(p)
        if mask.any():err=max(err,T.close(p[mask],f.p.to_numpy()[mask]))
        for mode in ['SPLIT','GUARD']:
            chosen=T.chosen(mode,f.prefix_A.to_numpy(),f.p.to_numpy(),b['cuts'][mode],b['fallback'][mode]);assert np.array_equal(chosen,f[mode].to_numpy(bool));assert (f[mode+'_fallback']==b['fallback'][mode]).all()
            if mode=='GUARD':assert not ((f.GUARD==1)&(f.prefix_A<.9)).any()
        if (v,k,s) not in done:
            done.add((v,k,s));models+=len(b['meta'])+int(b['full'] is not None);fallbacks.append(dict(v=v,k=k,s=s,**b['fallback']))
            if b['full']:
                grad=max(grad,V.learning(iq,ix,b['full']))
                if repeats<2:
                    with T.P.M.threadpool_limits(limits=2):fresh=T.P.fit(iq,ix,np.arange(len(iq)),s)
                    assert fresh==b['full'];repeats+=1
            rr=refs[v,k]['tr'];rkeys=set(rr[['farm','day']].itertuples(index=False,name=None));qkeys=set(q[['farm','day']].itertuples(index=False,name=None))
            assert context=='outer'
            for z in sources:assert all((z['row_id'][:3],d) in rkeys and (z['row_id'][:3],d) not in qkeys for d in z['days'])
        f['v']=v;f['s']=s;f['context']=context;frames.append(f);print('REPAIR_VERIFY',v,k,s,context,flush=True)
    allrows=pd.concat(frames,ignore_index=True);results=[];segments=[]
    for (v,s,context),g in allrows.groupby(['v','s','context']):
        if context=='outer':assert g.row_id.is_unique
        for h in [0,6,12,23]:
            q=g[g.hour==h]
            for mode in ['SPLIT','GUARD']:results.append(dict(v=v,seed=int(s),context=context,hour=h,mode=mode,**metric(q,mode)))
        if context=='outer':
            for (farm,phase),q in g[g.hour==23].assign(phase=lambda z:(z.day>=179).astype(int)).groupby(['farm','phase']):
                for mode in ['SPLIT','GUARD']:segments.append(dict(v=v,seed=int(s),farm=farm,phase=int(phase),mode=mode,**metric(q,mode)))
    primary={}
    for mode in ['SPLIT','GUARD']:
        cells=[r for r in results if r['mode']==mode and r['context']=='outer' and r['hour']==23]
        primary[mode]=dict(nonworse_cells=sum(r['model']['tp']>=r['baseline']['tp'] and r['model']['fp']<=r['baseline']['fp'] for r in cells),total=len(cells),diag_strict=sum(r['model']['fp']<r['baseline']['fp'] for r in cells if r['v']=='DIAG10'))
        primary[mode]['pass']=primary[mode]['nonworse_cells']==9 and primary[mode]['diag_strict']==3
    T.save(T.H/'verification_v1.json',dict(status='PASS_REPLAY_LEARNING_SPLITS_CUTS_PAIRWISE_AUC',models=models,files=len(frames),max_error=err,max_gradient=grad,repeats=repeats,fallbacks=fallbacks,results=results,segments=segments,primary=primary,adoption=False))
    print('REPAIR_VERIFY_PASS',primary,flush=True)
if __name__=='__main__':main()
