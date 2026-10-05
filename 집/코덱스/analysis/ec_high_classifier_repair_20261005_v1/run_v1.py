from pathlib import Path
import sys,importlib.util,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sp=importlib.util.spec_from_file_location('previous_classifier',H.parent/'ec_high_classifier_20261005_v1/run_v1.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
np,pd=P.np,P.pd;save,load,sha,close=P.save,P.load,P.sha,P.close;R=P.R
OUT=ROOT/'집/코덱스/local'/H.name;SEEDS=P.SEEDS

def splits(q):
    d=q[['farm','day']].copy();d['high']=P.target(q);d=d.drop_duplicates().sort_values(['farm','day']);d['phase']=(d.day>=179).astype(int)
    for nfold in [3,2]:
        assigned={};offset=0
        for (high,f,phase),g in d.groupby(['high','farm','phase'],sort=True):
            for i,row in enumerate(g.itertuples(index=False)):
                assigned[row.farm,int(row.day)]=((i if high else i//5)+offset)%nfold
            if high:offset+=len(g)
        result=[]
        for j in range(nfold):
            chosen={key for key,v in assigned.items() if v==j};banned={(f,day+o) for f,day in chosen for o in [-1,0,1]}
            ti=np.array([i for i,(f,day) in enumerate(zip(q.farm,q.day)) if (f,int(day)) not in banned],int);vi=np.array([i for i,(f,day) in enumerate(zip(q.farm,q.day)) if (f,int(day)) in chosen],int)
            td=d[[ (f,int(day)) not in banned for f,day in zip(d.farm,d.day) ]];vd=d[[ (f,int(day)) in chosen for f,day in zip(d.farm,d.day) ]]
            counts=dict(train_high=int(td.high.sum()),train_low=int((td.high==0).sum()),valid_high=int(vd.high.sum()),valid_low=int((vd.high==0).sum()))
            result.append(dict(ti=ti.tolist(),vi=vi.tolist(),counts=counts))
        if all(z['counts']['train_high']>=2 and z['counts']['train_low']>=10 and z['counts']['valid_high']>=1 for z in result):return result
    return []

def prep():
    refs,sig,prior=P.D.prepare();P.SIG=sig;bank={};records=[]
    for (v,k),ref in refs.items():
        if v not in ['DIAG10','A','B']:continue
        split=splits(ref['b'])
        for inner in [True,False]:
            q=ref['b'] if inner else ref['q'];rr=ref['a'] if inner else ref['tr'];nx,sources=P.reference(rr,q,'ALL')
            assert not set(rr.row_id)&set(q.row_id)
            for s in SEEDS:
                qq,a,x=P.T.pair(ref,s,inner);xx=np.column_stack([x,nx]);assert xx.shape[1]==22 and np.isfinite(xx).all()
                bank[v,k,s,inner]=(qq,a,xx,sources)
                records.append(dict(v=v,k=k,s=s,inner=inner,x_sha=R.ar(xx),y_sha=R.ar(P.target(q)),q_ids=R.ids(q.row_id),ref_ids=R.ids(rr.row_id),split=split))
        print('REPAIR_PREP',v,k,'splits',len(split),flush=True)
    return refs,bank,dict(status='PREPARED_FIT0',prior=prior,records=records,params=P.CFG)

def cuts(q,a,meta):
    high=P.target(q)==1;h23=q.hour.to_numpy()==23;selected=R.prefix(q,a)>=.9
    zz=meta[high & h23 & selected]
    return dict(SPLIT=P.threshold(q,meta),GUARD=float(zz.min()) if len(zz)>=3 else None,guard_support=len(zz))

def chosen(mode,base,p,cut,fallback):
    if fallback:return base>=.9
    return (p>=cut) & ((base>=.9) if mode=='GUARD' else True)

def frozen():
    for name,h in load(H/'registration_v1.json')['hashes'].items():assert sha(H/name)==h,name

def actual():
    frozen();refs,bank,pr=prep();assert pr==load(H/'preparation_v1.json');OUT.mkdir(parents=True,exist_ok=False);manifest=[]
    for (v,k),ref in refs.items():
        if v not in ['DIAG10','A','B']:continue
        ss=splits(ref['b'])
        for s in SEEDS:
            q,a,x,sources=bank[v,k,s,True];oq,oa,ox,osources=bank[v,k,s,False];meta=np.full(len(q),np.nan);models=[];full=None;cut=dict(SPLIT=None,GUARD=None,guard_support=0);pr=np.full(len(oq),np.nan)
            if ss:
                seen=np.zeros(len(q),int)
                for z in ss:
                    ti=np.array(z['ti']);vi=np.array(z['vi'])
                    with P.M.threadpool_limits(limits=2):m=P.fit(q,x,ti,s)
                    assert m['constant'] is None;meta[vi]=P.forward(x[vi],m);seen[vi]+=1;models.append(m)
                assert (seen==1).all() and np.isfinite(meta).all();cut=cuts(q,a,meta)
                with P.M.threadpool_limits(limits=2):full=P.fit(q,x,np.arange(len(q)),s)
                assert full['constant'] is None;pr=P.forward(ox,full)
            fallback={m:bool(not ss or cut[m] is None or cut[m]<=0) for m in ['SPLIT','GUARD']}
            bundle=dict(v=v,k=k,s=s,split=ss,meta=models,full=full,cuts=cut,fallback=fallback,inner_sources=sources,outer_sources=osources)
            mp=OUT/f'model_{v}_{k}_{s}.json';save(mp,bundle)
            for context,qq,aa,prob in [('outer',oq,oa,pr),('meta',q,a,meta)]:
                df=qq[['row_id','farm','day','hour']].copy();df['high']=P.target(qq);df['A']=aa;df['prefix_A']=R.prefix(qq,aa);df['p']=prob
                for mode in ['SPLIT','GUARD']:
                    df[mode]=chosen(mode,df.prefix_A.to_numpy(),prob,cut[mode],fallback[mode]).astype(int);df[mode+'_fallback']=fallback[mode]
                fp=OUT/f'{context}_{v}_{k}_{s}.csv';df.to_csv(fp,index=False);manifest.append(dict(path=fp.name,sha=sha(fp),model=mp.name,model_sha=sha(mp),v=v,k=k,s=s,context=context))
            frozen();print('REPAIR_FIT',v,k,s,len(models)+int(full is not None),fallback,flush=True)
    save(H/'receipt_v1.json',dict(status='COMPLETE',manifest=manifest))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:
        refs,bank,p=prep();save(H/'preparation_v1.json',p)
    else:actual()
