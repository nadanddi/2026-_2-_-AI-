"""Same frozen models; preserve serial outputs, resume independent jobs in processes."""
import run_v1 as T
from concurrent.futures import ProcessPoolExecutor,as_completed
import json

def manifest(v,k,s,suffix=''):
    files=[];models=[]
    for mode in T.MODES:
        mp=T.OUT/f'model_{mode}_{v}_{k}_{s}{suffix}.json';b=T.load(mp);shared=T.OUT/b['shared'];assert T.sha(shared)==b['shared_sha']
        models.append(dict(path=mp.name,sha=T.sha(mp)))
        for context in ['outer','train']:
            fp=T.OUT/f'{context}_{mode}_{v}_{k}_{s}{suffix}.csv';assert fp.exists();files.append(dict(path=fp.name,sha=T.sha(fp),model=mp.name,model_sha=T.sha(mp),v=v,k=k,s=s,mode=mode,context=context))
    return files,models

def worker(args):
    v,k,s,r,sig,wv,cols=args;T.frozen();T.CORE=T.R.S.loadcore();T.WV=wv;T.COLS=cols;T.P.SIG=sig
    original=[T.OUT/f'{c}_{m}_{v}_{k}_{s}.csv' for m in T.MODES for c in ['outer','train']]
    if all(p.exists() for p in original):
        try:files,models=manifest(v,k,s);return files,models,dict(v=v,k=k,s=s,status='REUSED_SERIAL_COMPLETE')
        except (json.JSONDecodeError,AssertionError,FileNotFoundError):pass
    suffix='_resume_v2';newfiles=[T.OUT/f'{c}_{m}_{v}_{k}_{s}{suffix}.csv' for m in T.MODES for c in ['outer','train']]
    if all(p.exists() for p in newfiles):
        files,models=manifest(v,k,s,suffix);return files,models,dict(v=v,k=k,s=s,status='REUSED_PARALLEL_COMPLETE')
    tr,q=r['tr'],r['q'];ss=T.splits(tr);xx=T.np.full((len(tr),22),T.np.nan);px=T.np.full(len(tr),T.np.nan);shared_old=T.OUT/f'proxy_{v}_{k}_{s}.json';b=None
    if shared_old.exists():
        try:
            b=T.load(shared_old);assert b['split']==ss
        except (json.JSONDecodeError,AssertionError):b=None
    pm=[];sources=[]
    for j,z in enumerate(ss):
        ti,vi=T.np.array(z['ti']),T.np.array(z['vi']);a,c=tr.iloc[ti].reset_index(drop=True),tr.iloc[vi].reset_index(drop=True)
        if b is None:p,m=T.proxy_fit(a,c,s)
        else:m=b['proxy'][j];p=T.proxy_replay(a,c,m)
        x,src=T.design(a,c,p);xx[vi]=x;px[vi]=p;pm.append(m);sources.append(src)
    assert T.np.isfinite(xx).all()
    if b is None:op,full=T.proxy_fit(tr,q,s)
    else:full=b['proxy_full'];op=T.proxy_replay(tr,q,full)
    ox,osrc=T.design(tr,q,op);shared=T.OUT/f'proxy_{v}_{k}_{s}{suffix}.json';T.save(shared,dict(split=ss,proxy=pm,proxy_full=full,inner_sources=sources,outer_sources=osrc))
    for mode in T.MODES:
        m=T.classifier(tr,xx,s,mode);prob=T.P.forward(ox,m);ip=T.P.forward(xx,m);bundle=dict(v=v,k=k,s=s,mode=mode,shared=shared.name,shared_sha=T.sha(shared),classifier=m);mp=T.OUT/f'model_{mode}_{v}_{k}_{s}{suffix}.json';T.save(mp,bundle)
        for context,qq,proxy,x,p,base in [('outer',q,op,ox,prob,r['baseline'][s]),('train',tr,px,xx,ip,px)]:
            df=qq[['row_id','farm','day','hour']].copy();df['high']=T.P.target(qq);df['proxy']=proxy;df['prefix_proxy']=x[:,1];df['p']=p;df['A']=base;df['prefix_A']=T.R.prefix(qq,base);df['DIRECT']=(p>=.5).astype(int);df['GUARD']=((p>=.5)&(df.prefix_A.to_numpy()>=.9)).astype(int);df['hard_high']=((df.high==1)&(df.prefix_proxy<1.2)).astype(int);df['hard_low']=((df.high==0)&(df.prefix_proxy>=.9)).astype(int);fp=T.OUT/f'{context}_{mode}_{v}_{k}_{s}{suffix}.csv';df.to_csv(fp,index=False)
    T.frozen();files,models=manifest(v,k,s,suffix);return files,models,dict(v=v,k=k,s=s,status='SAME_SPEC_PARALLEL_RESUME',reused_proxy=b is not None)

def main():
    T.frozen();reg=T.load(T.H/'runtime_registration_v1.json');assert T.sha(T.H/'run_v2.py')==reg['run_v2_sha'];refs,p=T.prep();assert p==T.load(T.H/'preparation_v1.json');T.OUT.mkdir(parents=True,exist_ok=True);jobs=[(v,k,s,r,T.P.SIG,T.WV,T.COLS) for (v,k),r in refs.items() if v in ['DIAG10','A','B'] for s in T.SEEDS];files=[];models=[];resume=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,a) for a in jobs]
        for f in as_completed(futures):
            ff,mm,rec=f.result();files.extend(ff);models.extend(mm);resume.append(rec);print('HARD_RESUME',rec,len(resume),'/',len(jobs),flush=True)
    files.sort(key=lambda r:(r['v'],r['k'],r['s'],r['mode'],r['context']=='train'));models.sort(key=lambda r:r['path']);assert len(files)==240 and len(models)==120
    T.save(T.H/'receipt_v2.json',dict(status='COMPLETE_PROXY300_CLASSIFIER120_SAME_SPEC',files=files,models=models,resume=resume,workers=3,runtime_registration_sha=T.sha(T.H/'runtime_registration_v1.json')))
if __name__=='__main__':main()
