from pathlib import Path
import sys,importlib.util,argparse,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
sp=importlib.util.spec_from_file_location('actual_A_audit',H.parent/'ec_final_output_loss_20261004_v1/run_v3.py');F=importlib.util.module_from_spec(sp);sp.loader.exec_module(F)
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def prep():
    F.bootstrap(support=True);refs,p,cols=F.preparation();old=load(H.parent/'ec_final_output_loss_20261004_v1/preparation_v3.json')
    for key in ['source_sha256','config','dependencies','inputs','features','manifest','original_r3_guard']:assert p[key]==old[key],key
    return refs,p

def prefix(q,x):
    np=F.np;out=np.empty(len(q))
    for _,g in q.groupby(['farm','day'],sort=True):
        ix=g.sort_values('hour').index.to_numpy();assert np.array_equal(q.hour.iloc[ix],np.arange(24));out[ix]=np.cumsum(x[ix])/np.arange(1,25)
    return out

def stats(q):
    high=q.high.to_numpy(bool);chosen=q.selected.to_numpy(bool);low=q.hard_high.to_numpy(bool);miss=q.missed_high.to_numpy(bool);false=q.hard_low.to_numpy(bool)
    return dict(n=len(q),high=int(high.sum()),tp=int((high&chosen).sum()),fn=int(miss.sum()),fp=int(false.sum()),hard_high=int(low.sum()),captured_low_high=int((low&chosen).sum()),ordinary=int((~high).sum()))

def actual():
    for name,h in load(H/'registration_v1.json')['hashes'].items():assert F.sha(H/name)==h,name
    refs,p=prep();OUT.mkdir(parents=True,exist_ok=False);pd,np=F.pd,F.np;frames=[];days=[];manifest=[]
    oldreceipt=load(H.parent/'ec_hardcase_crossfit_20261005_v1/receipt_v2.json');proxyout=ROOT/'집/코덱스/local/ec_hardcase_crossfit_20261005_v1'
    for (v,k),r in refs.items():
        if v not in ['DIAG10','A','B']:continue
        tr,q=r['tr'],r['q'];bag=np.mean([F.arrayfile(F.OLD/f'{v}_{k}_pfn_{s}.npz')['raw_pfn'] for s in [1,2,3,4]],axis=0)
        for s in F.SEEDS:
            rr=F.arrayfile(F.OLD/f'{v}_{k}_r3_{s}.npz')['raw_r3'];raw=.8*rr+.2*bag;pred=F.final(raw,q,r['outer_bounds']);F.compare(pred,r['baseline'][s]);F.compare(pred,np.clip(F.smooth_scalar(raw,q),*r['outer_bounds']))
            proxyrec=next(z for z in oldreceipt['files'] if z['v']==v and z['k']==k and z['s']==s and z['mode']=='HARD' and z['context']=='outer');proxyfile=proxyout/proxyrec['path'];assert F.sha(proxyfile)==proxyrec['sha'];proxy=pd.read_csv(proxyfile,float_precision='round_trip');assert np.array_equal(proxy.row_id,q.row_id);F.compare(proxy.A,pred)
            df=q[['row_id','farm','day','hour']].copy();df['y']=q.sub_ec.to_numpy();df['r3']=rr;df['pfn']=bag;df['A']=pred;df['prefix_A']=prefix(q,pred);df['proxy']=proxy.proxy.to_numpy();df['prefix_proxy']=proxy.prefix_proxy.to_numpy();df['lo'],df['hi']=r['outer_bounds'];df['v'],df['k'],df['s']=v,k,s;frames.append(df)
            d=df[df.hour.isin([0,6,12,23])].copy();means=df.groupby(['farm','day']).y.mean();d['y_day']=[means[f,int(day)] for f,day in zip(d.farm,d.day)];d['high']=(d.y_day>=1).astype(int)
            for name,score in [('A','prefix_A'),('proxy','prefix_proxy')]:
                d[name+'_selected']=(d[score]>=.9).astype(int);d[name+'_hard_high']=((d.high==1)&(d[score]<1.2)).astype(int);d[name+'_missed_high']=((d.high==1)&(d[score]<.9)).astype(int);d[name+'_hard_low']=((d.high==0)&(d[score]>=.9)).astype(int)
            days.append(d);print('ACTUAL_CASE',v,k,s,flush=True)
    allrows=pd.concat(frames,ignore_index=True);allcases=pd.concat(days,ignore_index=True);summary=[];segments=[]
    for (v,s,h),g in allcases.groupby(['v','s','hour']):
        for name in ['A','proxy']:
            z=g.copy()
            for c in ['selected','hard_high','missed_high','hard_low']:z[c]=z[name+'_'+c]
            result=stats(z);result.update(v=v,s=int(s),hour=int(h),source=name);summary.append(result)
            for (farm,phase),b in z.assign(phase=(z.day>=179).astype(int)).groupby(['farm','phase']):segments.append(dict(v=v,s=int(s),hour=int(h),source=name,farm=farm,phase=int(phase),**stats(b)))
    consensus=[]
    for (v,k,farm,day,h),g in allcases.groupby(['v','k','farm','day','hour']):
        assert set(g.s)==set(F.SEEDS);consensus.append(dict(v=v,k=int(k),farm=farm,day=int(day),hour=int(h),y_day=float(g.y_day.iloc[0]),high=int(g.high.iloc[0]),A_prefix_min=float(g.prefix_A.min()),A_prefix_max=float(g.prefix_A.max()),A_hard_high_votes=int(g.A_hard_high.sum()),A_hard_low_votes=int(g.A_hard_low.sum()),A_missed_votes=int(g.A_missed_high.sum()),proxy_hard_high_votes=int(g.proxy_hard_high.sum()),proxy_hard_low_votes=int(g.proxy_hard_low.sum())))
    for name,data in [('reconstructed_rows_v1.csv',allrows),('case_registry_v1.csv',allcases),('seed_consensus_v1.csv',pd.DataFrame(consensus))]:
        fp=OUT/name;data.to_csv(fp,index=False);manifest.append(dict(path=name,sha=F.sha(fp),rows=len(data)))
    save(H/'result_v1.json',dict(status='COMPLETE_ACTUAL_A_CASE_DEFINITION',source_sha=F.sha(Path(__file__)),registration_sha=F.sha(H/'registration_v1.json'),audit_inputs=p['inputs'],audit_manifest=p['manifest'],summary=summary,segments=segments,files=manifest,new_model_fits=0,new_classifier=False,new_submission=False,limitation='global OOF registry is diagnostic; not safe to split directly for classifier validation'))

if __name__=='__main__':actual()
