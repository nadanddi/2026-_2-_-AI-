from pathlib import Path
import sys,os,json,hashlib,importlib.util,gc,time,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'연구실/코덱스/analysis/ec_resolution_sequence_20261006_v1'));import stage1_v2 as S
PKG=ROOT/'집/클로드/submission14_ec_sg2';sys.path.insert(0,str(PKG))
spec=importlib.util.spec_from_file_location('readonly_ec14_recipe',PKG/'model.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
sys.path.insert(0,str(H));import sg2_ref_v2 as SG
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
L=ROOT/'연구실/코덱스/local'/H.name;C=S.C;SEEDS=[7,101,2024]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def prepare():
    z=pd.read_csv(ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv',float_precision='round_trip');z=z[z.validator=='DIAG10'];y=z[['row_id','y']].drop_duplicates();assert len(y)==8640
    raw=pd.read_csv(Path(S.env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
    assert sha(Path(S.env.DATA)/'train_X.csv')=='21291a8237fadfca3addd88782f369f6a25159efe1b1508067470c5dfc9a6300'
    f=M.features(raw[['row_id']+M.RAW]).merge(y.rename(columns={'y':'sub_ec'}),on='row_id',validate='one_to_one');vec=S.M.vectors(raw);jobs={};records=[];allq=[]
    for k in range(10):
        with np.load(C/f'DIAG10_{k}_r3_7.npz',allow_pickle=False) as c:ti=c['train_row_id'].astype(str);qi=c['row_id'].astype(str)
        t,q,_=S.season(S.ordered(f,ti),S.ordered(f,qi),vec);assert not set(t.row_id)&set(q.row_id)
        for farm,day in q[['farm','day']].drop_duplicates().itertuples(index=False,name=None):assert min(abs(t.loc[t.farm==farm,'day']-day))>=2
        th=S.fhash(t,['row_id','sub_ec']+M.FULL);qh=S.fhash(q,['row_id','sub_ec']+M.FULL);bags=[];sources=[]
        for seed in [1,2,3,4]:
            p=C/f'DIAG10_{k}_pfn_{seed}.npz';meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
            assert meta['provenance']['train_hash']==th and meta['provenance']['validation_hash']==qh
            assert meta['prediction_sha256']==sha(p)
            with np.load(p,allow_pickle=False) as c:
                assert c['row_id'].tolist()==q.row_id.tolist()
                ix=np.random.default_rng(seed).choice(len(t),2000,replace=False);assert np.array_equal(ix,c['context_index'])
                bags.append(c['raw_pfn'].copy())
            sources.append(dict(seed=seed,sha=sha(p)))
        coverage=q.groupby(['farm','day']).sub_ec.mean().reset_index();coverage['pass2']=coverage.day>=179;coverage['high']=coverage.sub_ec>=1
        records.append(dict(k=k,train_rows=len(t),query_rows=len(q),train_ids=t.row_id.tolist(),query_ids=q.row_id.tolist(),train_full_hash=S.fhash(t,['row_id','sub_ec']+M.FULL_R3),query_full_hash=S.fhash(q,['row_id','sub_ec']+M.FULL_R3),train_base_hash=S.fhash(t,['row_id','sub_ec']+M.BASE_R3),query_base_hash=S.fhash(q,['row_id','sub_ec']+M.BASE_R3),PFN_sources=sources,bounds=[float(t.sub_ec.min()),float(t.sub_ec.max())],coverage=coverage[['farm','day','pass2','high']].to_dict('records')))
        jobs[k]=(t,q,np.mean(bags,axis=0));allq.extend(q.row_id)
    assert len(allq)==len(set(allq))==8640
    season_path=Path(sys.modules['season'].__file__)
    data=dict(status='PREPARED_CURRENT14_PUBLIC_RETRAIN',records=records,script_sha=sha(__file__),package_model_sha=sha(PKG/'model.py'),package_sg2_sha=sha(PKG/'sg2post.py'),imported_season_path=str(season_path),imported_season_sha=sha(season_path),adapter_sha=sha(H/'sg2_ref_v2.py'),plan_sha=sha(H/'PLAN_v2.md'),features=dict(FULL_R3=M.FULL_R3,BASE_R3=M.BASE_R3,FULL_PFN=M.FULL),versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=S.sklearn.__version__,lightgbm=S.lightgbm.__version__),fit=0)
    p=H/'preparation_v2.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==data
    else:write(p,data)
    return raw,jobs,data
def cache_r3(k,seed,t,q):
    O=L/'base';O.mkdir(parents=True,exist_ok=True);p=O/f'{k}_{seed}.npz';mp=p.with_suffix('.json')
    if p.exists() or mp.exists():
        assert p.exists() and mp.exists();meta=json.loads(mp.read_text(encoding='utf-8'));assert meta['prep_sha']==sha(H/'preparation_v2.json') and meta['sha']==sha(p)
        with np.load(p,allow_pickle=False) as c:return {n:c[n].copy() for n in ['et','lgb','mlp']}
    out={};start=time.monotonic()
    for name,model,cols in [('et',M.et(seed),M.FULL_R3),('lgb',M.lg(seed),M.BASE_R3),('mlp',M.mlp(seed),M.BASE_R3)]:
        with threadpool_limits(limits=2):model.fit(t[cols],t.sub_ec.to_numpy(float));out[name]=np.asarray(model.predict(q[cols]),float)
        assert np.isfinite(out[name]).all();del model;gc.collect()
    temp=p.with_suffix('.pending.npz');np.savez_compressed(temp,row_id=q.row_id.to_numpy(str),train_row_id=t.row_id.to_numpy(str),**out);os.replace(temp,p)
    write(mp,dict(sha=sha(p),prep_sha=sha(H/'preparation_v2.json'),k=k,seed=seed,seconds=time.monotonic()-start));print('BASE_R3_COMPLETE',k,seed,round(time.monotonic()-start,1),flush=True);return out
def post(k,tag,seed,t,q,rawmix,structure,cal,ec,ref):
    smooth=M.shrink(rawmix,q);lo,hi=float(t.sub_ec.min()),float(t.sub_ec.max());p13=np.clip(smooth,lo,hi)
    corrected,traces=SG.correct(q[['row_id']],p13,structure,ec,ref,cal,return_trace=True);final=np.clip(corrected,lo,hi)
    assert np.isfinite(final).all()
    frame=q[['row_id','farm','day','hour','sub_ec']].copy();frame['k']=k;frame['arm']=tag;frame['seed']=str(seed)
    frame['raw_mix']=rawmix;frame['smooth']=smooth;frame['pre_sg2']=p13;frame['sg2_raw']=corrected;frame['prediction']=final;frame['clip_lo']=lo;frame['clip_hi']=hi
    tf=pd.DataFrame(traces,columns=['row_id','query_calendar','prefix_model','candidate_day','candidate_ec','gate','delta']);frame=frame.merge(tf,on='row_id',how='left',validate='one_to_one');frame['gate']=frame.gate.fillna(False).astype(bool)
    return frame
def baseline(raw,jobs,prep):
    lock=L/'baseline.lock';write(lock,dict(pid=os.getpid(),script_sha=sha(__file__)));parts=[]
    for k,(t,q,pfn) in jobs.items():
        rs={s:cache_r3(k,s,t,q) for s in SEEDS};ref=set(t[['farm','day']].itertuples(index=False,name=None));ec=t.groupby(['farm','day']).sub_ec.mean();structure=SG.prepare(raw,ref);cal=SG.ref_calendar(structure,ref);r3=[]
        for s in SEEDS:
            b=.6*rs[s]['et']+.3*rs[s]['lgb']+.1*rs[s]['mlp'];r3.append(b);parts.append(post(k,'BASE',s,t,q,.8*b+.2*pfn,structure,cal,ec,ref))
        parts.append(post(k,'BASE','ensemble',t,q,.8*np.mean(r3,axis=0)+.2*pfn,structure,cal,ec,ref));print('BASE_FOLD_COMPLETE',k,flush=True)
    p=L/'baseline_rows.csv';assert not p.exists();pd.concat(parts,ignore_index=True).to_csv(p,index=False)
    write(H/'baseline_receipt_v2.json',dict(status='COMPLETE_CURRENT14_RECIPE_DIAG10',rows=34560,unique_rows=8640,unique_days=360,R3_groups=30,classic_fits=90,PFN_new_fit=0,rows_sha=sha(p),prep_sha=sha(H/'preparation_v2.json'),adoption=False,actual_ensemble_order=True))
    assert json.loads(lock.read_text())['pid']==os.getpid();lock.unlink();print('BASE_ALL_COMPLETE',flush=True)
def main():
    L.mkdir(parents=True,exist_ok=True);raw,jobs,prep=prepare()
    if '--prepare' in sys.argv:print('PREPARED_FIT0',len(jobs),flush=True);return
    baseline(raw,jobs,prep)
if __name__=='__main__':main()

