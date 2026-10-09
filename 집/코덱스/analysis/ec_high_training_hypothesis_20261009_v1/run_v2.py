from pathlib import Path
import os,sys,json,hashlib,time,gc,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research')); import env
sys.path.insert(0,str(ROOT/'연구실/코덱스/analysis/ec_current14_influence_20261007_v1')); import runner_v4 as R
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
M,SG,S=R.M,R.SG,R.S
L=ROOT/'집/코덱스/local'/H.name
RAW_INPUT=None
SEEDS=(7,101,2024); ARMS=('BASE','ALL_HIGH','DISCORD_HIGH','CONTROL'); ALPHA=.025/3

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with Path(p).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def pins():
    paths=[Path(__file__),H/'PLAN_v1.md',H/'PLAN_v2_addendum.md',H/'PLAN_v3_reporting.md',Path(R.__file__),Path(S.__file__),Path(M.__file__),Path(SG.__file__),Path(sys.modules['season'].__file__),Path(env.__file__),ROOT/'연구실/코덱스/analysis/ec_current14_influence_20261007_v1/preparation_v4.json',R.L/'baseline_rows.csv',ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv',Path(env.DATA)/'train_X.csv']
    return {str(p):sha(p) for p in paths}
def assert_pins():
    reg=json.loads((H/'registration_v2.json').read_text(encoding='utf8'))
    for p,v in reg['pins'].items():assert sha(p)==v,p

def select(t):
    d=t.groupby(['farm','day'],sort=True)[M.FULL_R3].mean()
    y=t.groupby(['farm','day'],sort=True).sub_ec.mean().reindex(d.index)
    assert (t.groupby(['farm','day']).size()==24).all()
    arr=d.to_numpy(float);med=np.nanmedian(arr,axis=0);assert np.isfinite(med).all()
    arr=np.where(np.isnan(arr),med,arr);sd=arr.std(axis=0);sd[sd<1e-12]=1
    z=(arr-arr.mean(axis=0))/sd
    idx=list(d.index); nn=[]; rows=[];discord=[];high=[]
    for i,(f,day) in enumerate(idx):
        eligible=np.array([ff==f and abs(dd-day)>1 for ff,dd in idx]);assert eligible.sum()>=5
        dist=np.sqrt(np.mean((z-z[i])**2,axis=1));order=sorted(np.where(eligible)[0],key=lambda j:(dist[j],idx[j]))[:5]
        neigh=float(np.median(y.iloc[order]));hi=bool(y.iloc[i]>=1);bad=bool(hi and y.iloc[i]-neigh>.5)
        if hi:high.append((str(f),int(day)))
        if bad:discord.append((str(f),int(day)))
        rows.append(dict(farm=str(f),day=int(day),y=float(y.iloc[i]),neighbor_y=neigh,gap=float(y.iloc[i]-neigh),high=hi,discord=bad,neighbors=[dict(farm=str(idx[j][0]),day=int(idx[j][1]),distance=float(dist[j]),y=float(y.iloc[j])) for j in order]))
    control=[];rng=np.random.default_rng(20261009)
    for f in ('F13','F47'):
        for p2 in (False,True):
            n=sum(ff==f and (day>=179)==p2 for ff,day in discord)
            pool=[(str(ff),int(day)) for (ff,day),v in y.items() if ff==f and (day>=179)==p2 and v<1]
            assert len(pool)>=n,(f,p2,n,len(pool))
            if n:control.extend([pool[j] for j in rng.choice(len(pool),n,replace=False)])
    assert len(control)==len(discord) and not set(control)&set(high)
    return {'BASE':set(),'ALL_HIGH':set(high),'DISCORD_HIGH':set(discord),'CONTROL':set(control)},rows

def prepare():
    global RAW_INPUT
    old=json.loads((R.H/'preparation_v4.json').read_text(encoding='utf8'))
    assert old['package_model_sha']==sha(Path(M.__file__)) and old['adapter_sha']==sha(Path(SG.__file__))
    oof=ROOT/'연구실/코덱스/local/ec_vent_gap_20261006_v1/inputs/oof.csv'
    z=pd.read_csv(oof,float_precision='round_trip');z=z[z.validator=='DIAG10'];y=z[['row_id','y']].drop_duplicates();assert len(y)==8640 and y.row_id.is_unique
    raw=pd.read_csv(Path(env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
    assert sha(Path(env.DATA)/'train_X.csv')=='21291a8237fadfca3addd88782f369f6a25159efe1b1508067470c5dfc9a6300'
    RAW_INPUT=raw
    features=M.features(raw[['row_id']+M.RAW]).merge(y.rename(columns={'y':'sub_ec'}),on='row_id',validate='one_to_one')
    assert len(features)==8640 and np.isfinite(features.sub_ec).all()
    vec=S.M.vectors(raw);jobs={};records=[];qids=[];cachepins={}
    for rec in old['records']:
        k=rec['k'];t,q,_=S.season(S.ordered(features,rec['train_ids']),S.ordered(features,rec['query_ids']),vec)
        assert S.fhash(t,['row_id','sub_ec']+M.FULL_R3)==rec['train_full_hash']
        assert S.fhash(q,['row_id','sub_ec']+M.FULL_R3)==rec['query_full_hash']
        assert not set(t.row_id)&set(q.row_id)
        assert t.row_id.is_unique and q.row_id.is_unique and (q.groupby(['farm','day']).size()==24).all()
        for farm,day in q[['farm','day']].drop_duplicates().itertuples(index=False,name=None):assert abs(t.loc[t.farm==farm,'day']-day).min()>=2
        base={}
        for seed in SEEDS:
            p=R.L/'base'/f'{k}_{seed}.npz';mp=p.with_suffix('.json');meta=json.loads(mp.read_text(encoding='utf8'))
            assert meta['sha']==sha(p) and meta['prep_sha']==sha(R.H/'preparation_v4.json')
            with np.load(p,allow_pickle=False) as c:
                assert c['row_id'].tolist()==q.row_id.tolist() and c['train_row_id'].tolist()==t.row_id.tolist()
                base[seed]={n:c[n].copy() for n in ('et','lgb','mlp')}
            cachepins[str(p)]=sha(p);cachepins[str(mp)]=sha(mp)
        bags=[]
        for src in rec['PFN_sources']:
            ps=src['seed'];p=S.C/f'DIAG10_{k}_pfn_{ps}.npz';mp=p.with_suffix('.json');meta=json.loads(mp.read_text(encoding='utf8'))
            assert sha(p)==src['sha']==meta['prediction_sha256']
            assert meta['provenance']['train_hash']==S.fhash(t,['row_id','sub_ec']+M.FULL)
            assert meta['provenance']['validation_hash']==S.fhash(q,['row_id','sub_ec']+M.FULL)
            with np.load(p,allow_pickle=False) as c:
                assert c['row_id'].tolist()==q.row_id.tolist()
                ix=np.random.default_rng(ps).choice(len(t),2000,replace=False)
                assert np.array_equal(ix,c['context_index']) and c['context_row_id'].tolist()==t.row_id.iloc[ix].tolist()
                bags.append(c['raw_pfn'].copy())
            cachepins[str(p)]=sha(p);cachepins[str(mp)]=sha(mp)
        choices,selection=select(t);ref=set(t[['farm','day']].itertuples(index=False,name=None));structure=None;cal=None;ec=t.groupby(['farm','day']).sub_ec.mean()
        jobs[k]=(t,q,np.mean(bags,axis=0),base,choices,structure,cal,ec,ref)
        records.append(dict(k=k,train_ids=t.row_id.tolist(),query_ids=q.row_id.tolist(),selections={a:sorted(v) for a,v in choices.items()},selection_detail=selection))
        qids.extend(q.row_id)
    assert len(qids)==len(set(qids))==8640
    return jobs,dict(folds=records,cachepins=cachepins,pins=pins(),seeds=list(SEEDS),max_candidate_fit=90,new_baseline_replay_fit=1,adoption=False,evaluation_absence_claim=False)

def predframe(k,arm,seed,t,q,pfn,et,lg,ml,structure,cal,ec,ref):
    rm=.48*et+.24*lg+.08*ml+.2*pfn
    sm=M.shrink(rm,q);lo,hi=float(t.sub_ec.min()),float(t.sub_ec.max());pre=np.clip(sm,lo,hi)
    corr=SG.correct(q[['row_id']],pre,structure,ec,ref,cal);final=np.clip(corr,lo,hi)
    f=q[['row_id','farm','day','hour','sub_ec']].copy();f['k']=k;f['arm']=arm;f['seed']=str(seed)
    f['raw_et']=et;f['et_prediction']=np.clip(M.shrink(et,q),lo,hi);f['prediction']=final;f['raw_mix']=rm;f['clip_lo']=lo;f['clip_hi']=hi
    f['constant_prediction']=t.sub_ec.mean()
    assert np.isfinite(f[['prediction','raw_et','et_prediction']].to_numpy()).all()
    return f

def run(jobs,ks):
    L.mkdir(parents=True,exist_ok=True);CP=L/'folds';CP.mkdir(exist_ok=True)
    assert_pins();reg=json.loads((H/'registration_v2.json').read_text(encoding='utf8'))
    for p,v in reg['cachepins'].items():assert sha(p)==v,p
    for k in ks:
        assert_pins();dest=CP/f'fold{k}.csv';meta=dest.with_suffix('.json')
        if dest.exists() or meta.exists():
            assert dest.exists() and meta.exists();m=json.loads(meta.read_text(encoding='utf8'));assert m['sha']==sha(dest) and m['registration']==sha(H/'registration_v2.json');continue
        t,q,pfn,base,choices,structure,cal,ec,ref=jobs[k]
        structure=SG.prepare(RAW_INPUT,ref);cal=SG.ref_calendar(structure,ref)
        if k==0 and not (H/'baseline_replay_v2.json').exists():
            model=M.et(7)
            with threadpool_limits(limits=2):model.fit(t[M.FULL_R3],t.sub_ec.to_numpy(float));model.steps[-1][1].n_jobs=1;pr=model.predict(q[M.FULL_R3])
            gap=float(np.max(abs(pr-base[7]['et'])));assert gap<1e-10,gap
            write(H/'baseline_replay_v2.json',dict(maxdiff=gap,registration=sha(H/'registration_v2.json'),new_fit=1));del model;gc.collect()
        frames=[];fitinfo=[]
        for arm in ARMS:
            removed=np.array([(f,int(d)) in choices[arm] for f,d in zip(t.farm,t.day)]);tt=t.loc[~removed].reset_index(drop=True);ets=[]
            for seed in SEEDS:
                start=time.monotonic()
                if removed.any():
                    model=M.et(seed)
                    with threadpool_limits(limits=2):model.fit(tt[M.FULL_R3],tt.sub_ec.to_numpy(float));model.steps[-1][1].n_jobs=1;et=np.asarray(model.predict(q[M.FULL_R3]),float)
                    assert np.max(abs(et[:8]-model.predict(q.iloc[:8][M.FULL_R3])))<1e-10
                    del model;gc.collect()
                else:et=base[seed]['et'].copy()
                ets.append(et);f=predframe(k,arm,seed,t,q,pfn,et,base[seed]['lgb'],base[seed]['mlp'],structure,cal,ec,ref);frames.append(f)
                fitinfo.append(dict(arm=arm,seed=seed,removed_days=len(choices[arm]),removed_rows=int(removed.sum()),train_ids=tt.row_id.tolist(),new_fit=bool(removed.any()),seconds=time.monotonic()-start))
                print('FIT',k,arm,seed,'deleted',len(choices[arm]),'seconds',round(time.monotonic()-start,1),flush=True)
            frames.append(predframe(k,arm,'ensemble',t,q,pfn,np.mean(ets,axis=0),np.mean([base[s]['lgb'] for s in SEEDS],axis=0),np.mean([base[s]['mlp'] for s in SEEDS],axis=0),structure,cal,ec,ref))
        allf=pd.concat(frames,ignore_index=True)
        original=pd.read_csv(R.L/'baseline_rows.csv',float_precision='round_trip');original=original[original.k==k]
        b=allf[allf.arm=='BASE'].merge(original[['row_id','seed','prediction']],on=['row_id','seed'],validate='one_to_one',suffixes=('','_old'))
        gap=float(abs(b.prediction-b.prediction_old).max());assert gap<1e-10,gap
        tmp=dest.with_suffix('.pending.csv');allf.to_csv(tmp,index=False);os.replace(tmp,dest)
        write(meta,dict(k=k,sha=sha(dest),registration=sha(H/'registration_v2.json'),baseline_final_maxdiff=gap,fitinfo=fitinfo,rows=len(allf)))
        print('FOLD_COMPLETE',k,'baselinefinalgap',gap,flush=True)

def score(ks,label):
    frames=[]
    for k in ks:
        p=L/'folds'/f'fold{k}.csv';m=json.loads(p.with_suffix('.json').read_text(encoding='utf8'));assert m['sha']==sha(p);frames.append(pd.read_csv(p,float_precision='round_trip'))
    o=pd.concat(frames,ignore_index=True);d=o.groupby(['farm','day']).sub_ec.mean();high=d>=1
    o['high']=[bool(high[(f,d)]) for f,d in zip(o.farm,o.day)]
    masks={'all':np.ones(len(o),bool),'normal':~o.high,'high':o.high,'pass2':o.day>=179,'pass2_normal':(o.day>=179)&~o.high,'pass2_high':(o.day>=179)&o.high,'normal_without161':(~o.high)&~((o.farm=='F47')&(o.day==161)),'F13_normal':(~o.high)&(o.farm=='F13'),'F47_normal':(~o.high)&(o.farm=='F47')}
    groups=[];folds=[]
    for scope,mask in masks.items():
        g=o.loc[mask]
        for (arm,seed),a in g.groupby(['arm','seed']):
            if len(a):groups.append(dict(scope=scope,arm=arm,seed=str(seed),rows=len(a),days=a.groupby(['farm','day']).ngroups,rmse=float(np.sqrt(np.mean((a.prediction-a.sub_ec)**2))),et_rmse=float(np.sqrt(np.mean((a.et_prediction-a.sub_ec)**2))),bias=float((a.prediction-a.sub_ec).mean()),constant_rmse=float(np.sqrt(np.mean((a.constant_prediction-a.sub_ec)**2)))))
    for (k,arm,seed),a in o.groupby(['k','arm','seed']):folds.append(dict(k=int(k),arm=arm,seed=str(seed),rmse=float(np.sqrt(np.mean((a.prediction-a.sub_ec)**2)))))
    pd.DataFrame(groups).to_csv(H/f'{label}_groups_v1.csv',index=False);pd.DataFrame(folds).to_csv(H/f'{label}_folds_v1.csv',index=False)
    ens=o[(o.seed=='ensemble')&(~o.high)];piv=ens.pivot(index=['row_id','farm','day'],columns='arm',values='prediction').reset_index();truth=ens[['row_id','sub_ec']].drop_duplicates();piv=piv.merge(truth,on='row_id',validate='one_to_one')
    blocks=piv[['farm','day']].copy();blocks['block']=piv.day//5;piv['block']=blocks.block
    ss={a:piv.assign(loss=(piv[a]-piv.sub_ec)**2).groupby(['farm','block']).loss.sum() for a in ARMS};nc=piv.groupby(['farm','block']).size()
    keys=list(nc.index);rng=np.random.default_rng(20261009);draw=[]
    for farm in ('F13','F47'):
        ids=np.array([i for i,k in enumerate(keys) if k[0]==farm]);assert len(ids)>0
        draw.append(ids[rng.integers(0,len(ids),(20000,len(ids)))])
    ix=np.concatenate(draw,axis=1);counts=nc.to_numpy();den=counts[ix].sum(1);boot=[]
    for arm,refarm in [('ALL_HIGH','BASE'),('DISCORD_HIGH','BASE'),('DISCORD_HIGH','CONTROL')]:
        a=ss[arm].reindex(nc.index).to_numpy();b=ss[refarm].reindex(nc.index).to_numpy();delta=(a-b)[ix].sum(1);dr=np.sqrt(a[ix].sum(1)/den)-np.sqrt(b[ix].sum(1)/den)
        boot.append(dict(arm=arm,reference=refarm,scope='normal',blocks=len(keys),p_worse=float((delta>=0).mean()),rmse_delta=float(np.sqrt(a.sum()/counts.sum())-np.sqrt(b.sum()/counts.sum())),ci95_delta=[float(x) for x in np.quantile(dr,[.025,.975])],alpha=ALPHA))
    write(H/f'{label}_score_v1.json',dict(status='SCORED_PARTIAL' if len(ks)<10 else 'SCORED_FULL_DIAG10_DIAGNOSTIC',folds=ks,groups=groups,bootstrap=boot,adoption=False,full_model_removal=False,evaluation_high_absence=False,registration=sha(H/'registration_v2.json')))
    print('SCORED',label,boot,flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','first','rest','score']);args=a.parse_args()
    if args.mode=='prepare':
        jobs,reg=prepare();write(H/'registration_v2.json',reg);print('PREPARED_FIT0',len(jobs),[(k,{a:len(v) for a,v in j[4].items()}) for k,j in jobs.items()],flush=True);return
    assert (H/'critique_plan_v1.md').exists() and (H/'critique_preflight_v2.md').exists()
    jobs,reg=prepare();old=json.loads((H/'registration_v2.json').read_text(encoding='utf8'));assert reg==old
    if args.mode=='first':run(jobs,[0]);score([0],'midpoint')
    if args.mode=='rest':
        assert (H/'critique_midpoint_v1.md').exists();run(jobs,list(range(1,10)));score(list(range(10)),'final')
    if args.mode=='score':score(list(range(10)),'final')
if __name__=='__main__':main()