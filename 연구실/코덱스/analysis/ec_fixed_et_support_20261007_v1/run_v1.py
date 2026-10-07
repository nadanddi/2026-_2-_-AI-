from pathlib import Path
import sys,json,itertools,gc,time
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'연구실/코덱스/analysis/ec_current14_influence_20261007_v1'));import runner_v4 as B
sys.path.insert(0,str(H));import core_v1 as C
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
L=ROOT/'연구실/코덱스/local'/H.name
CASES=[('F47',161,1),('F47',130,4),('F13',231,8),('F13',233,8)]
COLS=B.M.FULL_R3
def write(p,v):B.write(p,v)
def prep():
    raw,jobs,_=B.prepare();days=pd.read_csv(B.H/'baseline_score_v1/days.csv',float_precision='round_trip');ens=days[days.seed.astype(str)=='ensemble']
    selections=[]
    for farm,day,k in CASES:
        t,q,_=jobs[k];pool=ens[(ens.farm==farm)&(ens.k==k)&ens.ordinary&(ens.rmse<=.1)].copy()
        bias={}
        for seed in B.SEEDS:
            p=B.L/f'base/{k}_{seed}.npz';m=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert m['sha']==B.sha(p)
            with np.load(p,allow_pickle=False) as z:
                assert z['row_id'].tolist()==q.row_id.tolist();et=z['et'].copy()
            xx=q[['farm','day','sub_ec']].copy();xx['p']=B.M.shrink(et,q);gg=xx.groupby(['farm','day']);bb=(gg.p.mean()-gg.sub_ec.mean()).abs();bias[seed]=bb
            pool=pool[[bb.loc[(f,d)]<=.1 for f,d in pool[['farm','day']].itertuples(index=False,name=None)]]
        assert len(pool)>0,'No eligible diagnostic control'
        truth=float(ens[(ens.farm==farm)&(ens.day==day)].truth.iloc[0]);pool['ec_gap']=abs(pool.truth-truth)
        stages=[pool[(pool.pass2==(day>=179))&(pool.ec_gap<=.2)],pool[pool.ec_gap<=.2],pool[pool.ec_gap<=.3],pool]
        distance_cols=['season']+[v+'_h0' for v in B.M.RAW];T=t[distance_cols].to_numpy(float);median=np.nanmedian(T,axis=0);T=np.where(np.isnan(T),median,T);std=T.std(axis=0,ddof=0);std=np.where(std==0,1,std)
        def x0(d):
            x=q[(q.farm==farm)&(q.day==d)&(q.hour==0)][distance_cols].to_numpy(float)[0];return np.where(np.isnan(x),median,x)
        target=x0(day)
        for step,pp in enumerate(stages,1):
            if len(pp):break
        candidates=[]
        for d in pp.day:candidates.append((float(np.mean(((x0(d)-target)/std)**2)),int(d)))
        dist,control=min(candidates)
        selections.append(dict(farm=farm,day=day,fold=k,control=control,step=step,pool=len(pool),stage_counts=[len(p) for p in stages],distance=dist,truth=truth,control_truth=float(pp[pp.day==control].truth.iloc[0]),ET_control_abs_bias={str(s):float(bias[s].loc[(farm,control)]) for s in B.SEEDS}))
    pairs=[dict(p,tag=f"{p['farm']}_{p['day']}_to_{p['control']}") for p in selections]+[dict(farm='F47',day=161,fold=1,control=160,tag='F47_161_to_160',historical=True)]
    public=pd.concat([v[1] for v in jobs.values()],ignore_index=True).drop_duplicates('row_id')
    ids=set(jobs[1][0].row_id)&set(jobs[4][0].row_id)&set(jobs[8][0].row_id)
    excluded=set((p['farm'],d) for p in pairs for base in [p['day'],p['control']] for d in range(base-1,base+2))
    common=public[public.row_id.isin(ids)&~pd.Series(list(zip(public.farm,public.day)),index=public.index).isin(excluded)].copy().sort_values(['farm','day','hour']).reset_index(drop=True)
    assert (common.groupby(['farm','day']).size()==24).all();assert not set(zip(common.farm,common.day))&excluded
    querydays=set((p['farm'],d) for p in pairs for d in [p['day'],p['control']]);query=public[pd.Series(list(zip(public.farm,public.day)),index=public.index).isin(querydays)].copy().sort_values(['farm','day','hour']).reset_index(drop=True)
    vec=B.S.M.vectors(raw);ct,cq,notes=B.S.season(common,query,vec)
    nonseason=[c for c in COLS if c!='season'];assert np.array_equal(cq[nonseason].to_numpy(),query[nonseason].to_numpy(),equal_nan=True)
    data=dict(fit=0,pairs=pairs,common_train_days=len(ct)//24,common_train_ids=ct.row_id.tolist(),common_query_ids=cq.row_id.tolist(),common_train_hash=B.S.fhash(ct,['row_id','sub_ec']+COLS),common_query_hash=B.S.fhash(cq,['row_id','sub_ec']+COLS),common_season_changes=query[['row_id','season']].assign(common=cq.season.values).query('season != common').to_dict('records'),groups=C.GROUPS,distance=dict(ddof=0,std_zero=1,missing='fold training per-column median',no_candidate='stop'),sources={str(p):B.sha(p) for p in [H/'PLAN_v1.md',H/'run_v1.py',H/'core_v1.py',B.H/'runner_v4.py',B.PKG/'model.py']},new_fits=5)
    p=H/'preparation_v1.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==data
    else:write(p,data)
    return raw,jobs,pairs,ct,cq,public,vec

def snapshot(label,t,q,seed,existing=None):
    p=L/f'{label}.npz';assert not p.exists()
    if existing:
        with np.load(existing,allow_pickle=False) as z:d={k:z[k].copy() for k in ['offsets','left','right','feature','threshold','value','n_samples','weighted_n_samples','train_X','train_y','train_row_id','imputer_median','train_leaf']}
        assert d['train_row_id'].tolist()==t.row_id.tolist();assert np.array_equal(d['train_y'],t.sub_ec.to_numpy())
        forest=C.Forest(d);median=d['imputer_median'];newfit=0
    else:
        model=B.M.et(seed)
        with threadpool_limits(limits=2):model.fit(t[COLS],t.sub_ec.to_numpy(float))
        im,forest=model.steps[0][1],model.steps[1][1];forest.n_jobs=1;median=im.statistics_;X=im.transform(t[COLS]).astype(np.float32);d=C.pack(forest,X,t.sub_ec.to_numpy(float),t.row_id.to_numpy(str),median);newfit=1
    Q=np.where(np.isnan(q[COLS].to_numpy(float)),median,q[COLS].to_numpy(float)).astype(np.float32)
    pred=forest.predict(Q)
    if label.startswith('original'):
        k=int(label.split('_')[1]);cache=B.L/f'base/{k}_7.npz'
        with np.load(cache,allow_pickle=False) as z:err=float(np.max(abs(pred-z['et'])))
        assert err<=1e-10
    else:err=None
    np.savez_compressed(p,**d);write(p.with_suffix('.json'),dict(label=label,seed=seed,sha=B.sha(p),train_hash=B.S.fhash(t,['row_id','sub_ec']+COLS),new_fit=newfit,baseline_cache_error=err,prep_sha=B.sha(H/'preparation_v1.json')))
    print('FROZEN',label,'fit',newfit,'cache_error',err,flush=True)
    return C.Forest(d),d

def support_rows(w,t):
    x=t[['farm','day','sub_ec']].copy();x['w']=w;x['wy']=w*t.sub_ec.to_numpy();gg=x.groupby(['farm','day']).agg(weight=('w','sum'),weighted_y=('wy','sum'),day_truth=('sub_ec','mean'));gg['selected_y']=gg.weighted_y/gg.weight.replace(0,np.nan)
    return gg[gg.weight>0].sort_values('weight',ascending=False).reset_index()

def paths(forest,d,control,target,tag,label):
    z=d;records=[]
    for hour in [0,6,12,23]:
        for tree,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
            node=0;prefix=[];rec=dict(model=label,pair=tag,hour=hour,tree=tree,diverged=False)
            while z['left'][a+node]!=-1:
                loc=a+node;f=int(z['feature'][loc]);th=float(z['threshold'][loc]);cl=control[hour,f]<=th;tl=target[hour,f]<=th
                if cl!=tl:
                    ci=int(z['left'][loc] if cl else z['right'][loc]);ti=int(z['left'][loc] if tl else z['right'][loc]);rec.update(diverged=True,node=int(node),column=COLS[f],group=next(g for g,cc in C.GROUPS.items() if COLS[f] in cc),threshold=th,control_value=float(control[hour,f]),target_value=float(target[hour,f]),control_left=bool(cl),control_child=ci,target_child=ti,control_count=int(z['n_samples'][a+ci]),target_count=int(z['n_samples'][a+ti]),control_node_mean=float(z['value'][a+ci]),target_node_mean=float(z['value'][a+ti]),prefix=prefix);break
                prefix.append([int(node),COLS[f],th,bool(cl)]);node=int(z['left'][loc] if cl else z['right'][loc])
            records.append(rec)
    write(H/f'paths_{label}_{tag}.json',dict(tree_denominator=600,records=records))

def analyze(label,forest,d,t,q,pairs):
    summary=[];phis=[];inter=[];profiles=[]
    daymean=t.groupby(['farm','day']).sub_ec.mean();high=np.array([daymean.loc[(f,day)]>=1 for f,day in zip(t.farm,t.day)])
    for p in pairs:
        if label.startswith('original') and p['fold']!=int(label.split('_')[1]):continue
        farm,day,control=p['farm'],p['day'],p['control'];tag=p['tag']
        def dayframe(day):
            v=q[(q.farm==farm)&(q.day==day)].sort_values('hour');assert v.hour.tolist()==list(range(24));return v
        aa,bb=dayframe(control),dayframe(day)
        def impute(v):
            x=v[COLS].to_numpy(float);return np.where(np.isnan(x),d['imputer_median'],x).astype(np.float32)
        A,T=impute(aa),impute(bb);groups,X=C.coalitions(A,T,COLS);n=len(groups);flat=X.reshape(-1,47);leaves=forest.apply(flat);pred=d['value'][leaves+d['offsets'][:-1]].mean(axis=1).reshape(-1,24)
        wr=C.aggregate(d['train_leaf'],leaves,False);ws=C.aggregate(d['train_leaf'],leaves,True);raw=wr@d['train_y'];smooth=ws@d['train_y'];hs=ws@high
        assert np.max(abs(raw-pred.mean(axis=1)))<1e-10 and np.max(abs(smooth-pred@C.alpha(True)))<1e-10
        out=L/f'coalition_{label}_{tag}.npz';np.savez_compressed(out,X=flat,query_leaf=leaves,raw_curve=pred,raw=raw,smooth=smooth,high_support=hs,weights_raw=wr,weights_smooth=ws,groups=np.array(groups),train_row_id=d['train_row_id']);write(out.with_suffix('.json'),dict(sha=B.sha(out),forest_sha=B.sha(L/f'{label}.npz'),groups=groups,coalitions=len(X)))
        pr,ps,ph=C.shapley(raw,n),C.shapley(smooth,n),C.shapley(hs,n)
        for i,g in enumerate(groups):phis.append(dict(model=label,pair=tag,group=g,raw_phi=float(pr[i]),smooth_phi=float(ps[i]),high_phi=float(ph[i])))
        full=(1<<n)-1
        for i,j in itertools.combinations(range(n),2):
            mi,mj=1<<i,1<<j
            for context in ['control','failure']:
                for name,v in [('raw',raw),('smooth',smooth),('high_support',hs)]:
                    delta=v[mi|mj]-v[mi]-v[mj]+v[0] if context=='control' else v[full]-v[full^mi]-v[full^mj]+v[full^mi^mj]
                    inter.append(dict(model=label,pair=tag,g1=groups[i],g2=groups[j],context=context,metric=name,nonadditivity=float(delta)))
        for end,frame,index in [('control',aa,0),('failure',bb,-1)]:
            ss=support_rows(ws[index],t);ss.to_csv(H/f'support_{label}_{tag}_{end}.csv',index=False)
            summary.append(dict(model=label,pair=tag,endpoint=end,farm=farm,day=int(frame.day.iloc[0]),truth=float(frame.sub_ec.mean()),raw=float(raw[index]),smooth=float(smooth[index]),high_support=float(hs[index]),samefarm_support=float(ws[index][t.farm.to_numpy()==farm].sum()),coalitions=len(X),train_days=len(t)//24))
            rr=frame[['row_id','hour']+COLS].copy();rr.insert(0,'endpoint',end);rr.insert(0,'pair',tag);rr.insert(0,'model',label);profiles.append(rr)
        if label.endswith('_7'):paths(forest,d,A,T,tag,label)
        print('PAIR_COMPLETE',label,tag,'groups',n,'smooth',round(smooth[0],4),round(smooth[-1],4),'high',round(hs[0],4),round(hs[-1],4),flush=True)
    return summary,phis,inter,profiles

def main():
    L.mkdir(parents=True,exist_ok=True);raw,jobs,pairs,ct,cq,public,vec=prep()
    if '--prepare' in sys.argv:print('PREPARED',len(ct)//24,'days',pairs,flush=True);return
    assert not (H/'completion_v1.json').exists();allresults=[[],[],[],[]];fits=0
    for k in [1,4,8]:
        t,q,_=jobs[k];label=f'original_{k}_7';existing=B.L/'trace/BASE.npz' if k==1 else None
        forest,d=snapshot(label,t,q,7,existing);fits+=int(k!=1)
        if k==1:
            extra=public[(public.farm=='F47')&(public.day==160)].copy();assert not set(extra.row_id)&set(t.row_id);_,extra,_=B.S.season(t,extra,vec);q=pd.concat([q,extra],ignore_index=True)
        results=analyze(label,forest,d,t,q,pairs)
        for dest,src in zip(allresults,results):dest.extend(src)
        del forest,d;gc.collect()
    for seed in B.SEEDS:
        label=f'common_{seed}';forest,d=snapshot(label,ct,cq,seed);fits+=1;results=analyze(label,forest,d,ct,cq,pairs)
        for dest,src in zip(allresults,results):dest.extend(src)
        del forest,d;gc.collect()
    assert fits==5
    for name,records in zip(['endpoints','shapley','interactions'],allresults[:3]):pd.DataFrame(records).to_csv(H/f'{name}_v1.csv',index=False)
    pd.concat(allresults[3],ignore_index=True).to_csv(H/'profiles_v1.csv',index=False)
    write(H/'completion_v1.json',dict(status='COMPLETE_FIXED_ET_DIAGNOSTIC',new_ET_fits=fits,other_fits=0,models=6,pairs=20,prep_sha=B.sha(H/'preparation_v1.json'),raw_y_new_reads=0,adoption=False,files={p.name:B.sha(p) for p in H.glob('*_v1.csv')}))
if __name__=='__main__':main()
