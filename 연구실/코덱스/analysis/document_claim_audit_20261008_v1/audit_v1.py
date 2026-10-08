"""Official competition data only; fixed descriptive claim audit, no fitting models."""
from pathlib import Path
import sys, csv, json, hashlib, math, argparse
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
ROOT = H.parents[3]
sys.path.insert(0, str(ROOT / '집/클로드/research'))
import env
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

DATA = Path(env.DATA)
OLD = ROOT / '연구실/코덱스/analysis/ec_current14_influence_20261007_v1'
BASE = ROOT / '연구실/코덱스/local/ec_current14_influence_20261007_v1/baseline_rows.csv'
W = ['out_temp','out_hum','out_rad','out_wspd']
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,float)): return float(x) if math.isfinite(x) else None
    if isinstance(x,np.bool_): return bool(x)
    return x
def save(p,x):
    with p.open('x',encoding='utf-8') as f: json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False)
def ident(d):
    d=d.copy(); d['farm']=d.row_id.str[:3]; d['day']=d.row_id.str[4:7].astype(int); d['hour']=d.row_id.str[8:10].astype(int)
    return d.sort_values(['farm','day','hour']).reset_index(drop=True)
def corr(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    return float(np.corrcoef(x[m],y[m])[0,1]) if m.sum()>2 and np.std(x[m])>0 and np.std(y[m])>0 else np.nan
def rho(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y)
    return float(spearmanr(x[m],y[m]).statistic) if m.sum()>2 and np.std(x[m])>0 and np.std(y[m])>0 else np.nan
def read_inputs():
    receipt=json.loads((OLD/'baseline_receipt_v4.json').read_text())
    assert sha(BASE)==receipt['rows_sha']
    assert sha(OLD/'preparation_v4.json')==receipt['prep_sha']
    prep=json.loads((OLD/'preparation_v4.json').read_text())
    assert sha(OLD/'runner_v4.py')==prep['script_sha']
    assert sha(OLD/'sg2_ref_v2.py')==prep['adapter_sha']
    assert sha(ROOT/'집/클로드/submission14_ec_sg2/model.py')==prep['package_model_sha']
    # Numeric label parsing restricted before conversion to pre-existing public IDs.
    ids=[r for fold in prep['records'] for r in fold['query_ids']]
    assert len(ids)==len(set(ids))==8640
    allowed=set(ids)
    for fold in prep['records']:
        assert set(fold['train_ids']) <= allowed
        assert not set(fold['train_ids']) & set(fold['query_ids'])
    labels=[]; skipped=0
    with (DATA/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['row_id'] not in allowed: skipped+=1;continue
            labels.append({'row_id':r['row_id'],'sub_ec':float(r['sub_ec']),'sub_temp':float(r['sub_temp'])})
    y=ident(pd.DataFrame(labels));assert len(y)==8640 and y.row_id.is_unique
    x=pd.read_csv(DATA/'train_X.csv');te=pd.read_csv(DATA/'test_X.csv')
    assert sha(DATA/'train_X.csv')=='21291a8237fadfca3addd88782f369f6a25159efe1b1508067470c5dfc9a6300'
    assert x.row_id.is_unique and te.row_id.is_unique and set(x.row_id).isdisjoint(te.row_id)
    cols=[c for c in te if c!='row_id']; maskcols=[c for c in cols if te[c].isna().all()]
    x=x[x.row_id.str[:3].isin(['F13','F47'])].copy()
    x[maskcols]=np.nan
    x=ident(x);te=ident(te)
    assert x.groupby(['farm','day']).hour.apply(list).map(lambda z:z==list(range(24))).all()
    assert te.groupby(['farm','day']).hour.apply(list).map(lambda z:z==list(range(24))).all()
    b=ident(pd.read_csv(BASE,dtype={'seed':str},float_precision='round_trip'))
    assert len(b)==34560 and not b.duplicated(['seed','row_id']).any()
    assert set(b.seed)=={'7','101','2024','ensemble'}
    by=b.merge(y[['row_id','sub_ec']],on='row_id',suffixes=('','_source'),validate='many_to_one')
    assert np.max(abs(by.sub_ec-by.sub_ec_source))<1e-14
    for s,g in b.groupby('seed'): assert set(g.row_id)==allowed
    m={'data_sha':{n:sha(DATA/n) for n in ['train_X.csv','test_X.csv','train_y.csv']},
       'baseline_sha':sha(BASE),'prep_sha':sha(OLD/'preparation_v4.json'),
       'source_sha':sha(__file__),'public_label_rows':len(y),'skipped_label_rows_without_numeric_parse':skipped,
       'masked_columns':maskcols,'external_data':False,'fit':0,'submission':False,
       'baseline':'current EC14 recipe public DIAG10, seeds7/101/2024 plus actual ensemble; no current submission score',
       'input_missing':{c:{'test_fraction':te[c].isna().mean(),'train_masked_fraction':x[c].isna().mean()} for c in cols},
       'provenance_receipts':{n:sha(OLD/n) for n in ['critic_verify_baseline_v1.json','critic_verify_pfn_provenance_v1.json','runner_v4.py','sg2_ref_v2.py']}}
    return x,te,y,b,m
def days(x,te,y):
    rows=[]; target=y.set_index('row_id')
    for group,frame in [('train_inputs',x),('test_inputs',te)]:
        for (f,d),g in frame.groupby(['farm','day'],sort=True):
            co=g.in_co2.to_numpy(float); dc=np.diff(co);valid=np.isfinite(dc)
            med=np.median(abs(dc[valid])) if valid.any() else np.nan
            ac=corr(dc[:-1],dc[1:]); rough=bool(med>14);r2=rough and ac<0
            row=dict(farm=f,day=int(d),group=group,p2=d>=179,med=med,ac=ac,R1=rough,R2=r2,
                     public=g.row_id.isin(target.index).all(),sealed=bool(g.act_circfan.mean()<10 and (g.act_vent==0).mean()>.85),
                     in_temp=g.in_temp.mean(),in_hum=g.in_hum.mean(),out_temp=g.out_temp.mean())
            if row['public']:
                z=target.loc[g.row_id];st=z.sub_temp.to_numpy();it=g.in_temp.to_numpy()
                e=pd.Series(it).ewm(alpha=1-.5**.25,adjust=False).mean().to_numpy()
                ok=(np.arange(24)>=3)&np.isfinite(st)&np.isfinite(e)
                row['K1']=corr(st[ok],e[ok]);row['K2']=np.nan
                if ok.sum()>8 and np.std(e[ok])>0:
                    coef=np.linalg.lstsq(np.column_stack([np.ones(ok.sum()),e[ok]]),st[ok],rcond=None)[0]
                    row['K2']=np.sqrt(np.mean((st[ok]-coef[0]-coef[1]*e[ok])**2))
                row['K4']=corr(g.in_temp,g.in_hum);row['truth']=z.sub_ec.mean();row['sub_temp']=z.sub_temp.mean()
            rows.append(row)
    return pd.DataFrame(rows)
def groupsummary(a):
    out=[]
    scopes={'all_train_inputs':a.group.eq('train_inputs'),'public_train':a.public,
            'test':a.group.eq('test_inputs')}
    for f in ['F13','F47']:
        scopes[f+'_train']=a.group.eq('train_inputs')&a.farm.eq(f)
        scopes[f+'_test']=a.group.eq('test_inputs')&a.farm.eq(f)
    for p in [False,True]: scopes['train_pass'+str(1+int(p))]=a.group.eq('train_inputs')&a.p2.eq(p)
    for scope,m in scopes.items():
        for r in ['R1','R2']:
            g=a[m];v=g[r]
            out.append(dict(scope=scope,definition=r,n=len(g),rough=int(v.sum()),fraction=v.mean(),
                            median_rough=g.loc[v,'med'].median(),median_other=g.loc[~v,'med'].median()))
    return pd.DataFrame(out)
def ecstats(a,b):
    b=b.copy();b['err']=b.prediction-b.sub_ec;b['sq']=b.err**2
    daily=b.groupby(['seed','farm','day']).agg(n=('row_id','size'),truth=('sub_ec','mean'),prediction=('prediction','mean'),bias=('err','mean'),sse=('sq','sum')).reset_index()
    daily=daily.merge(a[a.public][['farm','day','R1','R2','sealed','p2','med','K1','K2']],on=['farm','day'],validate='many_to_one')
    assert (daily.n==24).all() and len(daily)==1440
    out=[];aucs=[];mix=[]
    for seed,g in daily.groupby('seed'):
        scopes={'all':np.ones(len(g),bool),'F13':g.farm.eq('F13'),'F47':g.farm.eq('F47'),
                'pass1':~g.p2,'pass2':g.p2,'ordinary':g.truth<1,'high':g.truth>=1,'sealed':g.sealed,'unsealed':~g.sealed}
        for scope,m in scopes.items():
            t=g[m]
            for r in ['R1','R2']:
                z=t[t[r]];o=t[~t[r]]
                msez=z.sse.sum()/z.n.sum() if len(z) else np.nan; mseo=o.sse.sum()/o.n.sum() if len(o) else np.nan
                out.append(dict(seed=seed,scope=scope,definition=r,n=len(t),rough_n=len(z),other_n=len(o),
                    rough_mse=msez,other_mse=mseo,rough_rmse=np.sqrt(msez),other_rmse=np.sqrt(mseo),
                    rmse_ratio=np.sqrt(msez/mseo),mse_ratio=msez/mseo,rough_sse_share=z.sse.sum()/t.sse.sum(),
                    rough_bias=z.bias.mean(),other_bias=o.bias.mean()))
        for scope,mask in [('all',np.ones(len(g),bool)),('ordinary',g.truth<1),('pass2_ordinary',g.p2&(g.truth<1))]:
            t=g[mask]
            for cutoff in [0,.1]:
                flag=t.bias>cutoff
                aucs.append(dict(seed=seed,scope=scope,cutoff=cutoff,n=len(t),positive=int(flag.sum()),auc=roc_auc_score(flag,t.prediction) if flag.nunique()==2 else np.nan))
        t=g[g.prediction>=.9]
        mix.append(dict(seed=seed,n=len(t),true_high=int((t.truth>=1).sum()),ordinary=int((t.truth<1).sum()),
                        over=int((t.bias>0).sum()),under=int((t.bias<0).sum()),rho=rho(t.prediction,t.truth)))
    return daily,pd.DataFrame(out),pd.DataFrame(aucs),pd.DataFrame(mix)
def structure(x,y,a):
    # Exact same full weather vector. Keep every tie. No test inputs used here.
    keys={};groups={}
    for (f,d),g in x.groupby(['farm','day']):
        v=g[W].to_numpy(float)
        if len(g)!=24 or not np.isfinite(v).all(): continue
        key=tuple(v.ravel().tolist());keys[(f,int(d))]=key;groups.setdefault(key,[]).append((f,int(d)))
    pub=a[a.public].set_index(['farm','day']);roles=[];pairs=[];cross=[]
    for gid,(key,kk) in enumerate(groups.items()):
        byfarm={f:sorted(d for ff,d in kk if ff==f) for f in ['F13','F47']}
        for f,ds in byfarm.items():
            if len(ds)==2:
                for d,role in zip(ds,['A','B']):roles.append(dict(farm=f,day=d,role=role,gid=gid))
                if all((f,d) in pub.index for d in ds):
                    p,q=[pub.loc[(f,d)] for d in ds]
                    pairs.append(dict(farm=f,gid=gid,dayA=ds[0],dayB=ds[1],gap=ds[1]-ds[0],
                        ecA=p.truth,ecB=q.truth,dt=q.in_temp-p.in_temp,dst=q.sub_temp-p.sub_temp))
        vals={f:[pub.loc[(f,d),'truth'] for d in ds if (f,d) in pub.index] for f,ds in byfarm.items()}
        if all(vals.values()):cross.append(dict(gid=gid,F13=np.mean(vals['F13']),F47=np.mean(vals['F47']),n13=len(vals['F13']),n47=len(vals['F47'])))
    r=pd.DataFrame(roles);p=pd.DataFrame(pairs);c=pd.DataFrame(cross)
    stats={'weather_complete_days':len(keys),'groups':len(groups),'tie_size_counts':pd.Series([len(v) for v in groups.values()]).value_counts().to_dict(),
           'proxy_only':True,'public_pairs':len(p),'cross_farm_groups':len(c),
           'cross_farm_pearson':corr(c.F13,c.F47),'cross_farm_spearman':rho(c.F13,c.F47),
           'both_farms_high':int(((c.F13>=1)&(c.F47>=1)).sum())}
    pst=[];persist=[]
    for f,g in p.groupby('farm'):
        pst.append(dict(farm=f,n=len(g),pearson=corr(g.ecA,g.ecB),spearman=rho(g.ecA,g.ecB),both_high=int(((g.ecA>=1)&(g.ecB>=1)).sum()),Aonly=int(((g.ecA>=1)&(g.ecB<1)).sum()),Bonly=int(((g.ecB>=1)&(g.ecA<1)).sum()),nonadjacent=int((g.gap!=1).sum())))
    rp=r.merge(pub.reset_index()[['farm','day','truth']],on=['farm','day'],validate='one_to_one')
    for (f,ps),g in rp.groupby(['farm',rp.day>=179]):
        g=g.sort_values('day')
        for same in [True,False]:
            xx=[];yy=[]
            for row in g.itertuples():
                prev=g[(g.day<row.day)&(g.day>=row.day-10)&(g.gid!=row.gid)]
                prev=prev[(prev.role==row.role) if same else (prev.role!=row.role)]
                if len(prev):xx.append(prev.iloc[-1].truth);yy.append(row.truth)
            persist.append(dict(farm=f,pass2=bool(ps),same_role=same,n=len(xx),rho=rho(xx,yy)))
    lag=[]
    for f in ['F13','F47']:
        g=pub.loc[f]
        for k in [1,2,3]:
            ds=[d for d in g.index if d-k in g.index and (d>=179)==(d-k>=179)]
            lag.append(dict(farm=f,lag=k,n=len(ds),rho=rho([g.loc[d-k,'truth'] for d in ds],[g.loc[d,'truth'] for d in ds])))
    yz=y.set_index(['farm','day','hour']);rs=r.set_index(['farm','day']);bound=[]
    for row in rp.itertuples():
        f,d=row.farm,row.day
        if (f,d-1) not in rs.index or (f,d-1,23) not in yz.index: continue
        prev=rs.loc[(f,d-1)]
        bound.append(dict(farm=f,day=d,same_role=row.role==prev.role,same_weather=row.gid==prev.gid,
                          jump=abs(yz.loc[(f,d,0),'sub_temp']-yz.loc[(f,d-1,23),'sub_temp'])))
    boundary=pd.DataFrame(bound)
    stats['boundary']=boundary.groupby('same_role').jump.agg(['size','median','mean']).reset_index().to_dict('records') if len(boundary) else []
    if len(p)>2:
        coef=np.linalg.lstsq(np.column_stack([np.ones(len(p)),p.dt]),p.dst,rcond=None)[0];pred=coef[0]+coef[1]*p.dt
        stats['pair_temp_relation']={'n':len(p),'corr_squared':corr(p.dt,p.dst)**2,'ols_r2':1-np.sum((p.dst-pred)**2)/np.sum((p.dst-p.dst.mean())**2),'slope':coef[1],'intercept':coef[0]}
    return stats,{'roles':r,'pairs':p,'cross_farm':c,'persistence':pd.DataFrame(persist),'record_lag':pd.DataFrame(lag),'midnight':boundary,'pair_ec':pd.DataFrame(pst)}
def label_checks(y):
    spikes=[];edge=[]
    for (f,d),g in y.groupby(['farm','day']):
        st=g.sub_temp.to_numpy();ec=g.sub_ec.to_numpy()
        for h in range(1,23):
            l,rr=st[h]-st[h-1],st[h]-st[h+1]
            if abs(l)>3 and abs(rr)>3 and l*rr>0:spikes.append(dict(row_id=g.iloc[h].row_id,temp=st[h],prev=st[h-1],next=st[h+1]))
            if abs(l)>3 or abs(rr)>3:edge.append(g.iloc[h].row_id)
    chosen=y[y.row_id.isin(['F47_114_09','F47_114_10','F47_114_11'])][['row_id','sub_ec']].to_dict('records')
    return dict(F47_114=chosen,temperature_spikes=spikes,spike_count=len(spikes),either_neighbor_jump_count=len(set(edge)),scope='F13/F47 public 360 days only, not all other49 farms')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['first','rest'],required=True);args=ap.parse_args()
    out=H/('first_v1' if args.stage=='first' else 'rest_v1');out.mkdir(exist_ok=False)
    x,te,y,b,m=read_inputs();a=days(x,te,y)
    if args.stage=='first':
        a.to_csv(out/'day_inputs.csv',index=False);groupsummary(a).to_csv(out/'rough_groups.csv',index=False)
        d,s,u,z=ecstats(a,b)
        for name,df in [('ec_days',d),('ec_rough',s),('prediction_auc',u),('high_prediction_mix',z)]:df.to_csv(out/(name+'.csv'),index=False)
        k=[]
        for r in ['R1','R2']:
            for v in [False,True]:
                g=a[a.public & a[r].eq(v)]
                k.append(dict(definition=r,rough=v,n=len(g),K1_median=g.K1.median(),K2_median=g.K2.median(),K4_median=g.K4.median()))
        save(out/'coupling.json',k)
        print(groupsummary(a).to_string(index=False));print(s[(s.seed=='ensemble') & s.scope.isin(['all','ordinary','high','pass2'])].to_string(index=False));print(u[u.seed=='ensemble'].to_string(index=False))
    else:
        s,frames=structure(x,y,a)
        for n,df in frames.items():df.to_csv(out/(n+'.csv'),index=False)
        save(out/'structure.json',s);save(out/'labels.json',label_checks(y));print(json.dumps(clean(s),ensure_ascii=False,indent=2));print(json.dumps(clean(label_checks(y)),ensure_ascii=False,indent=2))
    m['output_sha']={p.name:sha(p) for p in out.iterdir() if p.is_file()};save(out/'manifest.json',m)
    print('COMPLETE',args.stage,flush=True)
if __name__=='__main__':main()
