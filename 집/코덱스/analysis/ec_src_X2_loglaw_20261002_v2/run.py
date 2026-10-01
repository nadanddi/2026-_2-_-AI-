from pathlib import Path
import sys,os,csv,json,math,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
lockpath=Path(env.CODEX)/'ec_final_lock/locked_days.json'
lock={(r['farm'],int(r['day'])) for r in json.loads(lockpath.read_text(encoding='utf-8'))['selected']}
records=[]
with (Path(env.DATA)/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        farm,day,h=r['row_id'].split('_')
        if farm not in ['F13','F47'] or (farm,int(day)) in lock:continue
        records.append({'row_id':r['row_id'],'farm':farm,'day':int(day),'hour':int(h),'sub_ec':float(r['sub_ec']),'sub_temp':float(r['sub_temp'])})
d=pd.DataFrame(records).merge(pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id','in_temp']),on='row_id',validate='one_to_one')
sp=ROOT/'집/코덱스/analysis/local/rl_ec_v1/20260927_173801/splits.csv'
d=d.merge(pd.read_csv(sp)[['row_id','block','fold']],on='row_id',validate='one_to_one').sort_values(['farm','day','hour']).reset_index(drop=True)
assert len(d)==8640 and d.row_id.is_unique and d.sub_ec.gt(0).all()
g=d.groupby(['farm','day']);assert g.size().eq(24).all()
d['level']=g.sub_ec.transform('mean');d['logec']=np.log(d.sub_ec);d['centerlog']=d.logec-d.groupby(['farm','day']).logec.transform('mean')
pred={};betas=[]
for col in ['sub_temp','in_temp']:
    for lag in [0,1,3]:
        name=f'{col}_lag{lag}'; t=g[col].shift(lag).fillna(g[col].transform(lambda z:z.iloc[0])) if lag else d[col]
        c=t-t.groupby([d.farm,d.day]).transform('mean');p=np.full(len(d),np.nan)
        for fold in range(10):
            va=d.fold.eq(fold);vd=set(d.loc[va,['farm','day']].itertuples(index=False,name=None))
            forbidden={(f,z+j) for f,z in lock|vd for j in [-1,0,1]}
            tr=np.array([(f,z) not in forbidden for f,z in zip(d.farm,d.day)])
            valid=tr & c.notna().to_numpy()
            beta=float(np.dot(c[valid],d.loc[valid,'centerlog'])/np.dot(c[valid],c[valid]))
            betas.append({'model':name,'fold':fold,'beta':beta})
            logp=beta*c[va].fillna(0);z=np.exp(logp);z=z/z.groupby([d.loc[va,'farm'],d.loc[va,'day']]).transform('mean')
            p[va]=d.loc[va,'level']*z
        assert np.isfinite(p).all();pred[name]=p
result={'status':'PASS','rows':len(d),'days':g.ngroups,'oracle_only':True,'final_lock_scored':False,'models':{}}
rng=np.random.default_rng(2610022);draws={}
for farm in ['F13','F47']:
    n=d.loc[d.farm.eq(farm),'block'].nunique();draws[farm]=rng.integers(0,n,size=(20000,n))
def score(y,p):
    a=float(np.sqrt(np.mean((np.asarray(y)-p)**2)))
    b=math.sqrt(math.fsum((float(u)-float(v))**2 for u,v in zip(y,p))/len(y));assert math.isclose(a,b,abs_tol=1e-12);return a
for name,p in pred.items():
    totals=np.zeros((20000,2));errors=(d.sub_ec.to_numpy()-p)**2
    for farm in ['F13','F47']:
        m=d.farm.eq(farm);a=pd.DataFrame({'block':d.loc[m,'block'],'n':1,'ss':errors[m]}).groupby('block')[['n','ss']].sum().to_numpy();totals+=a[draws[farm]].sum(axis=1)
    bs=np.sqrt(totals[:,1]/totals[:,0]);ci=np.quantile(bs,[.05/12,1-.05/12]).tolist()
    strata={}
    for label,m in [('F13',d.farm.eq('F13')),('F47',d.farm.eq('F47')),('early',d.day.lt(179)),('late',d.day.ge(179)),('hours0to6',d.hour.le(6)),('hours7to23',d.hour.gt(6)),('high_ec',d.level.ge(1.2))]:strata[label]={'n':int(m.sum()),'rmse':score(d.loc[m,'sub_ec'],p[m])}
    result['models'][name]={'rmse':score(d.sub_ec,p),'ci_bonferroni6':ci,'bootstrap_probability_rmse_le_005':float((bs<=.05).mean()),'strata':strata,'threshold_pass':bool(ci[1]<=.05 and all(strata[f]['rmse']<=.05 for f in ['F13','F47']))}
    d[name]=p
d.to_csv(HERE/'predictions.csv',index=False);pd.DataFrame(betas).to_csv(HERE/'coefficients.csv',index=False)
result['hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'PROTOCOL.md',Path(__file__),Path(env.DATA)/'train_y.csv',Path(env.DATA)/'train_X.csv',sp,lockpath]}
(HERE/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
