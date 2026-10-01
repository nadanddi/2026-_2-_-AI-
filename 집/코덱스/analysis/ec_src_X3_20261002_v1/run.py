"""X3 사전 고정 라벨 생성 지문 진단. 제출/모델 생성 없음."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집'/'클로드'/'research'))
import env
sys.dont_write_bytecode=True
import csv, json, math, hashlib
from decimal import Decimal, InvalidOperation
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
LOCK=Path(env.CODEX)/'ec_final_lock'/'locked_days.json'
LABEL=Path(env.DATA)/'train_y.csv'
OOF=ROOT/'집'/'코덱스'/'local'/'ec_restart_phase3_20261001_v1'/'oof_predictions.csv'
METRICS=['low_precision_fraction','digit05_fraction','grid001_fraction','strict_fraction','missing_fraction']
OUTCOMES=['ec_mean','level_sqerr','shape_mse']
B=20000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def to_json(d):
    if isinstance(d,dict):return {k:to_json(v) for k,v in d.items()}
    if isinstance(d,(list,tuple)):return [to_json(v) for v in d]
    if isinstance(d,np.generic):return to_json(d.item())
    if isinstance(d,float) and not math.isfinite(d):return None
    return d

def fingerprint(s):
    if s.strip().lower() in ('','nan','na','none','null'):
        return dict(ec=np.nan,raw_decimal_digits=np.nan,significant_digits=np.nan,last_digit=np.nan,
                    low_precision=np.nan,digit05=np.nan,grid001=np.nan,grid01=np.nan,grid1=np.nan,
                    scientific=0,magnitude=np.nan)
    d=Decimal(s)
    if not d.is_finite():raise ValueError(s)
    n=d.normalize()
    digits=n.as_tuple().digits
    return dict(ec=float(d),raw_decimal_digits=max(0,-d.as_tuple().exponent),significant_digits=len(digits),
                last_digit=digits[-1],low_precision=int(len(digits)<=12),digit05=int(digits[-1] in (0,5)),
                grid001=int(d%Decimal('.001')==0),grid01=int(d%Decimal('.01')==0),grid1=int(d%Decimal('.1')==0),
                scientific=int('e' in s.lower()),magnitude=d.adjusted() if d!=0 else 0)

def strict_flags(vals):
    linear=np.zeros(len(vals),bool);plateau=np.zeros(len(vals),bool)
    for start in range(len(vals)-5):
        q=vals[start:start+6]
        if np.isfinite(q).all():
            if np.all(np.diff(q)==0):plateau[start:start+6]=True
            if np.max(q)-np.min(q)>=.02 and np.max(np.abs(np.diff(q,n=2)))<=1e-6:linear[start:start+6]=True
    return linear,plateau

def correlation(a,b):
    aa=a-a.mean();bb=b-b.mean()
    den=np.linalg.norm(aa)*np.linalg.norm(bb)
    return float(np.dot(aa,bb)/den) if den>1e-12 else np.nan

def residualize(d,vec,outcome):
    base=pd.get_dummies(d.farm+'_'+d.section,dtype=float).to_numpy()
    cols=[base]
    for f in sorted(set(d.farm+'_'+d.section)):
        z=(d.farm+'_'+d.section).eq(f).to_numpy().astype(float)
        day=d.day.to_numpy(float)/250
        cols.extend([z[:,None]*day[:,None],z[:,None]*day[:,None]**2])
    cols.append(d.magnitude_mean.to_numpy()[:,None])
    if outcome!='ec_mean':
        lv=np.log(np.maximum(d.ec_mean.to_numpy(),1e-9))
        cols.extend([lv[:,None],lv[:,None]**2])
    design=np.concatenate(cols,axis=1)
    return vec-design@np.linalg.lstsq(design,vec,rcond=None)[0]

def make_blocks(d):
    chunks=[]
    for _,g in d.groupby(['farm','section'],sort=True):
        ids=g.sort_values('day').index.to_numpy()
        chunks.append([ids[k:k+5] for k in range(0,len(ids),5)])
    return chunks

def block_stats(d,x,y,seed1=3103,seed2=3104):
    obs=correlation(x,y)
    if not math.isfinite(obs):return obs,np.nan,np.nan,np.nan
    blocks=make_blocks(d)
    pgen=np.random.default_rng(seed1)
    permutation=np.empty((B,len(d)),dtype=np.int32)
    for chunks in blocks:
        target=np.concatenate(chunks)
        for j in range(B):permutation[j,target]=np.concatenate([chunks[k] for k in pgen.permutation(len(chunks))])
    # y residual center/norm unaffected by permutation; each stratum is preserved.
    null=((y[permutation]-y.mean())@(x-x.mean()))/(np.linalg.norm(y-y.mean())*np.linalg.norm(x-x.mean()))
    p=(1+int((np.abs(null)>=abs(obs)).sum()))/(B+1)
    bgen=np.random.default_rng(seed2)
    samples=[]
    for j in range(B):
        ii=np.concatenate([np.concatenate([chunks[k] for k in bgen.integers(0,len(chunks),len(chunks))]) for chunks in blocks])
        samples.append(correlation(x[ii],y[ii]))
    lo,hi=np.nanquantile(samples,[.01/15/2,1-.01/15/2])
    return obs,p,float(lo),float(hi)

def build():
    locks={(r['farm'],int(r['day'])) for r in json.loads(LOCK.read_text(encoding='utf-8'))['selected']}
    records=[];skip=0
    with LABEL.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            farm,day,hour=r['row_id'].split('_');day=int(day);hour=int(hour)
            if farm not in ['F13','F47']:continue
            if (farm,day) in locks:skip+=1;continue
            s=r['sub_ec']
            records.append(dict(row_id=r['row_id'],farm=farm,day=day,hour=hour,raw_ec=s,**fingerprint(s)))
    rows=pd.DataFrame(records).sort_values(['farm','day','hour']).reset_index(drop=True)
    assert len(rows)==8640 and rows.row_id.is_unique
    raw_oof=pd.read_csv(OOF)
    diag=raw_oof[raw_oof.validator.eq('DIAG10')][['row_id','sub_ec','v2']]
    assert diag.row_id.is_unique and len(diag)==8640
    rows=rows.merge(diag,on='row_id',how='left',validate='one_to_one')
    assert rows.v2.notna().all()
    assert np.max(np.abs(rows.ec-rows.sub_ec))<=1e-12
    rows['strict_linear']=False;rows['plateau6']=False
    days=[]
    for (farm,day),g in rows.groupby(['farm','day']):
        assert g.hour.tolist()==list(range(24))
        linear,plateau=strict_flags(g.ec.to_numpy())
        rows.loc[g.index,'strict_linear']=linear;rows.loc[g.index,'plateau6']=plateau
        err=g.v2.to_numpy()-g.ec.to_numpy();bias=np.mean(err)
        d=dict(farm=farm,day=day,section='late' if day>=179 else 'first',n=24,n_missing=int(g.ec.isna().sum()),
               ec_mean=float(g.ec.mean()),ec_median=float(g.ec.median()),level_error=float(bias),level_sqerr=float(bias*bias),
               shape_mse=float(np.mean((err-bias)**2)),mse=float(np.mean(err**2)),magnitude_mean=float(g.magnitude.mean()),
               low_precision_fraction=float(g.low_precision.mean()),digit05_fraction=float(g.digit05.mean()),
               grid001_fraction=float(g.grid001.mean()),strict_linear_fraction=float(linear.mean()),
               plateau_fraction=float(plateau.mean()),strict_fraction=float(max(linear.mean(),plateau.mean())),
               missing_fraction=float(g.ec.isna().mean()),digit_entropy=float(-sum((v/len(g))*math.log2(v/len(g)) for v in Counter(g.last_digit.dropna()).values())))
        for m in METRICS:d[m+'_group']=int(d[m]>=(1/24 if m in ['strict_fraction','missing_fraction'] else .5))
        days.append(d)
    daily=pd.DataFrame(days)
    assert len(daily)==360
    assert np.max(np.abs(daily.mse-daily.level_sqerr-daily.shape_mse))<1e-12
    return rows,daily,skip

def main():
    rows,daily,skip=build()
    stats=[]
    for mi,m in enumerate(METRICS):
        for yi,yname in enumerate(OUTCOMES):
            d=daily[np.isfinite(daily[m])&np.isfinite(daily[yname])].reset_index(drop=True)
            x=residualize(d,rankdata(d[m]),yname);y=residualize(d,rankdata(d[yname]),yname)
            rho,p,lo,hi=block_stats(d,x,y)
            splits={int(k):v for k,v in d.groupby(m+'_group')}
            group_counts={k:len(splits.get(k,[])) for k in [0,1]}
            farm_counts={f:{k:int(((d.farm==f)&(d[m+'_group']==k)).sum()) for k in [0,1]} for f in ['F13','F47']}
            enough=all(n>=15 for n in group_counts.values()) and all(n>=5 for z in farm_counts.values() for n in z.values())
            farm_rho={f:correlation(x[d.farm.eq(f)],y[d.farm.eq(f)]) for f in ['F13','F47']}
            consistent=all(math.isfinite(v) and v*rho>0 for v in farm_rho.values()) if math.isfinite(rho) else False
            rec=dict(metric=m,outcome=yname,n=len(d),rho_raw=correlation(rankdata(d[m]),rankdata(d[yname])),
                     rho_adjusted=rho,p_block=p,p_bonferroni=min(1,p*15) if math.isfinite(p) else np.nan,
                     ci_lower=lo,ci_upper=hi,group0_days=group_counts[0],group1_days=group_counts[1],
                     enough_groups=bool(enough),farm_F13_rho=farm_rho['F13'],farm_F47_rho=farm_rho['F47'],
                     farm_sign_consistent=bool(consistent),found=bool(enough and consistent and math.isfinite(p) and p*15<.01))
            for k in [0,1]:
                ss=splits.get(k)
                rec['group'+str(k)+'_outcome_mean']=float(ss[yname].mean()) if ss is not None else np.nan
                rec['group'+str(k)+'_outcome_median']=float(ss[yname].median()) if ss is not None else np.nan
            group1=splits.get(1)
            removed=float(group1.level_sqerr.sum()) if group1 is not None else 0.0
            rec['group1_level_sse_share']=removed/float(d.mse.sum())
            rec['group1_level_oracle_rmse']=float(np.sqrt((d.mse.sum()-removed)/len(d)))
            stats.append(rec)
    rows.to_csv(HERE/'row_features.csv',index=False,encoding='utf-8-sig')
    daily.to_csv(HERE/'day_features.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(stats).to_csv(HERE/'stats.csv',index=False,encoding='utf-8-sig')
    strata=[]
    r=rows.copy()
    r['section']=np.where(r.day>=179,'late','first')
    r['hours']=np.where(r.hour<=6,'00_06','07_23')
    dm=daily.set_index(['farm','day']).ec_mean
    r['ec_band']=pd.cut([dm.loc[(f,d)] for f,d in zip(r.farm,r.day)],[-np.inf,.3,.6,1.2,np.inf],right=False).astype(str)
    for dim in ['farm','section','hours','ec_band']:
        for k,g in r.groupby(dim):
            strata.append(dict(dimension=dim,group=str(k),rows=len(g),days=len(g[['farm','day']].drop_duplicates()),
                ec_mean=float(g.ec.mean()),low_precision_fraction=float(g.low_precision.mean()),digit05_fraction=float(g.digit05.mean()),
                grid001_fraction=float(g.grid001.mean()),strict_linear_fraction=float(g.strict_linear.mean()),
                plateau_fraction=float(g.plateau6.mean()),missing_fraction=float(g.ec.isna().mean())))
    pd.DataFrame(strata).to_csv(HERE/'strata_summary.csv',index=False,encoding='utf-8-sig')
    # Fixed samples: two highest daily level errors and one median-error day per farm; no hypothesized mechanism selection.
    selected=[]
    for f,g in daily.groupby('farm'):
        q=g.sort_values(['level_sqerr','day']);selected.extend([(f,int(d)) for d in q.tail(2).day])
        selected.append((f,int(q.iloc[len(q)//2].day)))
    rows[[(f,d) in selected for f,d in zip(rows.farm,rows.day)]].to_csv(HERE/'raw_examples.csv',index=False,encoding='utf-8-sig')
    summary=dict(rows=len(rows),days=len(daily),locked_rows_skipped_before_label_access=skip,
        rmse=float(np.sqrt(daily.mse.mean())),level_mse=float(daily.level_sqerr.mean()),shape_mse=float(daily.shape_mse.mean()),
        level_sse_fraction=float(daily.level_sqerr.sum()/daily.mse.sum()),missing_rows=int(rows.ec.isna().sum()),
        low_precision_rows=int(rows.low_precision.sum()),digit05_rows=int(rows.digit05.sum()),grid001_rows=int(rows.grid001.sum()),
        strict_linear_rows=int(rows.strict_linear.sum()),plateau_rows=int(rows.plateau6.sum()),
        raw_decimal_digits={str(k):int(v) for k,v in rows.raw_decimal_digits.value_counts().sort_index().items()},
        significant_digits={str(k):int(v) for k,v in rows.significant_digits.value_counts().sort_index().items()},
        scientific_rows=int(rows.scientific.sum()),groups={m:int(daily[m+'_group'].sum()) for m in METRICS},
        found_comparisons=[s for s in stats if s['found']],planned_comparisons=15,permutations=B,
        seeds=dict(permutation=3103,bootstrap=3104),hashes={str(p.relative_to(ROOT)):sha(p) for p in [LABEL,OOF,LOCK,HERE/'PROTOCOL.md',HERE/'run.py']},
        final_lock_scored=False,model_created=False,submission_created=False)
    dump(HERE/'results.json',to_json(summary))
    print(json.dumps(to_json(summary),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
