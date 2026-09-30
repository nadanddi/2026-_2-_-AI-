import sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import json, hashlib
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

def main():
    here=Path(__file__).resolve().parent
    out=ROOT/'연구실/코덱스/local/ec_measurement_d13_v1'/datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    def read(p):
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return pd.read_csv(p)
    for p in (here/'run.py',here/'PROTOCOL.md'):
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    old=ROOT/'집/코덱스/analysis/local'
    vs=[]
    for f in (0,2,4,6,8,9):
        folder='ec_three_seed_ensemble_cv/20260928_035914' if f<8 else 'ec_locked_confirmation/20260928_044934'
        z=read(old/folder/f'fold{f}.csv')
        vs.append(z[['row_id','farm','day','hour','sub_ec']])
    v=pd.concat(vs,ignore_index=True)
    assert len(v)==5616 and v.row_id.is_unique
    labels=read(Path(env.DATA)/'train_y.csv')
    labels=labels[labels.row_id.isin(v.row_id)][['row_id','sub_temp']]
    raw=read(Path(env.DATA)/'train_X.csv')
    raw=raw[raw.row_id.isin(v.row_id)]
    d=v.merge(labels,on='row_id',validate='one_to_one').merge(raw,on='row_id',validate='one_to_one')
    excluded=int((d.sub_ec<=0).sum())
    d=d[d.sub_ec.gt(0)].sort_values(['farm','day','hour']).reset_index(drop=True)
    d['lec']=np.log(d.sub_ec)
    d['vpd']=(.6108*np.exp(17.27*d.in_temp/(d.in_temp+237.3))*(1-d.in_hum/100)).clip(lower=0)
    d['radiation']=d.out_rad/100
    d['vent']=d.act_vent/100
    def demean(frame,cols,two=False):
        x=frame[cols].copy()
        for _ in range(30 if two else 1):
            x=x-x.groupby([frame.farm,frame.day]).transform('mean')
            if two:x=x-x.groupby([frame.farm,frame.hour]).transform('mean')
        return x
    def coeff(frame,two=False,controls=False):
        cols=['lec','sub_temp']+(['in_temp','vpd','radiation','vent'] if controls else [])
        clean=frame.dropna(subset=cols).copy()
        x=demean(clean,cols,two)
        b=np.linalg.lstsq(x[cols[1:]],x.lec,rcond=None)[0]
        return dict(rows=len(clean),coefficients=dict(zip(cols[1:],map(float,b))))
    findings=[]
    for farm,g in [('all',d),*list(d.groupby('farm'))]:
        findings.append(dict(farm=farm,within_day=coeff(g),two_way=coeff(g,True),controlled=coeff(g,True,True)))
    d['dt']=d.groupby(['farm','day']).sub_temp.diff()
    d['de']=d.groupby(['farm','day']).lec.diff()
    night=d[d.hour.between(1,5)].dropna(subset=['dt','de'])
    night_b=float(np.sum(night.dt*night.de)/np.sum(night.dt**2))
    days=list(d.groupby(['farm','day']).indices.values())
    rng=np.random.default_rng(300913)
    boots=[]
    # Day bootstrap of day-demeaned sufficient statistics avoids resampled-ID aliasing.
    w=demean(d[['farm','day','hour','lec','sub_temp']].dropna(),['lec','sub_temp'])
    tmp=d.loc[w.index,['farm','day']].copy()
    tmp['xy']=w.lec*w.sub_temp;tmp['xx']=w.sub_temp**2
    ss=tmp.groupby(['farm','day'])[['xy','xx']].sum().to_numpy()
    for _ in range(2000):
        draw=ss[rng.integers(0,len(ss),len(ss))].sum(axis=0)
        boots.append(float(draw[0]/draw[1]))
    result=dict(status='PASS',rows=len(d),days=len(days),excluded_nonpositive=excluded,associations=findings,
        night_first_difference=dict(rows=len(night),slope=night_b),within_day_ci95=np.quantile(boots,[.025,.975]).tolist(),
        bootstrap_draws=2000,seed=300913,final_lock_scored=False,test_read=False,measurement_definition_identified=False,sha256=hashes)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},ensure_ascii=False,indent=2))
    print('OUTPUT',out)

if __name__=='__main__':main()
