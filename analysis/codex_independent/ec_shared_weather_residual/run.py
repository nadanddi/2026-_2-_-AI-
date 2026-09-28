"""Cross-farm residual concordance on exact shared-weather public days."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'온라인대회자료/정형데이터/참가자_배포'
SEARCH = ROOT/'analysis/local/ec_three_seed_ensemble_cv/20260928_035914'
CONFIRM = ROOT/'analysis/local/ec_locked_confirmation/20260928_044934'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths=[*[SEARCH/f'fold{i}.csv' for i in (0,2,4,6)],
           CONFIRM/'fold8.csv',CONFIRM/'fold9.csv']
    frames=[]
    for path in paths:
        a=pd.read_csv(path)
        a['candidate']=a['blend'] if 'blend' in a else a['candidate']
        frames.append(a[['row_id','farm','day','sub_ec','candidate']])
    oof=pd.concat(frames,ignore_index=True)
    assert len(oof)==5616 and oof.row_id.is_unique
    oof['residual']=oof.sub_ec-oof.candidate
    daily=oof.groupby(['farm','day'],as_index=False).agg(residual=('residual','mean'),rows=('row_id','size'))
    assert len(daily)==234 and daily.rows.eq(24).all()
    x=pd.read_csv(DATA/'train_X.csv',usecols=['row_id','out_temp','out_rad'])
    x=x[x.row_id.str[:3].isin(('F13','F47'))].copy()
    x['farm']=x.row_id.str[:3]; x['day']=x.row_id.str[4:7].astype(int)
    x['hour']=x.row_id.str[8:10].astype(int)
    x.sort_values(['farm','day','hour'],inplace=True)
    keys=[]
    for (farm,day),g in x.groupby(['farm','day']):
        assert len(g)==24
        v=np.nan_to_num(g[['out_temp','out_rad']].to_numpy(float),nan=-9999)
        weather_hash=hashlib.sha256(np.round(v,2).tobytes()).hexdigest()
        keys.append((farm,day,weather_hash))
    weather=pd.DataFrame(keys,columns=['farm','day','weather_hash'])
    assert len(weather)==400
    daily=daily.merge(weather,on=['farm','day'],validate='one_to_one')
    daily['section']=np.where(daily.day<179,'first','second')
    daily['centered']=daily.residual-daily.groupby(['farm','section']).residual.transform('mean')
    cell=daily.groupby(['weather_hash','section','farm'],as_index=False).agg(
        residual=('centered','mean'),days=('day','size'))
    wide=cell.pivot_table(index=['weather_hash','section'],columns='farm',values='residual')
    wide=wide.dropna(subset=['F13','F47']).reset_index()
    assert wide[['weather_hash','section']].duplicated().sum()==0
    def corr(g):
        if len(g)<3:return np.nan
        return float(np.corrcoef(g.F13,g.F47)[0,1])
    overall=corr(wide)
    first=corr(wide[wide.section.eq('first')])
    second=corr(wide[wide.section.eq('second')])
    sign=float(np.mean(np.sign(wide.F13)==np.sign(wide.F47)))
    groups={s:g.index.to_numpy() for s,g in wide.groupby('section')}
    y=wide.F47.to_numpy(float)
    rng=np.random.default_rng(20260929)
    null=np.empty(10000)
    for i in range(len(null)):
        yp=y.copy()
        for idx in groups.values(): yp[idx]=rng.permutation(yp[idx])
        null[i]=np.corrcoef(wide.F13,yp)[0,1]
    p=float((1+np.sum(null>=overall))/(len(null)+1))
    result=dict(oof_days=len(daily),paired_cells=len(wide),
                paired_first=int(wide.section.eq('first').sum()),
                paired_second=int(wide.section.eq('second').sum()),
                correlations=dict(overall=overall,first=first,second=second),
                sign_agreement=sign,permutation_p_ge=p,
                passes_screen=bool(len(wide)>=30 and overall>=.35 and
                    wide.section.eq('second').sum()>=8 and second>0 and p<.01),
                sha256={str(z):sha(z) for z in [DATA/'train_X.csv',*paths]},
                hidden_labels_read=False,test_input_read=False)
    out=ROOT/'analysis/local/ec_shared_weather_residual'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
