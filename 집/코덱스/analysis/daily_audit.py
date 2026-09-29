"""Daily boundaries, periodicity, repeated weather and target/input consistency."""
import json
import numpy as np
import pandas as pd
from profile_tabular import load, OUT, TARGETS


def main():
    x=load('train_X.csv'); y=load('train_y.csv')
    d=x.merge(y[['row_id']+TARGETS],on='row_id',how='left',validate='one_to_one')
    out={}
    for farm in ['F13','F47']:
        g=d[d.farm==farm].sort_values('time').copy()
        result={}
        for c in ['in_temp','out_temp','out_hum','out_rad','in_hum','in_co2']+TARGETS:
            dif=g[c].diff().where(g.time.diff()==1)
            byhour=dif.abs().groupby(g.hour).agg(['count','mean','median','max'])
            threshold=.2 if c=='sub_ec' else 3 if c in ['sub_temp','in_temp','out_temp'] else None
            stat={'difference_by_hour':byhour.to_dict('index')}
            if threshold is not None:
                big=dif.abs()>threshold
                stat['large_jump_threshold']=threshold
                stat['large_jump_count']=int(big.sum())
                stat['midnight_jump_count']=int((big&g.hour.eq(0)).sum())
            daily=g.groupby('day')[c].mean()
            stat['daily_autocorr']={str(l):float(daily.corr(daily.reindex(daily.index-l).set_axis(daily.index))) for l in [1,2,3,4,5,6,7,10,14,21,28]}
            # Descriptive only: R2 of residue means, not a predictive validation.
            residues={}
            for period in range(2,32):
                p=daily.groupby(daily.index%period).transform('mean')
                residues[str(period)]=float(1-((daily-p)**2).sum()/((daily-daily.mean())**2).sum())
            stat['best_residue_r2_descriptive']=sorted(residues.items(),key=lambda z:z[1],reverse=True)[:5]
            result[c]=stat
        # Exact 24-hour vectors, require all hours and nonmissing.
        repeat={}
        for c in ['out_temp','out_hum','out_rad','out_wspd','in_temp']+TARGETS:
            table=g.pivot(index='day',columns='hour',values=c).dropna()
            h=pd.util.hash_pandas_object(table,index=False)
            counts=h.value_counts()
            groups=[h.index[h.eq(k)].tolist() for k,n in counts.items() if n>1]
            repeat[c]=dict(complete_days=len(table),duplicated_days=int(h.duplicated(False).sum()),groups=groups[:20])
        result['repeated_daily_vectors']=repeat
        # Relation to day-local demeaned input removes between-day confounding.
        anomalies=g[['in_temp','out_temp','out_rad','in_hum','act_heating']+TARGETS].copy()
        anomalies-=anomalies.groupby(g.day).transform('mean')
        result['within_day_correlations']=anomalies.corr()[TARGETS].to_dict()
        out[farm]=result
    (OUT/'daily_audit.json').write_text(json.dumps(out,indent=2,allow_nan=True),encoding='utf8')
    for f,r in out.items():
        print(f)
        for c in ['in_temp','out_temp']+TARGETS:
            z=r[c]
            print(c,{k:v for k,v in z.items() if k!='difference_by_hour'})
        print('DAILY REPEATS',r['repeated_daily_vectors'])
        print('WITHIN DAY',r['within_day_correlations'])


if __name__=='__main__':main()
