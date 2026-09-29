"""Descriptive training-only follow-up of EC exceptions. No fitting or edits to data."""
import json
import numpy as np
import pandas as pd
from profile_tabular import load, OUT, TARGETS


def main():
    x, y = load('train_X.csv'), load('train_y.csv')
    d = x.merge(y[['row_id']+TARGETS], on='row_id', how='left', validate='one_to_one')
    cols = ['in_temp','in_hum','in_co2','act_circfan','act_vent','act_heating','act_co2','act_fog','act_shade','act_thermal']+TARGETS
    result = {}
    for farm, lo, hi in [('F13',124,132), ('F47',24,100)]:
        g = d[d.farm.eq(farm)].sort_values('time').copy()
        z = g[g.day.between(lo,hi)]
        z[['row_id','day','hour']+cols].to_csv(OUT/f'exception_hourly_{farm}.csv',index=False)
        daily = z.groupby('day')[cols].mean()
        daily.to_csv(OUT/f'exception_daily_{farm}.csv')
        missing = z.groupby('day')[cols].agg(lambda s:int(s.isna().sum()))
        missing.to_csv(OUT/f'exception_missing_{farm}.csv')
        boundaries=[]
        indexed=g.set_index(['day','hour'])
        for day in range(lo,hi+1):
            if (day,0) not in indexed.index or (day-1,23) not in indexed.index: continue
            a,b=indexed.loc[(day-1,23)],indexed.loc[(day,0)]
            row={'day':day}
            for c in cols:
                row[c+'_before']=a[c];row[c+'_after']=b[c];row[c+'_delta']=b[c]-a[c]
            boundaries.append(row)
        pd.DataFrame(boundaries).to_csv(OUT/f'exception_boundaries_{farm}.csv',index=False)
        # A non-missing zero and a missing observation are counted separately.
        result[farm]={'window':[lo,hi], 'rows':len(z),'missing':z[cols].isna().sum().to_dict()}
        if farm=='F13':
            result[farm]['daily_126_129']=daily.loc[126:129].reset_index().to_dict('records')
            result[farm]['boundaries_128_129']=[b for b in boundaries if b['day'] in [128,129]]
        else:
            p=pd.read_csv(OUT/'matched_weather_pairs_F47.csv')
            p=p[p.gap.eq(1)].sort_values('first_day')
            result[farm]['pair_deltas_24_100']=p[p.first_day.between(lo,hi)][['first_day','second_day','sub_ec_delta','act_circfan_delta','act_heating_delta','in_temp_delta','in_co2_delta']].to_dict('records')
        # Across all days: missing internal observations at current or previous
        # hour versus midnight EC jumps. Structural absent columns excluded.
        internal=['in_temp','in_hum','in_co2']
        consecutive=g.time.diff().eq(1)
        g['jump']=g.sub_ec.diff().abs().where(consecutive)
        miss=g[internal].isna().any(axis=1)
        g['boundary_missing']=miss | miss.shift(1,fill_value=False)
        midnight=g[g.hour.eq(0)&g.jump.notna()]
        result[farm]['midnight_missing']=[dict(missing_either_endpoint=bool(k),n=len(v),large_jump_n=int(v.jump.gt(.2).sum())) for k,v in midnight.groupby('boundary_missing')]
    def clean(v):
        if isinstance(v,dict):return {k:clean(w) for k,w in v.items()}
        if isinstance(v,list):return [clean(w) for w in v]
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
        if isinstance(v,np.generic):return v.item()
        return v
    result=clean(result)
    (OUT/'exception_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
