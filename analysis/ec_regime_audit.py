"""Descriptive regime analysis only; full-day aggregates are NOT model inputs."""
import json
import pandas as pd
import numpy as np
from profile_tabular import load,OUT,TARGETS


def main():
    x=load('train_X.csv'); y=load('train_y.csv')
    d=x.merge(y[['row_id']+TARGETS],on='row_id',how='left')
    d=d[d.sub_ec.notna()].copy()
    d['high']=d.sub_ec.gt(1)
    d['period']=(d.day//30)*30
    d['low_fan']=d.act_circfan.le(5)
    summary={}
    for f,g in d.groupby('farm'):
        total_sse=((g.sub_ec-g.sub_ec.mean())**2).sum()
        z=dict(n=len(g),n_high=int(g.high.sum()),high_fraction=float(g.high.mean()),
            high_share_of_constant_baseline_sse=float(((g.loc[g.high,'sub_ec']-g.sub_ec.mean())**2).sum()/total_sse),
            by_low_fan=g.groupby('low_fan').agg(n=('high','size'),high_rate=('high','mean'),ec_mean=('sub_ec','mean')).to_dict('index'),
            conditional_by_period=[])
        for (period,low),a in g.groupby(['period','low_fan']):
            z['conditional_by_period'].append(dict(period=int(period),low_fan=bool(low),n=len(a),high_rate=float(a.high.mean()),ec_mean=float(a.sub_ec.mean())))
        daily=g.groupby('day').agg(ec_mean=('sub_ec','mean'),ec_max=('sub_ec','max'),
            fan_mean=('act_circfan','mean'),fan_max=('act_circfan','max'),outside_mean=('out_temp','mean'),
            heat_mean=('act_heating','mean'),vent_mean=('act_vent','mean'))
        z['daily_correlations']=daily.corr()['ec_mean'].to_dict()
        z['high_days']=int(daily.ec_max.gt(1).sum())
        z['daily_fan_condition']=daily.groupby(daily.fan_mean.le(5)).agg(n=('ec_mean','size'),mean_ec=('ec_mean','mean')).to_dict('index')
        summary[f]=z
    (OUT/'ec_regime_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
