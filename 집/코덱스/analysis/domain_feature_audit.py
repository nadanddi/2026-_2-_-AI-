"""Training-only descriptive check of literature-motivated features; no fitting."""
import json
import numpy as np
from profile_tabular import load, OUT, TARGETS


def main():
    x,y=load('train_X.csv'),load('train_y.csv')
    d=x.merge(y[['row_id']+TARGETS],on='row_id',how='left',validate='one_to_one')
    result={}
    for farm in ['F13','F47']:
        g=d[d.farm.eq(farm)].sort_values('time').copy()
        es=lambda t: .6108*np.exp(17.27*t/(t+237.3))
        g['vpd']=es(g.in_temp)*(1-g.in_hum/100)
        g['vapour_difference']=es(g.in_temp)*g.in_hum/100-es(g.out_temp)*g.out_hum/100
        g['radiation_open_proxy']=g.out_rad*g.act_shade/100
        g['radiation_closed_interaction']=g.out_rad*(1-g.act_shade/100)
        g['vpd_radiation']=g.vpd*g.out_rad
        variables=['in_temp','in_hum','out_rad','vpd','vapour_difference','radiation_open_proxy','radiation_closed_interaction','vpd_radiation']
        out={}
        for c in variables:
            z=g[[c,'sub_temp','sub_ec','day','hour']].dropna()
            # Retrospective demeaning ONLY, not a feature for prediction.
            within=z[[c]+TARGETS]-z.groupby('day')[[c]+TARGETS].transform('mean')
            entry={'n':len(z)}
            for target in TARGETS:
                entry[target+'_pooled_r']=float(z[c].corr(z[target]))
                entry[target+'_within_day_r']=float(within[c].corr(within[target]))
            if c=='vpd':
                entry['period_ec_r']={str(int(k)):float(v[c].corr(v.sub_ec)) for k,v in z.groupby((z.day//30)*30) if len(v)>100}
            out[c]=entry
        result[farm]=out
    (OUT/'domain_feature_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
