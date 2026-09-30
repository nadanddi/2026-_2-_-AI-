"""Post-hoc diagnostic only: held-out errors by exact repeated-weather exposure."""
import run_benchmark as b
import pandas as pd
import numpy as np
import json

def main():
    _,full,lab,lock,sigs,fds=b.prepare()
    d=pd.read_csv(b.OUT/'oof_predictions.csv')
    tags=[]
    for name,i,days in fds:
        tr=lab[b.near_mask(lab,days|lock)]
        seen={sigs[k]['weather'] for k in set(tr[['farm','day']].itertuples(index=False,name=None))}
        va=lab[[(f,int(day)) in days for f,day in zip(lab.farm,lab.day)]]
        for f,day in va[['farm','day']].drop_duplicates().itertuples(index=False,name=None):
            tags.append((name,i,f,int(day),sigs[(f,int(day))]['weather'] in seen))
    t=pd.DataFrame(tags,columns=['validator','validation_fold','farm','day','weather_repeated_in_training'])
    z=d.merge(t,on=['validator','validation_fold','farm','day'],validate='many_to_one')
    rec=[]
    for (vn,repeated),g in z.groupby(['validator','weather_repeated_in_training']):
        for c in ['mean','farm_mean','ridge','raw_et','full_et','r3','v2']:
            rec.append({'validator':vn,'weather_repeated':bool(repeated),'model':c,'n':len(g),
                        'unique_days':len(g[['farm','day']].drop_duplicates()),'rmse':b.score(g.sub_ec,g[c]),
                        'late_fraction':float(g.day.ge(179).mean())})
    pd.DataFrame(rec).to_csv(b.HERE/'weather_repeat_error.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(rec).query("model=='v2'").to_string(index=False))

if __name__=='__main__':main()
