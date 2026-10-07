from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import pandas as pd,numpy as np
R=H/'results_v2';O=H/'results_v4';O.mkdir(exist_ok=False)
d=pd.read_csv(R/'days.csv',float_precision='round_trip');d['fanlow']=d.act_circfan_mean<10
out=[]
for (closed,ordinary),q in d.groupby(['closed','ordinary']):
    out.append(dict(closed=bool(closed),ordinary=bool(ordinary),days=len(q),ymean=float(q.ymean.mean()),pmean=float(q.pmean.mean()),bias=float(q.bias.mean()),rmse=float(np.sqrt(q.mse.mean())),sse=float(q.sse.sum()),y_range_mean=float((q.ymax-q.ymin).mean()),y_day_std=float(q.ymean.std()),p_day_std=float(q.pmean.std()),corr_daily=float(q.ymean.corr(q.pmean))))
pd.DataFrame(out).to_csv(O/'state_target_context.csv',index=False)
out=[]
for (c,f),q in d[d.ordinary].groupby(['closed','fanlow']):
    record=dict(closed=bool(c),fanlow=bool(f),days=len(q))
    for col in ['in_temp_mean','out_temp_mean','in_hum_mean','in_co2_mean','act_thermal_mean','act_shade_mean','act_heating_mean','act_co2_mean','act_fog_mean','ventzero','ymean','ystd','ymax','ymin']:record[col]=float(q[col].mean())
    for key,qq in q.groupby('farm'):record['farm_'+key]=len(qq)
    record['pass2']=int(q.phase.sum());record['miss_in_temp']=float(q.in_temp_missing.sum());record['miss_in_hum']=float(q.in_hum_missing.sum());record['miss_in_co2']=float(q.in_co2_missing.sum());out.append(record)
pd.DataFrame(out).to_csv(O/'fan_state_profiles.csv',index=False)
# Standardize four groups over common farm/yband/tempband cells. Descriptive overlap, no model.
b=pd.read_csv(R/'ordinary_days.csv',float_precision='round_trip');b['fanlow']=b.act_circfan_mean<10
out=[]
for fans in [True,False]:
    bs=b[b.fanlow==fans]
    for cols in [['farm'],['yband'],['tempband'],['farm','phase']]:
        common=[]
        for k,q in bs.groupby(cols):
            if set(q.closed)=={True,False}:
                c=q[q.closed];r=q[~q.closed];common.append((len(c),len(r),c.mse.mean(),r.mse.mean(),c.bias.mean(),r.bias.mean()))
        if not common:continue
        nc=sum(z[0] for z in common);nr=sum(z[1] for z in common)
        cm=sum(z[1]/nr*z[2] for z in common);rm=sum(z[1]/nr*z[3] for z in common)
        out.append(dict(fanlow=fans,strata='/'.join(cols),closed_supported=nc,rest_supported=nr,common_strata=len(common),closed_rmse_restweights=float(np.sqrt(cm)),rest_rmse=float(np.sqrt(rm))))
pd.DataFrame(out).to_csv(O/'fan_stratification.csv',index=False)
# Unique-rest matching instead of treating repeatedly used controls as independent evidence.
m=pd.read_csv(R/'matched_pairs.csv',float_precision='round_trip');ms=[]
for kind,q in m.groupby('kind'):
    used=set();sel=[]
    for row in q.sort_values(['distance','farm','closed_day']).itertuples():
        key=(row.farm,row.rest_day)
        if key not in used:used.add(key);sel.append(row)
    ms.append(dict(kind=kind,pairs=len(sel),method='keep lowest-distance pair per reused control; sensitivity only',closed_rmse=float(np.sqrt(np.mean([t.closed_mse for t in sel]))),rest_rmse=float(np.sqrt(np.mean([t.rest_mse for t in sel])))))
(O/'unique_match_sensitivity.json').write_text(json.dumps(ms,indent=2),encoding='utf-8')
# Per seed prefix shows whether wider .9 false positives already exist at midnight.
rp=pd.read_csv(R/'row_predictions.csv',float_precision='round_trip');counts=[]
for gr,q in rp.groupby('group'):
    for hr in [0,6,12,23]:
        p=q[q.hour<=hr].groupby(['farm','day']).baseline.mean()
        counts.append(dict(group=gr,hour=hr,days=len(p),selected09=int((p>=.9).sum()),selected1=int((p>=1).sum())))
pd.DataFrame(counts).to_csv(O/'falsehigh_onset.csv',index=False)
print(json.dumps(out,ensure_ascii=False));print('COMPLETE_V4')
