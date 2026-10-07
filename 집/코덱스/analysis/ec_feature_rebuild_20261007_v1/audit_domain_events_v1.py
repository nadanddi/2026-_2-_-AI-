"""Independent synthetic event/current/lag/rolling arithmetic, no fit or truth."""
from domain_features_v2 import build_domain,event_response
from blk_baseline_data_v1 import pd,np,HERE,sha
from blk_context_v1 import RAW
import json,math
from checkpoint_v1 import atomic

hours=[0,1,2,3,4,6,7]
act=[0,0,10,10,0,20,20]
temp=[20,21,22,23,24,25,26]
hum=[50,51,54,55,53,60,62]
co2=[400,402,410,408,403,420,430]
rows=[]
for i,h in enumerate(hours):
    row={c:0. for c in RAW}
    row.update(row_id=f'F13_001_{h:02d}',in_temp=temp[i],in_hum=hum[i],in_co2=co2[i],
               out_temp=12.,out_hum=40.,act_fog=float(act[i]),act_co2=float(act[i]),act_vent=float(act[i]))
    rows.append(row)
frame=pd.DataFrame(rows)
features,_,_=build_domain(frame)
vap=[.6108*math.exp(17.27*t/(t+237.3))*(1-u/100) for t,u in zip(temp,hum)]
def reference_event(values):
    anchor=None;out=[]
    for i,h in enumerate(hours):
        if i==0 or h-hours[i-1]!=1:
            anchor=None;out.append(float('nan'));continue
        if act[i]!=act[i-1]:anchor=values[i-1]
        out.append(values[i]-anchor if anchor is not None else float('nan'))
    return out
expected={
    'humidity_gap__vent_event_response':reference_event([u-40 for u in hum]),
    'humidity_gap__fog_event_response':reference_event([u-40 for u in hum]),
    'fog_vpd__event_response':reference_event(vap),
    'co2_dose_response__onset_delta':[np.nan,np.nan,8.,np.nan,np.nan,np.nan,np.nan],
    'co2_dose_response__offset_delta':[np.nan,np.nan,np.nan,np.nan,-5.,np.nan,np.nan],
    'act_fog__temperature_response':reference_event(temp),
    'act_fog__humidity_response':reference_event(hum),
    'act_fog__vpd_response':reference_event(vap),
}
checks=0
def compare(a,b):
    global checks
    assert (math.isnan(a) and math.isnan(b)) or abs(a-b)<1e-9,(a,b)
    checks+=1
for origin,values in expected.items():
    series=dict(zip(hours,values))
    for h in hours:
        rid=f'F13_001_{h:02d}'
        compare(float(features.loc[rid,origin]),series[h])
        for w in [1,2,3,4,6]:
            past=series.get(h-w,np.nan)
            compare(float(features.loc[rid,f'{origin}__lag{w}']),past)
            compare(float(features.loc[rid,f'{origin}__rate{w}']),(series[h]-past)/w)
            vals=[series[j] for j in range(max(0,h-w+1),h+1) if j in series and math.isfinite(series[j])]
            count=len(vals)
            total=math.fsum(vals)
            mean=total/count if count else np.nan
            manual={'count':count,'sum':total if count else np.nan,'mean':mean,
                    'std':math.sqrt(math.fsum((v-mean)**2 for v in vals)/count) if count else np.nan}
            for op,value in manual.items():compare(float(features.loc[rid,f'{origin}__{op}{w}']),value)
        compare(float(features.loc[rid,origin+'__d2']),series[h]-2*series.get(h-1,np.nan)+series.get(h-2,np.nan))
np.testing.assert_allclose(features['sealed_run'],[1,2,0,0,1,0,0],atol=0,rtol=0)
missing=event_response(pd.DataFrame({'hour':[0,1,2,3,4],'act_fog':[0,10,10,np.nan,10]}),'act_fog',[50,52,53,54,55])
np.testing.assert_allclose(missing,[np.nan,2,3,np.nan,np.nan],rtol=0,atol=0,equal_nan=True)
missing_env=event_response(pd.DataFrame({'hour':[0,1,2,3,4],'act_fog':[0,10,10,10,10]}),'act_fog',[50,52,np.nan,54,55])
np.testing.assert_allclose(missing_env,[np.nan,2,np.nan,np.nan,np.nan],rtol=0,atol=0,equal_nan=True)
for h in hours:
    short,_,_=build_domain(frame[frame.row_id.str[-2:].astype(int)<=h])
    np.testing.assert_allclose(short.loc[f'F13_001_{h:02d}'],features.loc[f'F13_001_{h:02d}'],rtol=0,atol=0,equal_nan=True)
receipt={'status':'PASS','event_series':8,'independent_current_lag_rate_d2_rolling_scalar_checks':checks,
         'synthetic_prefix_checks':len(hours),'missing_actuator_and_environment_checks':2,'sealed_run_gap_check':1,
         'code_sha256':sha(__file__),'features_code_sha256':sha(HERE/'domain_features_v2.py'),
         'heldout_truth_loaded':False,'model_fit':False,'limits':['Synthetic arithmetic only; real input preparation audit separate']}
out=HERE/'DOMAIN24_event_audit_v1.json';assert not out.exists()
atomic(out,json.dumps(receipt,ensure_ascii=False,allow_nan=False))
print(f'Domain event independent audit PASS: {checks} scalar checks',flush=True)
