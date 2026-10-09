"""Actual frozen-model prediction checks on both farms and both periods."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'집/코덱스/local'/H.name
import recipe_ec_v1 as C
from predict_model_v1 import predict_bundle
import pandas as pd,numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
D=L/'model_full_clean_v1';reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'))
public={rid for r in reg['folds'] for rid in r['query_ids']}
x=pd.read_csv(L/'dataset_full_train/train_X_clean_v1.csv');x=x[x.row_id.isin(public)].copy();meta=C.identify(x).sort_values(['farm','day','hour']);meta['pass2']=meta.day>=179
chosen=meta[['farm','day','pass2']].drop_duplicates().groupby(['farm','pass2']).head(2)
assert len(chosen)==8
keys=set(chosen[['farm','day']].itertuples(index=False,name=None));probe=x[[(f,int(d)) in keys for f,d in zip(C.identify(x).farm,C.identify(x).day)]].copy();assert len(probe)==192
m=C.identify(probe);base=predict_bundle(D,probe).set_index('row_id').prediction
checks=[]
for (farm,pass2),g in chosen.groupby(['farm','pass2']):
    day=int(g.day.max())
    for hour in (5,17):
        allowed=(m.farm==farm)&((m.day*24+m.hour)<=day*24+hour);ids=m.loc[allowed,'row_id'];assert len(ids)==24+hour+1
        changed=probe.copy();changed.loc[~allowed,C.RAW]=changed.loc[~allowed,C.RAW]*17+311
        alt=predict_bundle(D,changed).set_index('row_id').prediction
        diff=float(np.max(abs(base.loc[ids]-alt.loc[ids])));assert diff<1e-10
        prefix=predict_bundle(D,probe.loc[allowed]).set_index('row_id').prediction
        dropdiff=float(np.max(abs(base.loc[ids]-prefix.loc[ids])));assert dropdiff<1e-10
        checks.append(dict(farm=farm,pass2=bool(pass2),day=day,hour=hour,allowed_rows=len(ids),future_other_farm_perturbed_maxdiff=diff,future_other_farm_deleted_maxdiff=dropdiff))
        print('CAUSAL_PASS',farm,'pass2',bool(pass2),'day',day,'hour',hour,flush=True)
shuffled=predict_bundle(D,probe.sample(frac=1,random_state=151)).set_index('row_id').prediction;reorder=float(np.max(abs(base-shuffled.reindex(base.index))));assert reorder<1e-10
weather=probe.copy();weather[C.W]=9999.;weather_pred=predict_bundle(D,weather).set_index('row_id').prediction;w=float(np.max(abs(base-weather_pred.reindex(base.index))));assert w<1e-10
out=dict(status='PASS',probe_rows=192,probe_days=8,checks=checks,input_order_maxdiff=reorder,query_weather_change_maxdiff=w,script_sha=sha(Path(__file__)),model_manifest_sha=sha(D/'model_manifest_v1.json'),future_hours_tested=[5,17],sample_scope='public retained training inputs, four farm/period strata, no validation labels read')
with (H/'final_model_causality_v1.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('ALL_MODEL_CAUSAL_CHECKS_PASS',flush=True)
