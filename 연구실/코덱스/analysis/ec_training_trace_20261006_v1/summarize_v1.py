from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env,env_extra
import numpy as np,pandas as pd
L=ROOT/'연구실/코덱스/local'/H.name;D=H/'results_v4'
assert not (L/'worker.lock').exists();receipt=json.loads((D/'completion.json').read_text());assert receipt['fits']==6
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,digest in receipt['files'].items():assert sha(D/name)==digest
O=H/'summary_v1';O.mkdir(exist_ok=False)
p=pd.read_csv(D/'support_profiles.csv',float_precision='round_trip');p.to_csv(O/'raw_support_profiles.csv',index=False)
days=pd.read_csv(D/'support_days.csv',float_precision='round_trip');top=[]
for key,g in days.groupby(['context','query_farm','query_day','level']):
    a=g.sort_values('weight',ascending=False).head(10).copy();a['rank']=range(1,len(a)+1);top.append(a)
pd.concat(top,ignore_index=True).to_csv(O/'top_support_days.csv',index=False)
delta=pd.read_csv(D/'pair_weight_changes.csv',float_precision='round_trip');dc=delta.groupby(['context','query_farm','level','farm','day'],as_index=False)[['delta_weight','raw_contribution','centered_contribution']].sum();dc.to_csv(O/'pair_day_changes.csv',index=False)
phi=pd.read_csv(D/'weight_shapley_rows.csv',float_precision='round_trip');pc=phi.groupby(['query_farm','group','farm','day'],as_index=False)[['phi_weight','raw_contribution','centered_contribution']].sum();pc.to_csv(O/'group_shapley_days.csv',index=False)
split=pd.read_csv(D/'first_split_training.csv',float_precision='round_trip');b=split.set_index(['context','farm','tree','side'])
ss=[]
for key,g in split.groupby(['context','farm','feature']):
    good=g[g.side=='good'].set_index('tree');bad=g[g.side=='bad'].set_index('tree');assert good.index.tolist()==bad.index.tolist()
    ss.append(dict(context=key[0],farm=key[1],feature=key[2],trees=len(good),mean_good_branch_ec=float(good.mean_ec.mean()),mean_bad_branch_ec=float(bad.mean_ec.mean()),mean_branch_ec_difference=float((bad.mean_ec-good.mean_ec).mean()),mean_high_row_fraction_difference=float((bad.high_row_fraction-good.high_row_fraction).mean()),positive_branch_ec_trees=int((bad.mean_ec>good.mean_ec).sum())))
pd.DataFrame(ss).to_csv(O/'first_branch_associations.csv',index=False)
smooth=[];smoothdays=[]
for m in receipt['models']:
    name=m['name'];path=L/(name+'.npz');assert sha(path)==m['npz_sha']
    with np.load(path,allow_pickle=False) as z:
        y=z['train_y'];tyids=z['train_row_id'];qids=z['query_row_id'];qy=z['query_y'];w=z['weights'];raw=z['raw_et']
        tm=pd.DataFrame(dict(row_id=tyids,y=y));tm['farm']=tm.row_id.str[:3];tm['day']=tm.row_id.str[4:7].astype(int);ym=tm.groupby(['farm','day']).y.transform('mean').to_numpy()
        qm=pd.DataFrame(dict(row_id=qids));qm['farm']=qm.row_id.str[:3];qm['day']=qm.row_id.str[4:7].astype(int);qm['hour']=qm.row_id.str[8:10].astype(int)
        for (farm,day),ix in qm.groupby(['farm','day']).indices.items():
            ix=np.asarray(ix);ix=ix[np.argsort(qm.hour.iloc[ix])];assert qm.hour.iloc[ix].tolist()==list(range(24))
            block=w[ix];sw=.5*block+.5*np.cumsum(block,axis=0)/np.arange(1,25)[:,None];v=sw.mean(axis=0)
            check=np.mean(.5*raw[ix]+.5*np.cumsum(raw[ix])/np.arange(1,25));assert abs(v@y-check)<1e-10
            smooth.append(dict(context=name,farm=farm,day=int(day),prediction=float(v@y),true_day=float(qy[ix].mean()),high_row_weight=float(v[y>=1].sum()),high_day_weight=float(v[ym>=1].sum()),F13_weight=float(v[tm.farm.eq('F13')].sum()),support_days=len(tm.loc[v>0,['farm','day']].drop_duplicates())))
            a=tm[['farm','day']].copy();a['weight']=v;a['contribution']=v*y;a=a.groupby(['farm','day'],as_index=False)[['weight','contribution']].sum();a=a[a.weight>0];a['context']=name;a['query_farm']=farm;a['query_day']=int(day);a['weighted_ec']=a.contribution/a.weight;smoothdays.append(a)
pd.DataFrame(smooth).to_csv(O/'smooth_day_support_profiles.csv',index=False);pd.concat(smoothdays,ignore_index=True).to_csv(O/'smooth_support_days.csv',index=False)
summary=dict(status='COMPLETE_SUPPORT_SUMMARY',new_fit=0,files={f.name:sha(f) for f in O.glob('*.csv')},raw_vs_smooth_separated=True,physical_causality=False)
with (O/'completion.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
print(pd.DataFrame(smooth).to_string(index=False),flush=True)
