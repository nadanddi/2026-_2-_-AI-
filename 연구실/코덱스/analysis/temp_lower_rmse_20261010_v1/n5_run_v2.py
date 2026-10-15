"""Frozen-parent zero-fit pilot: causal day-prefix minimum gates IRR correction."""
from pathlib import Path
import sys,json,hashlib,argparse
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'.analysis-tools/python'))
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--audit',action='store_true');ap.add_argument('--first',action='store_true');args=ap.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=json.loads((H/'n4_registration_v2.json').read_text(encoding='utf-8'))
prs=hashlib.sha256(json.dumps(parent,sort_keys=True).encode()).hexdigest()
assert parent['source_sha']==sha(H/'n4_run_v2.py')
prep=json.loads((H.parent/'temp_round9_reproduction_20261008_v1/preparation_v1.json').read_text(encoding='utf-8'))
data=Path(prep['temp_code']).parent/'data'
for n,v in parent['data_sha'].items():assert sha(data/n)==v
P=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n4_v1'
O=P.parent/'n5_v1';O.mkdir(parents=True,exist_ok=True)
raw=pd.concat([pd.read_csv(data/n,usecols=['row_id','in_temp']) for n in ['train_X.csv','test_X.csv']],ignore_index=True)
raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].copy()
assert raw.row_id.is_unique
assert raw.row_id.str.match(r'^F\d{2}_\d{3}_\d{2}$').all()
assert np.isfinite(raw.in_temp.dropna().to_numpy(float)).all()
parents={}
for fd in parent['folds']:
    name=fd['set']+'_'+str(fd['fold']);fn=P/(name+'.npz');mn=fn.with_suffix('.json')
    m=json.loads(mn.read_text(encoding='utf-8'))
    assert m['recipe_sha']==prs and m['npz_sha']==sha(fn) and m['query_ids']==fd['query_ids'] and m['train_ids']==fd['train_ids']
    parents[name]={'npz_sha':sha(fn),'receipt_sha':sha(mn)}

def gates(input_frame):
    assert 'sub_temp' not in input_frame and 'sub_ec' not in input_frame
    a=input_frame[['row_id','in_temp']].copy();a['farm']=a.row_id.str[:3]
    a['day']=a.row_id.str[4:7].astype(int);a['hour']=a.row_id.str[8:10].astype(int)
    a=a.sort_values(['farm','day','hour']).set_index('row_id')
    a['prefix_min']=a.groupby(['farm','day']).in_temp.transform(lambda v:v.expanding(min_periods=1).min())
    a['q']=np.where(a.prefix_min.isna(),1.,np.clip((a.prefix_min-8)/2,0,1))
    a['g']=np.where(a.in_temp.isna(),1.,np.clip((a.in_temp-8)/2,0,1))
    assert (a.q<=a.g).all()
    return a

gf=gates(raw)
packet=gf.groupby(['farm','day']).hour.agg(['size','nunique'])
assert (packet['size']==24).all() and (packet['nunique']==24).all() and gf.hour.between(0,23).all()
assert (gf.groupby(['farm','day']).q.diff().fillna(0)<=0).all()
recipe={'candidate':'N5_PREFIX_MIN_IRR4','source_sha':sha(__file__),'parent_recipe_sha':prs,'parent_registration_sha':sha(H/'n4_registration_v2.json'),'parents':parents,'data_sha':parent['data_sha'],'folds':parent['folds'],'alpha':.025/7,'bootstrap_n':20000,'bootstrap_seed':20261010,'bootstrap':'pooled farm/rawday//5, deltaRMSE','formula':'0.4*BASE+(.2+.4*(1-g_current))*(CODEX+q_prefix*(IRR-CODEX))+.4*g_current*PFN','q':'clip((min finite in_temp hour0..current -8)/2,0,1); no valid past ->1','n_estimators_pfn':4,'pfn_context_seed':49,'pfn_samples':1,'new_fit':0,'adoption':False,'posthoc_candidate_number':7,'raw_finite_or_nan':True,'complete24_hour_days':len(packet),'parent_source_receipt_sha':sha(H/'n4_source_receipt_v1.json')}
rs=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
def save(name,obj):
    with (H/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
regp=H/'n5_registration_v2.json'
if regp.exists():assert json.loads(regp.read_text(encoding='utf-8'))==recipe
else:save(regp.name,recipe)

if args.audit:
    cells=0
    for cut in range(23):
        changed=raw.copy();changed.loc[changed.row_id.str[8:10].astype(int)>cut,'in_temp']=-12345
        other=gates(changed);ids=gf.index[gf.hour<=cut]
        assert np.array_equal(gf.loc[ids,'q'],other.loc[ids,'q']);cells+=len(ids)
    for f in ['F13','F47']:
        changed=raw.copy();changed.loc[changed.row_id.str[:3]!=f,'in_temp']=-12345
        other=gates(changed);ids=gf.index[gf.farm==f]
        assert np.array_equal(gf.loc[ids,'q'],other.loc[ids,'q']);cells+=len(ids)
    other=gates(raw.sample(frac=1,random_state=20261010));assert np.array_equal(gf.q,other.loc[gf.index,'q']);cells+=len(gf)
    for label in ['sub_temp','sub_ec']:
        try:gates(raw.assign(**{label:np.nan}))
        except AssertionError:label_reject=True
        else:raise AssertionError('label column not rejected')
    synth=pd.DataFrame({'row_id':['F13_001_'+str(h).zfill(2) for h in range(24)],'in_temp':[np.nan,10.,np.nan,7.,12.]+[np.nan]*19})
    sg=gates(synth)
    assert np.array_equal(sg.q.to_numpy(),np.array([1.,1.,1.]+[0.]*21))
    assert np.array_equal(sg.prefix_min.iloc[1:3].to_numpy(),[10.,10.])
    assert (sg.q.diff().fillna(0)<=0).all()
    # Model training is immutable parent reuse; gate is never a training feature.
    save('n5_audit_v2.json',{'recipe_sha':rs,'new_fit':0,'gate_future_otherfarm_shuffle_comparison_cells':cells,'all23_hour_cutoffs':True,'label_columns_rejected':label_reject,'parent_models_unchanged':True,'full_prediction_causality_test':False})
    print('N5_AUDIT',cells,rs,flush=True);raise SystemExit(0)

assert (H/'N4_FINAL_REVIEW_v1.md').exists() and (H/'N5_PLAN_REVIEW_v1.md').exists()
assert json.loads((H/'n5_audit_v2.json').read_text(encoding='utf-8'))['recipe_sha']==rs
labels=pd.read_csv(data/'train_y.csv',usecols=['row_id','sub_temp']).set_index('row_id').sub_temp
for fd in (parent['folds'][:1] if args.first else parent['folds']):
    name=fd['set']+'_'+str(fd['fold']);fn=O/(name+'.npz');mn=fn.with_suffix('.json')
    if fn.exists():
        m=json.loads(mn.read_text(encoding='utf-8'));assert m['recipe_sha']==rs and m['npz_sha']==sha(fn);continue
    z=np.load(P/(name+'.npz'),allow_pickle=True);ids=z['row_id'].tolist();assert ids==fd['query_ids']
    q=gf.loc[ids,'q'].to_numpy();g=gf.loc[ids,'g'].to_numpy();assert np.array_equal(g,z['gate'])
    p=.4*z['BASE']+(.2+.4*(1-g))*(z['CODEX']+q*(z['IRR']-z['CODEX']))+.4*g*z['PFN']
    assert np.isfinite(p).all() and np.array_equal(p[q==0],z['BASELINE'][q==0])
    assert np.array_equal(p[q==g],z['N4'][q==g])
    assert np.array_equal(p[g==0],z['BASELINE'][g==0])
    assert np.allclose(p-z['BASELINE'],(.6-.4*g)*q*(z['IRR']-z['CODEX']),rtol=0,atol=1e-13)
    np.savez(fn,row_id=z['row_id'],BASELINE=z['BASELINE'],N5=p,q=q,g=g,prefix_min=gf.loc[ids,'prefix_min'].to_numpy())
    m={'recipe_sha':rs,'npz_sha':sha(fn),'parent_npz_sha':sha(P/(name+'.npz')),'query_ids':ids,'new_fit':0,'q0_rows':int((q==0).sum()),'q_reduced_rows':int((q<g).sum())}
    with mn.open('x',encoding='utf-8') as f:json.dump(m,f,ensure_ascii=False,indent=2)
    y=labels.loc[ids].to_numpy();print('N5_FOLD',name,float(np.sqrt(np.mean((z['BASELINE']-y)**2))),float(np.sqrt(np.mean((p-y)**2))),m['q0_rows'],flush=True)
if args.first:raise SystemExit(0)
results={}
for s in ['DIAG10','EXT10','EXT12']:
    ids=[];a=[];b=[]
    for fd in [f for f in parent['folds'] if f['set']==s]:
        fn=O/(s+'_'+str(fd['fold'])+'.npz');m=json.loads(fn.with_suffix('.json').read_text(encoding='utf-8'));assert m['recipe_sha']==rs and m['npz_sha']==sha(fn)
        z=np.load(fn,allow_pickle=True);ids.extend(z['row_id'].tolist());a.extend(z['BASELINE']);b.extend(z['N5'])
    a=np.array(a);b=np.array(b);y=labels.loc[ids].to_numpy();ra=float(np.sqrt(np.mean((a-y)**2)));rb=float(np.sqrt(np.mean((b-y)**2)))
    m={'base_rmse':ra,'candidate_rmse':rb,'pct':100*(rb/ra-1),'better':rb<ra,'rows':len(y)}
    if s=='DIAG10':
        _,inv=np.unique([i[:3]+'_'+str(int(i[4:7])//5) for i in ids],return_inverse=True)
        n=inv.max()+1;sa=np.bincount(inv,weights=(a-y)**2);sb=np.bincount(inv,weights=(b-y)**2);sz=np.bincount(inv)
        rng=np.random.default_rng(20261010);ds=[]
        for k in range(20):
            draw=rng.integers(n,size=(1000,n));den=sz[draw].sum(1);ds.extend((np.sqrt(sb[draw].sum(1)/den)-np.sqrt(sa[draw].sum(1)/den)).tolist())
        m.update(p_worse=float(np.mean(np.array(ds)>=0)),ci=list(map(float,np.quantile(ds,[recipe['alpha'],1-recipe['alpha']]))))
    results[s]=m
passed=all(m['better'] for m in results.values()) and results['DIAG10']['p_worse']<recipe['alpha'] and results['DIAG10']['ci'][1]<0
save('n5_results_v2.json',{'results':results,'pilot_pass':passed,'adoption':False,'new_fit':0,'recipe_sha':rs})
print('N5_RESULTS',json.dumps(results),passed,flush=True)
