"""One frozen irradiance candidate, compatible current CODEX component baseline."""
from pathlib import Path
import sys
TASK=Path(__file__).resolve().parent;BASE_HOME=TASK.parent/'temp_repair_20261010_v1';sys.path.insert(0,str(BASE_HOME))
from runtime_v1 import *
HERE=TASK
from n3_model_v2 import predict as n3predict
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import argparse,time
ap=argparse.ArgumentParser();ap.add_argument('--first',action='store_true');args=ap.parse_args()
IRR=[];FEATURES=FEATURE_COLUMNS
ALPHA=.025/5
def save(name,obj):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
REG=json.loads((BASE_HOME/'registration_v1.json').read_text(encoding='utf-8'))
assert sha(BASE_HOME/'run_v2.py')==REG['source_sha'] and sha(BASE_HOME/'runtime_v1.py')==REG['module_sha']
assert REG['features']==FEATURE_COLUMNS and REG['physics']==PHYSICS_COLUMNS
assert REG['data_sha']==P['data_sha']=={n:sha(DATA/n) for n in REG['data_sha']}
parent_sha=hashlib.sha256(json.dumps(REG,sort_keys=True).encode()).hexdigest()
IOUT=ROOT/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n3_target_v1';IOUT.mkdir(parents=True,exist_ok=True)

def features(raw=None,mask=True):
    tx,_,sx=masks(RAW if raw is None else raw) if mask else loader(raw)
    result=build_features(tx,sx).set_index('row_id')
    return result

cf=features();source=features(mask=False);checks=eq(cf,cframes(),list(dict.fromkeys(FEATURE_COLUMNS+PHYSICS_COLUMNS)))
assert (cf.groupby(['farm','day']).size()==24).all() and (cf.groupby(['farm','day']).hour.nunique()==24).all()
for farm in common.TARGET_FARMS:
    for cut in [5,17]:
        raw=loader();sx=raw[2];days=sorted(sx.loc[sx.farm==farm,'day'].unique());T=24*days[len(days)//2]+cut
        sx.loc[(sx.farm==farm)&(sx.t>T),common.USABLE]=2222
        compare=features(raw,False);ids=sx.loc[(sx.farm==farm)&(sx.t<=T),'row_id']
        checks+=eq(source.loc[ids],compare.loc[ids],FEATURES)
        checks+=eq(cf,features(raw),FEATURES)
    raw=loader();sx=raw[2];sx.loc[sx.farm!=farm,common.USABLE]=-3333
    ids=sx.loc[sx.farm==farm,'row_id'];checks+=eq(source.loc[ids],features(raw,False).loc[ids],FEATURES)
sh=tuple(d.sample(frac=1,random_state=9109).reset_index(drop=True) for d in loader())
checks+=eq(source,features(sh,False).loc[source.index],FEATURES)
recipe={'name':'N3_TWOHEAD','features':FEATURES,'physics':PHYSICS_COLUMNS,'seed':9109,'alpha':ALPHA,'parent_recipe_sha':parent_sha,'source_sha':sha(__file__),'runtime_sha':sha(BASE_HOME/'runtime_v1.py'),'formula':'level_hat0+raw_shape_h-raw_shape0','model_source_sha':sha(HERE/'n3_model_v2.py'),'level_params':[120,.03,8,12,15],'shape_params':[220,.035,12,100,15],'ridge_alpha':100,'shape_scale_with_mean':False,'shape_intercept':False,'data_sha':REG['data_sha'],'baseline_exact_probe':True,'hour_grid':'full 0..23, raw farm/day midnight anchor','clip':None,'bootstrap':20000,'bootstrap_seed':20261010,'ci_quantiles':[ALPHA,1-ALPHA],'full_mixture_adoption':False}
recipe_sha=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
regp=HERE/'n3_registration_v1.json'
if regp.exists():assert json.loads(regp.read_text(encoding='utf-8'))==recipe
else:save('n3_registration_v1.json',recipe)
print('N3_AUDIT_REGISTERED',checks,recipe_sha,flush=True)
labels=RAW[1].set_index('row_id').sub_temp

def lab(ids):
    a=cf.loc[ids].reset_index();a['sub_temp']=labels.loc[ids].to_numpy();return a

def fit(tr,va,w,cols):
    lin=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.0))
    lin.fit(tr[PHYSICS_COLUMNS],tr.sub_temp,ridge__sample_weight=w)
    mod=LGBMRegressor(n_estimators=220,learning_rate=.035,num_leaves=12,max_depth=-1,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=9109)
    mod.fit(tr[cols],tr.sub_temp.to_numpy()-lin.predict(tr[PHYSICS_COLUMNS]),sample_weight=w)
    return lin.predict(va[PHYSICS_COLUMNS])+mod.predict(va[cols])

newfits=0
for s,fds in REG['folds'].items():
    for k,fd in enumerate(fds):
        start=time.monotonic();parent_path=OUT/(s+'_'+str(k)+'_v1.npz')
        parent=json.loads((OUT/(s+'_'+str(k)+'_v1.json')).read_text(encoding='utf-8'))
        assert parent['recipe_sha']==parent_sha and sha(parent_path)==parent['npz_sha']
        z=np.load(parent_path,allow_pickle=True);ids=z['row_id'].tolist();assert ids==parent['query_ids']
        fn=IOUT/(s+'_'+str(k)+'_v1.npz');mn=IOUT/(s+'_'+str(k)+'_v1.json')
        if fn.exists():
            cached=json.loads(mn.read_text(encoding='utf-8'));assert cached['recipe_sha']==recipe_sha and cached['npz_sha']==sha(fn)
            print('N3_REUSE',s,k,flush=True);continue
        tr=lab(parent['train_ids']);va=lab(ids);w=weights(tr)
        masked=loader();masked[0].loc[~masked[0].row_id.isin(tr.row_id),common.USABLE]=np.nan
        protected=features(masked);maskcells=eq(cf.loc[tr.row_id],protected.loc[tr.row_id],FEATURES)
        tr,va,notes=season_frames(tr,va);assert notes==parent['season']
        assert {str(v):int((w==v).sum()) for v in np.unique(w)}==parent['weights']
        probe=None
        if s=='DIAG10' and k==0:
            baseline=fit(tr,va,w,FEATURE_COLUMNS);newfits+=1
            assert np.array_equal(baseline,z['BASELINE']);probe=0.0
        try:
            n3predict(tr,va.assign(sub_temp=987654.0),w)
        except AssertionError:
            negative_query_label=True
        else:raise AssertionError('query labels were not rejected before fitting')
        clean=va.drop(columns=['sub_temp']);pred,heads=n3predict(tr,clean,w);newfits+=2;assert np.isfinite(pred).all()
        np.savez(fn,row_id=np.array(ids,dtype=object),BASELINE=z['BASELINE'],N3=pred,level=heads['level'],shape_raw=heads['shape_raw'],shape=heads['shape'])
        receipt={'recipe_sha':recipe_sha,'parent_npz_sha':sha(parent_path),'npz_sha':sha(fn),'query_ids':ids,'query_label_negative_pass':negative_query_label,'level_train_rows':heads['level_rows'],'shape_train_rows':heads['shape_rows'],'feature_isolation_cells':maskcells,'baseline_exact_probe_max':probe,'weights':parent['weights'],'season':notes,'seconds':time.monotonic()-start}
        mn.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        print('N3_FOLD_DONE',s,k,receipt['seconds'],flush=True)
        if args.first:raise SystemExit(0)

results={};rmse=lambda x,y:float(np.sqrt(np.mean((x-y)**2)))
for s,fds in REG['folds'].items():
    ids=[];a=[];b=[]
    for k in range(len(fds)):
        fn=IOUT/(s+'_'+str(k)+'_v1.npz');m=json.loads((IOUT/(s+'_'+str(k)+'_v1.json')).read_text(encoding='utf-8'));assert sha(fn)==m['npz_sha']
        z=np.load(fn,allow_pickle=True);assert z['row_id'].tolist()==m['query_ids'];ids.extend(z['row_id'].tolist());a.extend(z['BASELINE']);b.extend(z['N3'])
    a=np.asarray(a);b=np.asarray(b);y=labels.loc[ids].to_numpy();ma=rmse(a,y);mb=rmse(b,y)
    metric={'base_rmse':ma,'candidate_rmse':mb,'pct':100*(mb/ma-1),'rows':len(y),'better':mb<ma}
    if s=='DIAG10':
        key=[i[:3]+'_'+str(int(i[4:7])//5) for i in ids];_,inv=np.unique(key,return_inverse=True);n=inv.max()+1
        ea=np.bincount(inv,weights=(a-y)**2);eb=np.bincount(inv,weights=(b-y)**2);sz=np.bincount(inv)
        rng=np.random.default_rng(20261010);ds=[]
        for j in range(20):
            draw=rng.integers(n,size=(1000,n));den=sz[draw].sum(1);ds.extend((np.sqrt(eb[draw].sum(1)/den)-np.sqrt(ea[draw].sum(1)/den)).tolist())
        metric.update(p_worse=float(np.mean(np.array(ds)>=0)),ci=list(map(float,np.quantile(ds,[ALPHA,1-ALPHA]))))
    results[s]=metric
passed=all(m['better'] for m in results.values()) and results['DIAG10']['p_worse']<ALPHA and results['DIAG10']['ci'][1]<0
assert all(sha(CODE.parent/n)==v for n,v in P['temp_package_files'].items())
save('n3_results_v1.json',{'status':'N3_COMPONENT_SCREEN_COMPLETED','results':results,'screen_pass':passed,'full_mixture_adoption':False,'new_fit':newfits,'total_component_fits':25,'feature_audit_cells':checks,'recipe_sha':recipe_sha})
print('N3_RESULTS',json.dumps(results),passed,flush=True)
