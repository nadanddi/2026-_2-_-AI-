"""One frozen irradiance candidate, compatible current CODEX component baseline."""
from runtime_v1 import *
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import argparse,time
ap=argparse.ArgumentParser();ap.add_argument('--first',action='store_true');args=ap.parse_args()
IRR=['rad_eff','rad_eff_reset1','rad_eff_reset3','rad_eff_reset8'];FEATURES=FEATURE_COLUMNS+IRR
ALPHA=.025/3
REG=json.loads((HERE/'registration_v1.json').read_text(encoding='utf-8'))
parent_sha=hashlib.sha256(json.dumps(REG,sort_keys=True).encode()).hexdigest()
IOUT=OUT/'irr4_v1';IOUT.mkdir(parents=True,exist_ok=True)

def extra(cf):
    a=cf.copy();a['rad_eff']=a.out_rad*(a.act_shade.fillna(100)/100)*(a.act_thermal.fillna(100)/100)
    for h in [1,3,8]:a['rad_eff_reset'+str(h)]=a.groupby(['farm','day']).rad_eff.transform(lambda s,h=h:s.ewm(halflife=h).mean())
    return a

def features(raw=None,mask=True):
    tx,_,sx=masks(RAW if raw is None else raw) if mask else loader(raw)
    return extra(build_features(tx,sx).set_index('row_id'))

cf=features();source=features(mask=False);checks=0
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
recipe={'name':'IRR4','features':FEATURES,'physics':PHYSICS_COLUMNS,'seed':9109,'alpha':ALPHA,'parent_recipe_sha':parent_sha,'source_sha':sha(__file__),'runtime_sha':sha(HERE/'runtime_v1.py'),'formula':'out_rad*(act_shade.fillna(100)/100)*(act_thermal.fillna(100)/100), day-reset EWMA1/3/8','data_sha':REG['data_sha'],'baseline_exact_probe':True,'bootstrap':20000,'bootstrap_seed':20261009,'ci_quantiles':[ALPHA,1-ALPHA],'full_mixture_adoption':False}
recipe_sha=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
regp=HERE/'irr_registration_v1.json'
if regp.exists():assert json.loads(regp.read_text(encoding='utf-8'))==recipe
else:save('irr_registration_v1.json',recipe)
print('IRR_AUDIT_REGISTERED',checks,recipe_sha,flush=True)
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
            print('IRR_REUSE',s,k,flush=True);continue
        tr=lab(parent['train_ids']);va=lab(ids);w=weights(tr)
        masked=loader();masked[0].loc[~masked[0].row_id.isin(tr.row_id),common.USABLE]=np.nan
        protected=features(masked);maskcells=eq(cf.loc[tr.row_id],protected.loc[tr.row_id],FEATURES)
        tr,va,notes=season_frames(tr,va);assert notes==parent['season']
        assert {str(v):int((w==v).sum()) for v in np.unique(w)}==parent['weights']
        probe=None
        if s=='DIAG10' and k==0:
            baseline=fit(tr,va,w,FEATURE_COLUMNS);newfits+=1
            assert np.array_equal(baseline,z['BASELINE']);probe=0.0
        pred=fit(tr,va,w,FEATURES);newfits+=1;assert np.isfinite(pred).all()
        np.savez(fn,row_id=np.array(ids,dtype=object),BASELINE=z['BASELINE'],IRR4=pred)
        receipt={'recipe_sha':recipe_sha,'parent_npz_sha':sha(parent_path),'npz_sha':sha(fn),'query_ids':ids,'feature_isolation_cells':maskcells,'baseline_exact_probe_max':probe,'weights':parent['weights'],'season':notes,'seconds':time.monotonic()-start}
        mn.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        print('IRR_FOLD_DONE',s,k,receipt['seconds'],flush=True)
        if args.first:raise SystemExit(0)

results={};rmse=lambda x,y:float(np.sqrt(np.mean((x-y)**2)))
for s,fds in REG['folds'].items():
    ids=[];a=[];b=[]
    for k in range(len(fds)):
        fn=IOUT/(s+'_'+str(k)+'_v1.npz');m=json.loads((IOUT/(s+'_'+str(k)+'_v1.json')).read_text(encoding='utf-8'));assert sha(fn)==m['npz_sha']
        z=np.load(fn,allow_pickle=True);assert z['row_id'].tolist()==m['query_ids'];ids.extend(z['row_id'].tolist());a.extend(z['BASELINE']);b.extend(z['IRR4'])
    a=np.asarray(a);b=np.asarray(b);y=labels.loc[ids].to_numpy();ma=rmse(a,y);mb=rmse(b,y)
    metric={'base_rmse':ma,'candidate_rmse':mb,'pct':100*(mb/ma-1),'rows':len(y),'better':mb<ma}
    if s=='DIAG10':
        key=[i[:3]+'_'+str(int(i[4:7])//5) for i in ids];_,inv=np.unique(key,return_inverse=True);n=inv.max()+1
        ea=np.bincount(inv,weights=(a-y)**2);eb=np.bincount(inv,weights=(b-y)**2);sz=np.bincount(inv)
        rng=np.random.default_rng(20261009);ds=[]
        for j in range(20):
            draw=rng.integers(n,size=(1000,n));den=sz[draw].sum(1);ds.extend((np.sqrt(eb[draw].sum(1)/den)-np.sqrt(ea[draw].sum(1)/den)).tolist())
        metric.update(p_worse=float(np.mean(np.array(ds)>=0)),ci=list(map(float,np.quantile(ds,[ALPHA,1-ALPHA]))))
    results[s]=metric
passed=all(m['better'] for m in results.values()) and results['DIAG10']['p_worse']<ALPHA and results['DIAG10']['ci'][1]<0
assert all(sha(CODE.parent/n)==v for n,v in P['temp_package_files'].items())
save('irr_results_v1.json',{'status':'IRR4_COMPONENT_SCREEN_COMPLETED','results':results,'screen_pass':passed,'full_mixture_adoption':False,'new_fit':newfits,'total_component_fits':13,'feature_audit_cells':checks,'recipe_sha':recipe_sha})
print('IRR_RESULTS',json.dumps(results),passed,flush=True)
