"""Fixed two-candidate CODEX component screen, not a full mixture adoption."""
from runtime_v1 import *
import argparse,time
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
parser=argparse.ArgumentParser();parser.add_argument('--first',action='store_true');args=parser.parse_args()
SEED=9109
OUT.mkdir(parents=True,exist_ok=True)
cf=cframes();cm=cframes(smooth=True)
lab,_,ct,phc,_=V.build_frames(loader)
full,_,_=TM.build_world()
dmin=full.groupby(['farm','day']).ph_in_temp_3.min()
sets=[('DIAG10',diag_folds(lab))]
for th in [10,12]:
    ix=dmin[dmin<th].index
    sets.append(('EXT'+str(th),[{f:{int(d) for ff,d in ix if ff==f} for f in common.TARGET_FARMS}]))
frame=cf.loc[lab.row_id].reset_index();frame['sub_temp']=lab.sub_temp.to_numpy()
med=cm.loc[lab.row_id].reset_index();med['sub_temp']=lab.sub_temp.to_numpy()
recipe={'seed':SEED,'candidates':['DROP5','MED3'],'co2_columns':CO2,'features':FEATURE_COLUMNS,'physics':PHYSICS_COLUMNS,'fold_local_weights':True,'fold_local_season':True,'module_sha':sha(HERE/'runtime_v1.py'),'source_sha':sha(__file__),'data_sha':{n:sha(DATA/n) for n in ['train_X.csv','train_y.csv','test_X.csv']},'folds':{s:[{f:sorted(fd[f]) for f in common.TARGET_FARMS} for fd in fds] for s,fds in sets},'alpha':.0125,'bootstrap':20000,'ci_quantiles':[.0125,.9875],'full_mixture_adoption':False}
recipe_sha=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
reg=HERE/'registration_v1.json'
if reg.exists():assert json.loads(reg.read_text(encoding='utf-8'))==recipe
else:save('registration_v1.json',recipe)
print('REGISTERED',recipe_sha,flush=True)

def fit(tr,va,w,kind):
    cols=[c for c in FEATURE_COLUMNS if kind!='DROP5' or c not in CO2]
    lin=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.0))
    lin.fit(tr[PHYSICS_COLUMNS],tr.sub_temp,ridge__sample_weight=w)
    mod=LGBMRegressor(n_estimators=220,learning_rate=.035,num_leaves=12,max_depth=-1,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=SEED)
    mod.fit(tr[cols],tr.sub_temp.to_numpy()-lin.predict(tr[PHYSICS_COLUMNS]),sample_weight=w)
    return lin.predict(va[PHYSICS_COLUMNS])+mod.predict(va[cols])

count=0
for s,fds in sets:
    for k,fd in enumerate(fds):
        fn=OUT/(s+'_'+str(k)+'_v1.npz');mn=OUT/(s+'_'+str(k)+'_v1.json')
        if fn.exists():
            cached=json.loads(mn.read_text(encoding='utf-8'));assert cached['recipe_sha']==recipe_sha and cached['npz_sha']==sha(fn)
            print('REUSE',s,k,flush=True);continue
        start=time.monotonic();trm,vam=common.split_mask(lab,fd)
        assert trm.any() and vam.any() and not np.any(trm&vam)
        # Whole-day holdout means reset CODEX train features cannot depend on validation input.
        masked=loader();masked[0].loc[~masked[0].row_id.isin(lab.loc[trm,'row_id']),common.USABLE]=np.nan
        traincf=cframes(masked);traincm=cframes(masked,True)
        cells=eq(cf.loc[lab.loc[trm,'row_id']],traincf.loc[lab.loc[trm,'row_id']],FEATURE_COLUMNS)
        cells+=eq(cm.loc[lab.loc[trm,'row_id']],traincm.loc[lab.loc[trm,'row_id']],FEATURE_COLUMNS)
        w=weights(lab[trm]);tr,va,notes=season_frames(frame[trm],frame[vam])
        mt,mv,_=season_frames(med[trm],med[vam])
        assert np.array_equal(tr.day,mt.day) and np.array_equal(va.day,mv.day)
        pred={'BASELINE':fit(tr,va,w,'BASELINE'),'DROP5':fit(tr,va,w,'DROP5'),'MED3':fit(mt,mv,w,'MED3')}
        assert all(np.isfinite(p).all() for p in pred.values())
        np.savez(fn,row_id=lab.loc[vam,'row_id'].to_numpy(),**pred)
        record={'npz_sha':sha(fn),'recipe_sha':recipe_sha,'fold':k,'validator':s,'train_ids':lab.loc[trm,'row_id'].tolist(),'query_ids':lab.loc[vam,'row_id'].tolist(),'train_key_count':len(tr[['farm','t']].drop_duplicates()),'feature_isolation_cells':cells,'season':notes,'seconds':time.monotonic()-start,'weights':{str(v):int((w==v).sum()) for v in np.unique(w)}}
        mn.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        count+=1;print('FOLD_DONE',s,k,record['seconds'],len(va),flush=True)
        if args.first:raise SystemExit(0)

results={};OOF={}
y=lab.sub_temp.to_numpy();rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
for s,fds in sets:
    predictions={v:np.full(len(lab),np.nan) for v in ['BASELINE','DROP5','MED3']}
    idx=pd.Index(lab.row_id)
    for k in range(len(fds)):
        zp=OUT/(s+'_'+str(k)+'_v1.npz');zpmeta=json.loads((OUT/(s+'_'+str(k)+'_v1.json')).read_text(encoding='utf-8'));assert sha(zp)==zpmeta['npz_sha'];z=np.load(zp,allow_pickle=True);assert z['row_id'].tolist()==zpmeta['query_ids']
        ii=idx.get_indexer(z['row_id']);assert np.all(ii>=0)
        for name in predictions:
            assert np.isnan(predictions[name][ii]).all()
            predictions[name][ii]=z[name]
    take=np.isfinite(predictions['BASELINE']);a=predictions['BASELINE'][take]
    metrics={}
    for name in ['DROP5','MED3']:
        b=predictions[name][take];yy=y[take]
        ma=rmse(a,yy);mb=rmse(b,yy)
        metrics[name]={'base_rmse':ma,'candidate_rmse':mb,'pct':100*(mb/ma-1),'rows':int(take.sum()),'better':mb<ma}
        if s=='DIAG10':
            qlab=lab[take];key=qlab.farm.astype(str)+'_'+(qlab.day//5).astype(str)
            _,inv=np.unique(key,return_inverse=True);n=inv.max()+1
            ea=np.bincount(inv,weights=(a-yy)**2);eb=np.bincount(inv,weights=(b-yy)**2);sz=np.bincount(inv)
            rng=np.random.default_rng(20261009);ds=[]
            for j in range(20):
                draw=rng.integers(n,size=(1000,n));den=sz[draw].sum(1)
                ds.extend((np.sqrt(eb[draw].sum(1)/den)-np.sqrt(ea[draw].sum(1)/den)).tolist())
            ds=np.asarray(ds);metrics[name].update(p_worse=float(np.mean(ds>=0)),ci=list(map(float,np.quantile(ds,[.0125,.9875]))))
        groups=[]
        for f in common.TARGET_FARMS:
            for late in [False,True]:
                m=(lab.loc[take,'farm'].to_numpy()==f)&((lab.loc[take,'day'].to_numpy()>=179)==late)
                if m.any():groups.append({'farm':f,'late':late,'n':int(m.sum()),'base':rmse(a[m],yy[m]),'candidate':rmse(b[m],yy[m])})
        metrics[name]['groups']=groups
    results[s]=metrics;OOF.update({s+'__'+name:p for name,p in predictions.items()})
np.savez(OUT/'component_oof_v1.npz',row_id=lab.row_id.to_numpy(),**OOF)
passes={name:all(results[s][name]['better'] for s,_ in sets) and results['DIAG10'][name]['p_worse']<.0125 and results['DIAG10'][name]['ci'][1]<0 for name in ['DROP5','MED3']}
assert all(sha(CODE.parent/n)==v for n,v in P['temp_package_files'].items())
save('results_v1.json',{'status':'COMPONENT_SCREEN_COMPLETED','results':results,'screen_pass':passes,'new_fit':count*3,'total_models':sum(len(x) for _,x in sets)*3,'full_mixture_adoption':False,'recipe_sha':recipe_sha})
print('SCREEN',json.dumps(passes),json.dumps(results),flush=True)
