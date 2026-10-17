"""Paired whole-mixture pilot, official-only, immutable round9 source imports."""
from pathlib import Path
import sys, os, json, hashlib, argparse, time
TASK=Path(__file__).resolve().parent
BASE_HOME=TASK.parent/'temp_repair_20261009_v1'
sys.path.insert(0,str(BASE_HOME))
from runtime_v1 import *
import warnings
warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
import feat_new, features_v4 as F4
from make_submission_v3 import m_lgbh,m_ridge,m_nys
from make_submission_v7 import fit_mean_w
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

AP=argparse.ArgumentParser()
AP.add_argument('--audit',action='store_true')
AP.add_argument('--first',action='store_true')
args=AP.parse_args()
HERE=TASK
O=ROOT/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n4_v1'
O.mkdir(parents=True,exist_ok=True)
IRR=['rad_eff','rad_eff_reset1','rad_eff_reset3','rad_eff_reset8']
ALPHA=.025/6

def save(name,obj):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def raw_fold(ids,raw=None):
    tx,ty,sx=loader(raw)
    held=tx.farm.isin(common.TARGET_FARMS)&~tx.row_id.isin(ids)
    sx=pd.concat([sx,tx[held]],ignore_index=True)
    tx=tx[~held].copy()
    ty=ty[ty.row_id.isin(ids)].copy()
    assert not set(tx.row_id)&set(sx.row_id)
    return tx,ty,sx

def world(ids,raw=None,mask=False):
    tx,ty,sx=raw_fold(ids,raw)
    if mask:sx[common.USABLE]=np.nan
    common.load_raw=lambda:(tx.copy(),ty.copy(),sx.copy())
    harness._CACHE.clear()
    try:
        tr,ct,phc=TM.build_world()
        panel,_,_=harness.load()
        ex,_=feat_new.build_extra()
        sg,ph,fp=F4.seg_features(),F4.phys_features(),F4.fp_features()
        query=(panel[panel.is_test].reset_index(drop=True).merge(ex,on='row_id',how='left')
               .merge(sg,on='row_id',how='left').merge(ph,on='row_id',how='left').merge(fp,on='row_id',how='left'))
        cf=build_features(tx,sx).set_index('row_id')
        cf['rad_eff']=cf.out_rad*(cf.act_shade.fillna(100)/100)*(cf.act_thermal.fillna(100)/100)
        for h in [1,3,8]:
            cf['rad_eff_reset'+str(h)]=cf.groupby(['farm','day']).rad_eff.transform(lambda s,h=h:s.ewm(halflife=h,adjust=True,ignore_na=False).mean())
        for frame in [tr,query]:
            for c in list(dict.fromkeys(FEATURE_COLUMNS+PHYSICS_COLUMNS+IRR)):
                frame['cx__'+c]=cf.loc[frame.row_id,c].to_numpy()
        query=query.drop(columns=['sub_temp','sub_ec'])
        return tr.set_index('row_id'),query.set_index('row_id'),ct,phc
    finally:
        common.load_raw=loader;harness._CACHE.clear()

def cx(frame):
    a=pd.DataFrame({c[4:]:frame[c].to_numpy() for c in frame if c.startswith('cx__')})
    for c in ['farm','day','hour','t']:a[c]=frame[c].to_numpy()
    a['row_id']=frame.index.to_numpy()
    if 'sub_temp' in frame:a['sub_temp']=frame.sub_temp.to_numpy()
    return a

def folds():
    lab=cframes().loc[RAW[1].loc[RAW[1].sub_temp.notna()&RAW[1].farm.isin(common.TARGET_FARMS),'row_id']].reset_index()
    result=[{f:[] for f in common.TARGET_FARMS} for _ in range(10)]
    for f in common.TARGET_FARMS:
        for i,d in enumerate(sorted(lab.loc[lab.farm==f,'day'].unique())):
            result[((i+2)//5+(3 if f=='F47' else 0))%10][f].append(int(d))
    allids=lab.row_id.tolist()
    _,full,ct,phc=world(allids)
    dmin=full.ph_in_temp_3.groupby([full.farm,full.day]).min()
    # All training target rows are absent from query here; use training world.
    fulltr,_,_,_=world(allids)
    dmin=fulltr.ph_in_temp_3.groupby([fulltr.farm,fulltr.day]).min()
    sets={'DIAG10':result}
    for threshold in [10,12]:
        cold=dmin[dmin<threshold]
        sets['EXT'+str(threshold)]=[{f:[int(d) for ff,d in cold.index if ff==f] for f in common.TARGET_FARMS}]
    out=[]
    for name,fds in sets.items():
        for k,fd in enumerate(fds):
            tm,vm=common.split_mask(lab,fd)
            out.append({'set':name,'fold':k,'train_ids':lab.loc[tm,'row_id'].tolist(),'query_ids':lab.loc[vm,'row_id'].tolist(),'days':fd})
    return out,ct,phc

for n,v in P['data_sha'].items():assert sha(DATA/n)==v
for n,v in P['temp_package_files'].items():assert sha(CODE.parent/n)==v
assert sha(Path(P['cache'])/'tabpfn-v2-regressor.ckpt')==P['checkpoint_sha']
FD,CT,PH=folds()
recipe={'candidate':'N4_WARM_IRR4','alpha':ALPHA,'folds':FD,'base_features':CT,'base_physics':PH,'codex':FEATURE_COLUMNS,'cx_physics':PHYSICS_COLUMNS,'irr4':IRR,'base_seeds':[7,101,2024],'codex_seed':726,'irr_seed':9109,'seed_noop_each_fold':True,'pfn_context_model_seed':49,'pfn_context':2000,'pfn_estimators':4,'pfn_samples':1,'pfn_device':'cpu','bootstrap_seed':20261010,'bootstrap_n':20000,'candidate_formula':'c_new=c+g*(c_irr-c); W40G-S combination otherwise fixed','source_sha':sha(__file__),'runtime_sha':sha(BASE_HOME/'runtime_v1.py'),'data_sha':P['data_sha'],'checkpoint_sha':P['checkpoint_sha'],'adoption':False,'posthoc_hypothesis':True}
RS=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
rp=HERE/'n4_registration_v2.json'
if rp.exists():assert json.loads(rp.read_text(encoding='utf-8'))==recipe
else:save(rp.name,recipe)

def audit():
    fd=FD[0];ids=fd['train_ids'];qids=fd['query_ids']
    train,_,ct,ph=world(ids,mask=True)
    _,query,_,_=world(ids)
    cols=list(dict.fromkeys(ct+ph+['cx__'+c for c in FEATURE_COLUMNS+PHYSICS_COLUMNS+IRR]))
    cells=0
    assert 'sub_temp' not in query and 'sub_ec' not in query
    raw=loader();raw[1].loc[~raw[1].row_id.isin(ids),'sub_temp']=987654
    a,b,_,_=world(ids,raw,mask=True);cells+=eq(train,a.loc[train.index],cols)
    _,b,_,_=world(ids,raw);cells+=eq(query.loc[qids],b.loc[qids],cols)
    raw=loader();held=raw[0].farm.isin(common.TARGET_FARMS)&~raw[0].row_id.isin(ids)
    raw[0].loc[held,common.USABLE]=123456;raw[2][common.USABLE]=123456
    a,_,_,_=world(ids,raw,mask=True);cells+=eq(train,a.loc[train.index],cols)
    for f in common.TARGET_FARMS:
        q=query.loc[qids];d=sorted(q.loc[q.farm==f,'day'].unique())[0];cut=int(d)*24+8
        raw=loader()
        for table in [raw[0],raw[2]]:table.loc[(table.farm==f)&(table.t>cut),common.USABLE]=2222
        _,b,_,_=world(ids,raw)
        qi=q.index[(q.farm==f)&(q.t<=cut)];cells+=eq(q.loc[qi],b.loc[qi],cols)
        raw=loader()
        for table in [raw[0],raw[2]]:table.loc[table.farm!=f,common.USABLE]=-3333
        _,b,_,_=world(ids,raw)
        qi=q.index[q.farm==f];cells+=eq(q.loc[qi],b.loc[qi],cols)
    g=V.gate(np.array([7.,8.,9.,10.,np.nan]));assert np.array_equal(g,[0.,0.,.5,1.,1.])
    c=np.arange(5.);irr=c+2.;base=c-1.;pfn=c+1.
    a=V.combine(base,c,pfn,g);b=V.combine(base,c+g*(irr-c),pfn,g)
    assert np.array_equal(a[g==0],b[g==0]);assert np.allclose(b-a,(.6-.4*g)*g*(irr-c),rtol=0,atol=1e-14)
    return {'recipe_sha':RS,'cells':cells,'future_otherfarm_mask_label_pass':True,'gate_formula_pass':True,'query_rows':len(qids),'train_rows':len(ids),'new_fit':0}

if args.audit:
    report=audit();save('n4_audit_v2.json',report);print('N4_AUDIT',json.dumps(report),flush=True);raise SystemExit(0)

assert (HERE/'N4_PLAN_REVIEW_v1.md').exists(),'independent preflight review required'
assert json.loads((HERE/'n4_audit_v2.json').read_text(encoding='utf-8'))['recipe_sha']==RS
for n in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TABPFN_DISABLE_TELEMETRY','HF_HUB_DISABLE_TELEMETRY']:os.environ[n]='1'
os.environ['TABPFN_MODEL_CACHE_DIR']=P['cache']
import torch
torch.set_num_threads(6)
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import get_cache_dir
assert get_cache_dir().resolve()==Path(P['cache']).resolve()
labels=RAW[1].set_index('row_id').sub_temp

for fd in (FD[:1] if args.first else FD):
    name=fd['set']+'_'+str(fd['fold']);fn=O/(name+'.npz');mn=O/(name+'.json')
    if fn.exists():
        receipt=json.loads(mn.read_text(encoding='utf-8'));assert receipt['recipe_sha']==RS and receipt['npz_sha']==sha(fn)
        print('N4_REUSE',name,flush=True);continue
    start=time.monotonic();a,_,ct,ph=world(fd['train_ids'],mask=True);_,b,_,_=world(fd['train_ids'])
    tr=a.loc[fd['train_ids']];va=b.loc[fd['query_ids']];assert 'sub_temp' not in va
    ctr,cte=cx(tr),cx(va);w=weights(ctr);ctr,cte,notes=season_frames(ctr,cte)
    imp=SimpleImputer(strategy='median').fit(tr[ph])
    lin=LinearRegression().fit(imp.transform(tr[ph]),tr.sub_temp.to_numpy(),sample_weight=w)
    bt=lin.predict(imp.transform(tr[ph]));bv=lin.predict(imp.transform(va[ph]))
    base=(.65*(bv+fit_mean_w(m_lgbh,tr[ct],tr.sub_temp.to_numpy()-bt,va[ct],w))
          +.25*fit_mean_w(m_ridge,tr[ct],tr.sub_temp.to_numpy(),va[ct],w,'ridge')
          +.10*fit_mean_w(m_nys,tr[ct],tr.sub_temp.to_numpy(),va[ct],w,'ridge'))
    c=TM.codex_fit_predict(ctr,cte,w,726);c_probe=TM.codex_fit_predict(ctr,cte,w,9109)
    assert np.array_equal(c,c_probe),'seed confound'
    from lightgbm import LGBMRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    cl=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.))
    cl.fit(ctr[PHYSICS_COLUMNS],ctr.sub_temp.to_numpy(),ridge__sample_weight=w)
    cm=LGBMRegressor(n_estimators=220,learning_rate=.035,num_leaves=12,max_depth=-1,min_child_samples=100,reg_lambda=15,verbosity=-1,n_jobs=4,random_state=9109)
    cm.fit(ctr[FEATURE_COLUMNS+IRR],ctr.sub_temp.to_numpy()-cl.predict(ctr[PHYSICS_COLUMNS]),sample_weight=w)
    irr=cl.predict(cte[PHYSICS_COLUMNS])+cm.predict(cte[FEATURE_COLUMNS+IRR])
    rng=np.random.default_rng(49);idx=rng.choice(len(ctr),size=min(2000,len(ctr)),replace=False,p=w/w.sum())
    model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,device='cpu',n_estimators=4,random_state=49,ignore_pretraining_limits=True)
    model.fit(ctr[FEATURE_COLUMNS].to_numpy(np.float32)[idx],ctr.sub_temp.to_numpy()[idx])
    print('N4_PFN_FIT',name,time.monotonic()-start,flush=True)
    pfn=np.asarray(model.predict(cte[FEATURE_COLUMNS].to_numpy(np.float32)),float)
    gate=V.gate(va.in_temp.to_numpy());baseline=V.combine(base,c,pfn,gate)
    candidate=V.combine(base,c+gate*(irr-c),pfn,gate)
    assert np.array_equal(baseline[gate==0],candidate[gate==0])
    assert np.allclose(candidate-baseline,(.6-.4*gate)*gate*(irr-c),rtol=0,atol=1e-13)
    assert np.isfinite(candidate).all()
    np.savez(fn,row_id=np.array(fd['query_ids'],dtype=object),BASELINE=baseline,N4=candidate,BASE=base,CODEX=c,IRR=irr,PFN=pfn,gate=gate,context_ids=ctr.row_id.to_numpy()[idx])
    receipt={'recipe_sha':RS,'npz_sha':sha(fn),'query_ids':fd['query_ids'],'train_ids':fd['train_ids'],'season':notes,'weights':{str(v):int((w==v).sum()) for v in np.unique(w)},'seed_noop_max':float(np.max(np.abs(c-c_probe))),'seconds':time.monotonic()-start}
    with mn.open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print('N4_DONE',name,receipt['seconds'],float(np.sqrt(np.mean((baseline-labels.loc[fd['query_ids']].to_numpy())**2))),float(np.sqrt(np.mean((candidate-labels.loc[fd['query_ids']].to_numpy())**2))),flush=True)
    if args.first:raise SystemExit(0)

if args.first:raise SystemExit(0)
result={}
for s in ['DIAG10','EXT10','EXT12']:
    ids=[];a=[];b=[]
    for fd in [f for f in FD if f['set']==s]:
        fn=O/(s+'_'+str(fd['fold'])+'.npz');mn=fn.with_suffix('.json');receipt=json.loads(mn.read_text(encoding='utf-8'));assert receipt['recipe_sha']==RS and receipt['npz_sha']==sha(fn)
        z=np.load(fn,allow_pickle=True);assert z['row_id'].tolist()==fd['query_ids']
        ids.extend(z['row_id'].tolist());a.extend(z['BASELINE']);b.extend(z['N4'])
    a=np.array(a);b=np.array(b);y=labels.loc[ids].to_numpy();ra=float(np.sqrt(np.mean((a-y)**2)));rb=float(np.sqrt(np.mean((b-y)**2)))
    metric={'base_rmse':ra,'candidate_rmse':rb,'pct':100*(rb/ra-1),'better':rb<ra,'rows':len(y)}
    if s=='DIAG10':
        _,inv=np.unique([i[:3]+'_'+str(int(i[4:7])//5) for i in ids],return_inverse=True)
        n=inv.max()+1;sa=np.bincount(inv,weights=(a-y)**2);sb=np.bincount(inv,weights=(b-y)**2);sz=np.bincount(inv)
        rng=np.random.default_rng(20261010);ds=[]
        for j in range(20):
            draw=rng.integers(n,size=(1000,n));den=sz[draw].sum(1);ds.extend((np.sqrt(sb[draw].sum(1)/den)-np.sqrt(sa[draw].sum(1)/den)).tolist())
        metric.update(p_worse=float(np.mean(np.array(ds)>=0)),ci=list(map(float,np.quantile(ds,[ALPHA,1-ALPHA]))))
    result[s]=metric
passed=all(m['better'] for m in result.values()) and result['DIAG10']['p_worse']<ALPHA and result['DIAG10']['ci'][1]<0
save('n4_results_v2.json',{'results':result,'pilot_pass':passed,'adoption':False,'pfn_samples':1,'recipe_sha':RS})
print('N4_RESULTS',json.dumps(result),passed,flush=True)
