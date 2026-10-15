"""Official other-farm change prior: frozen leave-one-farm-out feasibility only."""
from pathlib import Path
import sys,json,time,hashlib,argparse
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'연구실/코덱스/analysis/temp_repair_20261009_v1'))
from runtime_v1 import RAW,common,np,pd,sha,P,DATA
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
O=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1';O.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--first',action='store_true');args=parser.parse_args()
U=common.USABLE;C=U+['delta_'+c for c in U]+['sin','cos']
assert len(C)==30
recipe={'name':'N2_LOFO','exclude':['F13','F47','F32'],'min_pairs_per_farm':96,'min_farms':10,'min_rows':1000,'fraction_better_both':.8,'ridge_alpha':100,'features':C,'empty_features':True,'farm_equal_weight_mean_one':True,'baseline':['zero','.5*delta_in_temp'],'ties':'fail','runtime_sha':sha(R/'연구실/코덱스/analysis/temp_repair_20261009_v1/runtime_v1.py'),'source_sha':sha(__file__),'data_sha':{n:sha(DATA/n) for n in ['train_X.csv','train_y.csv','test_X.csv']},'weight_scope':'Ridge loss only; imputer/scaler row-weighted','nonfinite_inputs':'inf->NaN fold median, finite target and air-delta required','full_target_adoption':False}
assert recipe['data_sha']==P['data_sha']
regpath=H/'n2_registration_v1.json'
if regpath.exists():assert json.loads(regpath.read_text(encoding='utf-8'))==recipe
else:
 with regpath.open('x',encoding='utf-8') as f:json.dump(recipe,f,ensure_ascii=False,indent=2,allow_nan=False)
a=RAW[0][['row_id','farm','day','hour','t']+U].merge(RAW[1][['row_id','sub_temp']],on='row_id',how='left',validate='one_to_one')
a=a[~a.farm.isin(recipe['exclude'])].sort_values(['farm','day','hour']).reset_index(drop=True)
g=a.groupby(['farm','day'],sort=False)
for c in U:a['delta_'+c]=g[c].diff()
a['target']=g.sub_temp.diff();a['sin']=np.sin(a.hour*np.pi/12);a['cos']=np.cos(a.hour*np.pi/12)
valid=(a.hour>0)&(g.hour.diff()==1)&np.isfinite(a.target)&np.isfinite(a.delta_in_temp)
a=a[valid].copy();a[C]=a[C].replace([np.inf,-np.inf],np.nan);counts=a.groupby('farm').size();eligible=counts[counts>=96].index.tolist();a=a[a.farm.isin(eligible)].reset_index(drop=True)
receipt={'eligible':eligible,'pairs':{f:int(counts[f]) for f in eligible},'rows':len(a),'features':C,'source_sha':recipe['source_sha'],'eligible_count':len(eligible),'ec_numeric_labels_read':False,'full_target_fit':0}
print('SUPPORT',len(eligible),len(a),receipt['pairs'],flush=True)
res=[];pred=np.full(len(a),np.nan)
if len(eligible)>=10 and len(a)>=1000:
 for farm in eligible:
    tr=a.farm!=farm;va=~tr;train=a[tr];query=a[va]
    n=train.groupby('farm').size();w=np.array([len(train)/len(n)/n[f] for f in train.farm])
    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.))
    cache=O/('n2_fold_'+farm+'_v1.npz');metafile=O/('n2_fold_'+farm+'_v1.json')
    if cache.exists():
      metadata=json.loads(metafile.read_text(encoding='utf-8'));assert metadata['source_sha']==recipe['source_sha'] and metadata['npz_sha']==sha(cache)
      z=np.load(cache,allow_pickle=True);assert z['row_id'].tolist()==query.row_id.tolist();p=z['prior']
    else:
      model.fit(train[C],train.target,ridge__sample_weight=w)
      p=model.predict(query[C]);assert np.isfinite(p).all()
      np.savez(cache,row_id=query.row_id.to_numpy(),prior=p)
      metadata={'source_sha':recipe['source_sha'],'npz_sha':sha(cache),'train_farms':sorted(train.farm.unique().tolist()),'query_farm':farm,'train_rows':len(train),'query_rows':len(query),'ridge_coef':model.named_steps['ridge'].coef_.tolist(),'intercept':float(model.named_steps['ridge'].intercept_)}
      metafile.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    assert np.isfinite(p).all();pred[va]=p;y=query.target.to_numpy();b=.5*query.delta_in_temp.to_numpy()
    mse=lambda e:float(np.mean(e**2))
    ms=mse(p-y);mz=mse(y);ma=mse(b-y);assert np.isfinite([ms,mz,ma]).all()
    row={'farm':farm,'train_farms':sorted(train.farm.unique().tolist()),'train_rows':len(train),'n':len(query),'mse_prior':ms,'mse_zero':mz,'mse_half_air':ma,'better_both':ms<mz and ms<ma}
    res.append(row);print('LOFO',farm,row,flush=True)
    if args.first:raise SystemExit(0)
 receipt['macro_mse_prior']=float(np.mean([r['mse_prior'] for r in res]));receipt['macro_mse_zero']=float(np.mean([r['mse_zero'] for r in res]));receipt['macro_mse_half_air']=float(np.mean([r['mse_half_air'] for r in res]))
 receipt['fraction_better_both']=float(np.mean([r['better_both'] for r in res]))
 receipt['gate_pass']=receipt['fraction_better_both']>=.8 and receipt['macro_mse_prior']<receipt['macro_mse_zero'] and receipt['macro_mse_prior']<receipt['macro_mse_half_air']
else:receipt['gate_pass']=False;receipt['reason']='INSUFFICIENT_SUPPORT'
np.savez(O/'n2_lofo_v1.npz',row_id=a.row_id.to_numpy(),farm=a.farm.to_numpy(),target=a.target.to_numpy(),half_air=.5*a.delta_in_temp.to_numpy(),prior=pred)
receipt.update(status='LOFO_FEASIBILITY_COMPLETED',results=res,models=len(res),npz_sha=sha(O/'n2_lofo_v1.npz'))
with (H/'n2_lofo_results_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False)
print('GATE',receipt['gate_pass'],receipt.get('fraction_better_both'),flush=True)
