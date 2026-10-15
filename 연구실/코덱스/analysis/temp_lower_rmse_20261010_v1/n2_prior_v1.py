"""Fixed official other-farm delta prior; no target-farm labels in prior fit."""
from pathlib import Path
import sys,json
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'연구실/코덱스/analysis/temp_repair_20261009_v1'))
from runtime_v1 import RAW,common,np,pd,sha,DATA,P
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib
REG=json.loads((H/'n2_registration_v1.json').read_text(encoding='utf-8'))
LOFO=json.loads((H/'n2_lofo_results_v1.json').read_text(encoding='utf-8'))
assert LOFO['gate_pass'] and REG['source_sha']==sha(H/'n2_lofo_v3.py') and REG['data_sha']==P['data_sha']
U=common.USABLE;C=U+['delta_'+c for c in U]+['sin','cos'];assert C==REG['features']
O=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1'

def predictor(a):
    a=a.sort_values(['farm','day','hour']).reset_index(drop=True).copy();g=a.groupby(['farm','day'],sort=False)
    for c in U:a['delta_'+c]=g[c].diff()
    a['sin']=np.sin(a.hour*np.pi/12);a['cos']=np.cos(a.hour*np.pi/12)
    valid=(a.hour>0)&(g.hour.diff()==1)&np.isfinite(a.delta_in_temp)
    a[C]=a[C].replace([np.inf,-np.inf],np.nan)
    return a,valid

def fit_prior():
    a=RAW[0][['row_id','farm','day','hour','t']+U].merge(RAW[1][['row_id','sub_temp']],on='row_id',how='left',validate='one_to_one')
    a=a[~a.farm.isin(REG['exclude'])]
    a,valid=predictor(a);a['target']=a.groupby(['farm','day']).sub_temp.diff()
    a=a[valid&np.isfinite(a.target)&a.farm.isin(LOFO['eligible'])].copy();assert len(a)==LOFO['rows']
    n=a.groupby('farm').size();assert {f:int(v) for f,v in n.items()}==LOFO['pairs']
    w=np.array([len(a)/len(n)/n[f] for f in a.farm])
    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.))
    model.fit(a[C],a.target,ridge__sample_weight=w)
    receipt={'source_sha':sha(__file__),'lofo_recipe_sha':sha(H/'n2_registration_v1.json'),'data_sha':REG['data_sha'],'rows':len(a),'farms':sorted(n.index.tolist()),'features':C,'ridge_coef':model.named_steps['ridge'].coef_.tolist(),'intercept':float(model.named_steps['ridge'].intercept_),'imputer_statistics':model.named_steps['simpleimputer'].statistics_.tolist(),'scaler_mean':model.named_steps['standardscaler'].mean_.tolist(),'scaler_scale':model.named_steps['standardscaler'].scale_.tolist(),'target_farm_labels_used':False}
    return model,receipt

MODEL,RECEIPT=fit_prior()

def target_prior(tx,sx):
    a=pd.concat([tx[['row_id','farm','day','hour','t']+U],sx[['row_id','farm','day','hour','t']+U]],ignore_index=True)
    a=a[a.farm.isin(['F13','F47'])];a,valid=predictor(a)
    a['step']=np.nan;a.loc[valid,'step']=MODEL.predict(a.loc[valid,C]);assert np.isfinite(a.loc[valid,'step']).all()
    a.loc[a.hour==0,'step']=0.0
    a['n2_cumulative']=a.groupby(['farm','day']).step.transform(lambda s:s.cumsum(skipna=False))
    return a[['row_id','n2_cumulative']].set_index('row_id')
