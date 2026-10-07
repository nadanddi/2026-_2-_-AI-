"""Input-only h0 identity and median scope; no labels/predictions/training."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import run_v1 as R
np,pd,B=R.np,R.pd,R.B
raw=pd.read_csv(Path(B.S.env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True);features=B.M.features(raw[['row_id']+B.M.RAW]);zero=features[features.hour.eq(0)].copy()
identity=[]
for c in ['in_co2','act_heating']:
    x=zero[c].to_numpy();y=zero[c+'_h0'].to_numpy();observed=np.isfinite(x)&np.isfinite(y);both=np.isnan(x)&np.isnan(y);assert np.array_equal(x,y,equal_nan=True)
    identity.append(dict(feature=c,days=len(zero),observed_equal=int(observed.sum()),both_missing=int(both.sum()),one_missing=int((np.isnan(x)^np.isnan(y)).sum())))
prior=json.loads((R.OLD/'preparation_v4.json').read_text(encoding='utf-8'));f=features.set_index('row_id');medians=[]
for rec in prior['records']:
    t=f.loc[rec['train_ids']];q=f.loc[rec['query_ids']];z=q[q.hour.eq(0)]
    values={}
    for c in ['in_co2','act_heating']:
        a=float(np.nanmedian(t[c]));b=float(np.nanmedian(t[c+'_h0']));values[c]=dict(current_median=a,h0_median=b,difference=a-b,query_h0_both_missing=int((z[c].isna()&z[c+'_h0'].isna()).sum()))
    medians.append(dict(fold=rec['k'],features=values))
selected=f.loc['F47_161_00'];assert selected['in_co2']==480 and selected['act_heating']==0
out=dict(status='PASS_INPUT_IDENTITY_AND_IMPUTATION_SCOPE',new_fit=0,labels_read=False,identity=identity,fold_medians=medians,selected161_hour0=dict(in_co2=480,act_heating=0,both_observed=True),scope='Raw400day input identities only. Equal raw NaNs need not yield equal imputed current/h0 values; no failure causality claim.')
with (H/'critic_input_imputation_scope_v1.json').open('x',encoding='utf-8') as g:json.dump(out,g,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False),flush=True)
