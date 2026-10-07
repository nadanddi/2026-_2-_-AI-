from pathlib import Path
import sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import run_v1 as R
np,pd,B=R.np,R.pd,R.B
raw=pd.read_csv(Path(B.S.env.DATA)/'train_X.csv');raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
assert B.sha(Path(B.S.env.DATA)/'train_X.csv')=='21291a8237fadfca3addd88782f369f6a25159efe1b1508067470c5dfc9a6300'
f=B.M.features(raw[['row_id']+B.M.RAW]);h0=f[f.hour==0];checks=[]
for c in ['in_co2','act_heating']:
    assert np.array_equal(h0[c].to_numpy(),h0[c+'_h0'].to_numpy(),equal_nan=True)
    checks.append(dict(kind='hour0_current_equals_h0',feature=c,rows=len(h0),equal_nan=True))
h1=f[(f.hour==1)&f[['act_heating','act_heating_tdm','act_heating_h0']].notna().all(axis=1)];gap=float(abs(2*h1.act_heating_tdm-h1.act_heating-h1.act_heating_h0).max());assert gap<1e-12
checks.append(dict(kind='hour1_heating_h0_recoverable_from_current_and_prefix_mean',rows=len(h1),maxdiff=gap))
B.write(H/'input_identity_v1.json',dict(status='PASS_INPUT_IDENTITIES_ONLY',checks=checks,new_fit=0,labels_read=False,source_sha=B.sha(__file__),scope='all400day train_X inputs; not a prediction equivalence or EC efficacy test'))
print('INPUT_IDENTITY_PASS',len(h0),len(h1),flush=True)
