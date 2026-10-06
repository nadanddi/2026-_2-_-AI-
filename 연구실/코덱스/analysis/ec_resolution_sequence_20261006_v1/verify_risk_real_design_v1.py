"""Real 24-file helper boundary audit. No feature/candidate/model selection."""
from pathlib import Path
import sys,json,hashlib,datetime
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];sys.path.insert(0,str(H));import verify_stage1_v1 as B
pd,np,M=B.pd,B.np,B.M
from risk_core_v2 import design,FORBIDDEN,MEMBERS
O=ROOT/'연구실/코덱스/local'/H.name/'risk_inputs_v1'
expected=list(M.FULL)+['farm_F47','A']+MEMBERS+['prefix_A','et_minus_lgb','mlp_minus_lgb','pfn_minus_lgb','member_range','A_minus_lgb'];assert len(expected)==len(set(expected))==50
files=[]
for path in sorted(O.glob('*.csv')):
    f=pd.read_csv(path,float_precision='round_trip');x=design(f,list(M.FULL));assert x.columns.tolist()==expected;assert not set(FORBIDDEN)&set(x.columns)
    np.testing.assert_allclose(x.prefix_A.to_numpy(),f.prefix_A.to_numpy(),atol=1e-12,rtol=1e-12)
    changed=f.copy();changed['sub_ec']=99.;changed['inner_j']=np.arange(len(changed))+1000;changed['row_id']=['masked_metadata']*len(changed)
    for label in ['y','y_day','high','hard_high','missed_high','hard_low']:changed[label]=np.arange(len(changed))+9999
    pd.testing.assert_frame_equal(x,design(changed,list(M.FULL)))
    shuffled=f.sample(frac=1,random_state=61006);pd.testing.assert_frame_equal(x,design(shuffled,list(M.FULL)).loc[f.index])
    cuts=[]
    for farm in ['F13','F47']:
        times=f.day*24+f.hour;cut=int(times[f.farm.eq(farm)].quantile(.5));allowed=f.farm.eq(farm)&times.le(cut);assert allowed.any()
        future=f.copy();changedmask=f.farm.eq(farm)&times.gt(cut);cols=list(M.FULL)+['A']+MEMBERS;future.loc[changedmask,cols]=future.loc[changedmask,cols]*13+97
        pd.testing.assert_frame_equal(x.loc[allowed],design(future,list(M.FULL)).loc[allowed])
        other=f.copy();mask=~f.farm.eq(farm);other.loc[mask,cols]=other.loc[mask,cols]*17+89
        pd.testing.assert_frame_equal(x.loc[allowed],design(other,list(M.FULL)).loc[allowed]);cuts.append(dict(farm=farm,cut=cut,allowed_rows=int(allowed.sum()),future_otherfarm_unchanged=True))
    files.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(f),columns=len(x.columns),label_metadata_order='PASS',prefix_same='PASS',cuts=cuts))
assert len(files)==24
out=dict(status='PASS_REAL_DESIGN_BOUNDARY_ONLY',files=len(files),base_columns=list(M.FULL),base_count=len(M.FULL),design_columns=expected,design_count=len(expected),extra_labels_added_to_design=False,known_labels_metadata_excluded=sorted(FORBIDDEN),helper_sha256=hashlib.sha256((H/'risk_core_v2.py').read_bytes()).hexdigest(),checks=files,risk_fit=0,candidate_selection=False,scope='helper accepts supplied frozen FULL and model scores; this audit does not regenerate season or model predictions under input perturbation')
target=H/('verify_risk_real_design_v1_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
with target.open('x',encoding='utf-8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=out['status'],files=24,base_columns=len(M.FULL),design_columns=len(expected),labels_added=False,output=str(target)),ensure_ascii=False,indent=2))
