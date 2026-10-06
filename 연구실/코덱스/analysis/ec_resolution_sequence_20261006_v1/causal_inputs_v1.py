from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));import stage1_v2 as S
import pandas as pd
raw=pd.read_csv(Path(S.env.DATA)/'train_X.csv',usecols=['row_id']+S.M.RAW)
raw=raw[raw.row_id.str[:3].isin(['F13','F47'])].reset_index(drop=True)
base=S.M.features(raw).set_index('row_id'); cols=[c for c in S.M.FULL if c!='season']
pd.testing.assert_frame_equal(base.sort_index(),S.M.features(raw.sample(frac=1,random_state=817)).set_index('row_id').sort_index())
checks=[]
meta=S.M.identify(raw)
for farm in ['F13','F47']:
    times=meta.day*24+meta.hour
    for fraction in [.2,.5,.8]:
        cut=int(times[meta.farm.eq(farm)].quantile(fraction))
        allowed=meta.farm.eq(farm)&times.le(cut)
        changed=raw.copy(); changed.loc[~allowed,S.M.RAW]=changed.loc[~allowed,S.M.RAW]*13+97
        test=S.M.features(changed).set_index('row_id'); ids=meta.loc[allowed,'row_id']
        pd.testing.assert_frame_equal(base.loc[ids,cols],test.loc[ids,cols])
        checks.append(dict(farm=farm,cut=cut,rows=len(ids),future_other_farm_unchanged=True))
result=dict(status='PASS_CAUSAL_NON_SEASON_INPUT_FEATURES',checks=checks,order_invariant=True,fit=0,
            season_checked=False,model_predictions_checked=False,
            scope='Non-season FULL columns only; season and PFN prediction causality are separate claims',
            source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
dest=H/'causal_inputs_v1.json'; assert not dest.exists()
dest.write_text(json.dumps(result,indent=2),encoding='utf-8');print(result['status'],len(checks),flush=True)
