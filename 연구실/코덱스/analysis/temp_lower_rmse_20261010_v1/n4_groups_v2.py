"""Post-prediction descriptions only; no model selection or retraining."""
from pathlib import Path
import sys,json,hashlib,math
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'.analysis-tools/python'))
import numpy as np,pandas as pd
reg=json.loads((H/'n4_registration_v2.json').read_text(encoding='utf-8'))
prep=json.loads((H.parent/'temp_round9_reproduction_20261008_v1/preparation_v1.json').read_text(encoding='utf-8'))
data=Path(prep['temp_code']).parent/'data'
y=pd.read_csv(data/'train_y.csv',usecols=['row_id','sub_temp']).set_index('row_id').sub_temp
air=pd.read_csv(data/'train_X.csv',usecols=['row_id','in_temp']).set_index('row_id').in_temp
O=R/'연구실/코덱스/local/temp_lower_rmse_20261010_v1/n4_v1'
def metric(df):
    n=len(df);a=math.sqrt(math.fsum(float(v)**2 for v in df.ea)/n);b=math.sqrt(math.fsum(float(v)**2 for v in df.eb)/n)
    return {'n':n,'base_rmse':a,'candidate_rmse':b,'pct':100*(b/a-1),'delta_mse':math.fsum(float(v)**2-float(w)**2 for v,w in zip(df.eb,df.ea))/n}
results={}
for s in ['DIAG10','EXT10','EXT12']:
    frames=[]
    for fd in [f for f in reg['folds'] if f['set']==s]:
        fn=O/(s+'_'+str(fd['fold'])+'.npz');receipt=json.loads(fn.with_suffix('.json').read_text(encoding='utf-8'))
        assert receipt['npz_sha']==hashlib.sha256(fn.read_bytes()).hexdigest()
        z=np.load(fn,allow_pickle=True);ids=z['row_id'].tolist();actual=y.loc[ids].to_numpy()
        df=pd.DataFrame({'row_id':ids,'ea':z['BASELINE']-actual,'eb':z['N4']-actual,'g':z['gate'],'component_delta':z['IRR']-z['CODEX'],'mix_delta':z['N4']-z['BASELINE']})
        df['farm']=df.row_id.str[:3];df['day']=df.row_id.str[4:7].astype(int)
        df['segment']=np.where(df.day<=178,'first','second')
        df['gate_band']=np.where(air.loc[ids].isna().to_numpy(),'in_temp_NaN',np.where(df.g==0,'in_temp_le8',np.where(df.g==1,'in_temp_ge10','in_temp_8to10')))
        frames.append(df)
    allrows=pd.concat(frames,ignore_index=True)
    groups={str(k):metric(v) for k,v in allrows.groupby(['farm','segment'])}
    gates={str(k):metric(v) for k,v in allrows.groupby('gate_band')}
    covariance=math.fsum(float(e)*float(d) for e,d in zip(allrows.ea,allrows.mix_delta))/len(allrows)
    delta2=math.fsum(float(d)**2 for d in allrows.mix_delta)/len(allrows)
    overall=metric(allrows);assert abs(overall['delta_mse']-(2*covariance+delta2))<1e-12
    results[s]={'overall':overall,'farm_segment':groups,'gate':gates,'error_delta_cross_2E':2*covariance,'correction_mse':delta2,'identity_pass':True}
with (H/'n4_groups_v2.json').open('x',encoding='utf-8') as f:json.dump(results,f,ensure_ascii=False,indent=2)
print('N4_GROUPS',json.dumps(results))
