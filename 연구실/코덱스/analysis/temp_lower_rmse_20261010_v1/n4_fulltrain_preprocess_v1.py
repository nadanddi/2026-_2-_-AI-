"""Check final-training season/weight contracts against immutable round9 tables; no fit."""
from pathlib import Path
import sys
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'temp_repair_20261009_v1'))
from runtime_v1 import *
cf=cframes();ids=RAW[1].loc[RAW[1].farm.isin(common.TARGET_FARMS)&RAW[1].sub_temp.notna(),'row_id']
tr=cf.loc[ids].reset_index();tr['sub_temp']=RAW[1].set_index('row_id').sub_temp.loc[ids].to_numpy()
q=cf.loc[RAW[2].row_id].reset_index()
a,b,notes=season_frames(tr,q)
td=pd.read_csv(CODE/'season_ec/training_day_to_season.csv').set_index(['farm','day']).season
qd=pd.read_csv(CODE/'season_ec/evaluation_day_to_season.csv').set_index(['farm','day']).season
ta=np.array([td[f,int(d)] for f,d in zip(tr.farm,tr.day)]);qa=np.array([qd[f,int(d)] for f,d in zip(q.farm,q.day)])
dt=a.day.to_numpy()-ta;dq=b.day.to_numpy()-qa;w=weights(tr)
obj={'training_season_max':float(np.max(np.abs(dt))),'query_season_max':float(np.max(np.abs(dq))),'training_season_exact':bool(np.array_equal(a.day.to_numpy(),ta)),'query_season_exact':bool(np.array_equal(b.day.to_numpy(),qa)),'training_rows':len(tr),'query_rows':len(q),'weights':{str(v):int((w==v).sum()) for v in np.unique(w)},'feature_order_preserved':a.columns.tolist()==tr.columns.tolist() and b.columns.tolist()==q.columns.tolist(),'new_fit':0,'evaluation_predictions_created':False,'scope':'preprocessing contract only, not original whole-model reproduction'}
with (H/'n4_fulltrain_preprocess_v1.json').open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
print('N4_FULLTRAIN_PREPROCESS',json.dumps(obj))
