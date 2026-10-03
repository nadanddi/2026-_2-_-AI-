from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
lab,core,wv,folds,outer=S.loadec();v,k,tm,vm=folds[0];assert v=='DIAG10' and k==0
tr,va=S.seasonal(lab[tm],lab[vm],wv);W=['out_temp','out_hum','out_rad','out_wspd'];weather=pd.read_csv(Path(env.DATA)/'train_X.csv',usecols=['row_id']+W);tr=tr.merge(weather,on='row_id',validate='one_to_one');va=va.merge(weather,on='row_id',validate='one_to_one');cols=[c for c in core.FULL if c!='day']+['season']+W
with threadpool_limits(limits=2):
 m=core.et(7);m.steps[-1][1].n_jobs=2;raw=core.predict_model(m,tr,va,cols);base=core.shrink(raw,va)
 old=pd.read_csv(ROOT/'집/코덱스/local/ec_current_weather_20261003_v1/DIAG10_0_7.csv',float_precision='round_trip');assert np.array_equal(old.row_id,va.row_id);diff=float(np.max(np.abs(old.new_et_raw-raw)));assert diff<1e-12
 checks=[]
 for farm in ['F13','F47']:
  for hour in [0,6,12]:
   allowed=va.farm.eq(farm)&va.hour.le(hour);q=va.copy();q.loc[~allowed,W]=q.loc[~allowed,W]*17+1000
   p=m.predict(q[cols]);s=core.shrink(p,q);assert np.array_equal(p[allowed],raw[allowed]) and np.array_equal(s[allowed],base[allowed]);checks.append(dict(farm=farm,hour=hour,rows=int(allowed.sum()),future_and_other_farm_weather_invariance='PASS'))
 single=m.predict(va.iloc[:8][cols]);assert np.array_equal(single,raw[:8])
(H/'causal_weather_verification_v1.json').write_text(json.dumps(dict(status='PASS',replay_maxdiff=diff,columns=len(cols),checks=checks,first8_batch_invariance='PASS',scope='first fold seed7 inference audit; current-row W4 only; training transform fixed; no test values or targets read'),ensure_ascii=False,indent=2),encoding='utf-8');print('PASS',len(checks),diff)
