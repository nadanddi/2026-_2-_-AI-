"""Q1g: dump daily series for eyeballing regimes."""
import env, common, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
tX, ty, sX = common.load_raw()
X = pd.concat([tX, sX], ignore_index=True).merge(ty[['row_id','sub_temp','sub_ec']],on='row_id',how='left')
X['test']=X.row_id.isin(sX.row_id)
for f in ['F13','F47']:
    d=X[X.farm==f]
    M=d.groupby('day').agg(ec=('sub_ec','mean'),ecsd=('sub_ec','std'),st=('sub_temp','mean'),it=('in_temp','mean'),ih=('in_hum','mean'),co2=('in_co2','mean'),heat=('act_heating','mean'),th=('act_thermal','mean'),vent=('act_vent','mean'),fan=('act_circfan','mean'),aco2=('act_co2','mean'),fog=('act_fog','mean'),shade=('act_shade','mean'),ot=('out_temp','mean'),test=('test','max'))
    M.round(2).to_csv(f'local/eda_time_daily_{f}.csv')
    print(f); print(M.round(2).to_string())
