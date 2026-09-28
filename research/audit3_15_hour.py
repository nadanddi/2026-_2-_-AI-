import env, common
import numpy as np, pandas as pd
tX, ty, sX = common.load_raw()
z=np.load('local/oof_temp_diag.npz',allow_pickle=True)
P=pd.DataFrame({'row_id':z['row_id'],'r3':z['r3']}).merge(ty[['row_id','sub_temp','farm','day','hour']],on='row_id')
fl=pd.read_csv('local/eda_forensic_11_flags.csv').set_index('row_id').Vany
P=P[~P.row_id.map(fl).fillna(False).astype(bool)]
P['res']=P.sub_temp-P.r3
P['w']=P.res-P.groupby(['farm','day']).res.transform('mean')
h=P.groupby('hour').agg(rms=('res',lambda v:np.sqrt((v**2).mean())),wrms=('w',lambda v:np.sqrt((v**2).mean())),bias=('res','mean'))
print(h.round(3).T.to_string())
# first-hours offset vs late offset: does error at h0-3 differ from rest-of-day offset (transient carry-over)?
e=P.assign(early=P.hour<=3).groupby(['farm','day','early']).res.mean().unstack()
print('corr early(0-3) vs rest offset %.3f; rms early-rest diff %.3f'%(e.corr().iloc[0,1],np.sqrt(((e[True]-e[False])**2).mean())))
