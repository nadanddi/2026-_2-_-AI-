import env, common
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
tX, ty, sX = common.load_raw()
z=np.load('local/oof_temp_diag.npz',allow_pickle=True)
P=pd.DataFrame({'row_id':z['row_id'],'r3':z['r3']}).merge(ty[['row_id','sub_temp','farm','day','hour']],on='row_id').merge(tX,on=['row_id','farm','day','hour'])
fl=pd.read_csv('local/eda_forensic_11_flags.csv').set_index('row_id').Vany
P['flag']=P.row_id.map(fl).fillna(False)
P['res']=P.sub_temp-P.r3
C=P[~P.flag.astype(bool)]
off=C.groupby(['farm','day']).res.mean().rename('off')
print('day offset rms %.3f, within-day rms %.3f'%(np.sqrt((off**2).mean()), np.sqrt(((C.res-C.groupby(['farm','day']).res.transform('mean'))**2).mean())))
cols=common.USABLE
agg=P.groupby(['farm','day'])[cols].agg(['mean','min','max','std'])
agg.columns=['_'.join(c) for c in agg.columns]
h0=P[P.hour==0].set_index(['farm','day'])[cols].add_suffix('_h0')
h23=P[P.hour==23].set_index(['farm','day'])[['in_temp','in_hum','in_co2']].add_suffix('_h23')
pm=P.groupby(['farm','day']).r3.mean().rename('predmean')
D=agg.join(h0).join(h23).join(pm).join(off,how='inner')
S=pd.read_csv('local/audit3_10_daystats.csv').set_index(['farm','day'])
D=D.join(S[['co2_d2','co2_ac1_d1','T_d1','grp']]); D['Q4']=D.grp.eq('Q4').astype(int); D=D.drop(columns='grp')
g=pd.read_csv('local/audit3_03_groups.csv').set_index(['farm','day'])
D=D.join(g.mine)
# twin deltas: day mean in_temp minus mean in_temp of same-date days (both farms)
D['it_rel_date']=D.in_temp_mean-D.groupby('mine').in_temp_mean.transform('mean')
D['farm13']=(D.index.get_level_values(0)=='F13').astype(int); D['seg2']=(D.index.get_level_values(1)>=179).astype(int)
X=D.drop(columns=['off','mine']).fillna(-999); y=D.off.values
grp=D.mine.values
gkf=GroupKFold(5)
et=ExtraTreesRegressor(300,min_samples_leaf=5,max_features=0.3,n_jobs=2,random_state=0)
p=cross_val_predict(et,X,y,cv=gkf,groups=grp)
print('ET R2 date-grouped CV: %.3f'%(1-((y-p)**2).sum()/((y-y.mean())**2).sum()))
Xr=X.replace(-999,0); Xr=(Xr-Xr.mean())/Xr.std().replace(0,1)
p2=cross_val_predict(RidgeCV(alphas=np.logspace(-1,4,20)),Xr,y,cv=gkf,groups=grp)
print('Ridge R2: %.3f'%(1-((y-p2)**2).sum()/((y-y.mean())**2).sum()))
for c in ['predmean','Q4','co2_d2','it_rel_date','in_temp_h0','in_hum_mean','seg2','farm13','in_temp_std']:
    print(c,'corr %.3f'%np.corrcoef(D[c].fillna(0),y)[0,1])
et.fit(X,y); imp=pd.Series(et.feature_importances_,X.columns).sort_values()[::-1][:8]; print(imp.round(3).to_dict())
# excluding Q4 days
m=D.Q4==0
p3=cross_val_predict(et,X[m],y[m],cv=gkf,groups=grp[m]); yy=y[m]
print('ET R2 on non-Q4 days: %.3f'%(1-((yy-p3)**2).sum()/((yy-yy.mean())**2).sum()))
print('Q4 offset mean %.3f rms %.3f ; nonQ4 mean %.3f rms %.3f'%(y[~m].mean(),np.sqrt((y[~m]**2).mean()),y[m].mean(),np.sqrt((y[m]**2).mean())))
et.fit(X[m],y[m]); imp=pd.Series(et.feature_importances_,X.columns).sort_values()[::-1][:10]; print(imp.round(3).to_dict())
for seed in [1,2]:
    blk=(D.index.get_level_values(1)//15).astype(str)+D.index.get_level_values(0)
    rs=[]
    et2=ExtraTreesRegressor(300,min_samples_leaf=5,max_features=0.3,n_jobs=2,random_state=seed)
    pp=cross_val_predict(et2,X[m],y[m],cv=GroupKFold(5),groups=blk[m])
    print('seed',seed,'block-grouped R2 nonQ4 %.3f'%(1-((yy-pp)**2).sum()/((yy-yy.mean())**2).sum()))
# top single features corr in nonQ4
Dm=D[m]
cs=Dm.drop(columns=['off','mine']).apply(lambda c: np.corrcoef(c.fillna(c.median()),Dm.off)[0,1]).abs().sort_values()[::-1][:10]
print(cs.round(3).to_dict())
