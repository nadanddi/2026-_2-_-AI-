import sys,os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np
def load():
    X=pd.read_csv(os.path.join(env.DATA,'train_X.csv'));y=pd.read_csv(os.path.join(env.DATA,'train_y.csv'))
    T=pd.read_csv(os.path.join(env.DATA,'test_X.csv'))
    X=X.merge(y,on='row_id',how='left'); X['src']='train'; T['src']='test'
    D=pd.concat([X,T],ignore_index=True)
    p=D.row_id.str.split('_',expand=True); D['farm']=p[0]; D['rec']=p[1].astype(int); D['h']=p[2].astype(int)
    D=D[D.farm.isin(['F13','F47'])].copy()
    D['phase']=np.where(D.rec<179,1,2)
    D['ec25']=D.sub_ec/(1+0.02*(D.sub_temp-25))
    D['dTs']=D.sub_temp-D.in_temp
    D=D.sort_values(['farm','rec','h']).reset_index(drop=True)
    return D
