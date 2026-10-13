import os, sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); os.chdir(r"C:\work\farmai\집\클로드\research")
import env
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score as A
D=pd.read_csv('ds1x_case_rows_v1.csv'); P=D[D.pred>=1.0]
print('pred>=1 records',len(P),'trueH',int(P.trueH.sum()))
q=P.dropna(subset=['prev_pred'])
print('AUC pred',round(A(P.trueH,P.pred),3),'| AUC prev_pred',round(A(q.trueH,q.prev_pred),3))
print(P.groupby('trueH').pred.describe().round(3))
print('trueH share by pred bin', P.groupby(pd.cut(P.pred,[1,1.1,1.2,1.3,1.5,3])).trueH.agg(['size','mean']).round(2).to_dict('index'))
