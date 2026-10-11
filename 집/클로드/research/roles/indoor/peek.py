import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import os, env, pandas as pd
D = str(env.DATA)
X = pd.read_csv(os.path.join(D, 'train_X.csv')); y = pd.read_csv(os.path.join(D, 'train_y.csv'))
print(X.shape, X.columns.tolist()); print(X.head(3).T); print(y.shape, y.columns.tolist()); print(y.head(3).T)
r = pd.read_csv(r'C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv')
print(r.shape, r.columns.tolist()); print(r.head(3).T)
for c in ['tag', 'validator']:
    if c in r: print(r[c].value_counts())
