import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, os
L=env.LOCAL
T=pd.read_pickle(os.path.join(L,'linker_v3','truth.pkl'));print(T.head());print(T.columns.tolist(),len(T),T.cert.sum())
for f in ['cv_all400_choice_v7m.csv','eval60_predecessor_v7.csv','eval60_confident_1step_anchor_v7.csv']:
  d=pd.read_csv(os.path.join(L,'linker_v7',f));print(f,d.shape);print(d.head(3).to_string())
