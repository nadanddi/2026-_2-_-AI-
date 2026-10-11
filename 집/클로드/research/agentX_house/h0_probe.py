import sys,os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import pickle,pandas as pd
L=os.path.join(env.LOCAL,'linker_v3') if hasattr(env,'LOCAL') else None
print(env.LOCAL)
t=pd.read_pickle(r'C:\work\farmai\집\클로드\research\local\linker_v3\truth.pkl');print(type(t));print(t.head());print(t.shape);print(t.describe())
R=pickle.load(open(r'C:\work\farmai\집\클로드\research\local\linker_v3\base.pkl','rb'));print(R.keys())
r=R['recs'].reset_index(drop=True);print(r.head());print(r.columns.tolist(), len(r))
for k,v in R.items():
    print(k, type(v), getattr(v,'shape',None))
