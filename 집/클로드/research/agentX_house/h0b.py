import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
t=pd.read_pickle(r'C:\work\farmai\집\클로드\research\local\linker_v3\truth.pkl')
print(t.head(10)); print(t.dtypes)
for c in t.columns:
    if t[c].nunique()<10: print(c, t[c].value_counts().to_dict())
print(t.cert.describe() if 'cert' in t else 'no cert')
