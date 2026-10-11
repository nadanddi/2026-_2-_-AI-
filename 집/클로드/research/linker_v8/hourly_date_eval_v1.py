# For evaluation records whose date is unknown at hour 0 but known later: linker v7 decision by hour (full v7 weights).
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v8")
from feat8 import *
import env
w=np.load(os.path.join(env.LOCAL,"linker_v7","w_v7.npy"))
idx={rid(i):i for i in range(N)}
for rec in ["F13_185","F13_201","F13_206","F13_228","F47_234"]:
    b=idx[rec]; out=[]
    for h in (0,3,6,9,12,18,23):
        df,info=features7(b,h,TRAIN.copy(),removed=set())
        s=df[FEATS7].values.astype(float)@w; p=np.exp(s-s.max()); p/=p.sum(); k=int(np.argmax(p)); a=int(df.a.values[k])
        out.append("h%d:%s p%.2f %s m%s"%(h,rid(a),p[k],"L" if TRAIN[a] else "U",info.get("nmatch")))
    print(rec," | ".join(out))
