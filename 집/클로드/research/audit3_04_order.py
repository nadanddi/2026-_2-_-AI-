import env, common
import numpy as np, pandas as pd
from scipy.stats import spearmanr
tX, ty, sX = common.load_raw()
tX['is_test']=False; sX['is_test']=True
A=pd.concat([tX,sX]).sort_values(['farm','t'])
g=pd.read_csv('local/audit3_03_groups.csv').set_index(['farm','day'])
# monotonic F13 vs F47 day within shared groups, per segment
rows=[]
for k,s in g.groupby('mine'):
    a=[d for (f,d) in s.index if f=='F13']; b=[d for (f,d) in s.index if f=='F47']
    if a and b: rows.append((min(a),max(a),min(b),max(b)))
r=pd.DataFrame(rows,columns=['a0','a1','b0','b1'])
for name,m in [('both seg1',(r.a0<179)&(r.b0<179)),('both seg2',(r.a1>=179)&(r.b1>=179))]:
    x=r[m]; print(name,len(x),'spearman F13min vs F47min',spearmanr(x.a0,x.b0)[0].round(3))
# midnight jump in out_temp (23->0) per farm, versus other hour jumps
def jump(df,col):
    df=df.sort_values('t'); d=df[col].diff(); dt=df.t.diff()
    h=df.hour
    m0=(dt==1)&(h==0); mo=(dt==1)&(h!=0)
    return d[m0].abs().median(), d[mo].abs().median(), d[m0].abs().mean(), d[mo].abs().mean()
for f in ['F13','F47','F32']:
    df=A[A.farm==f]
    for c in ['out_temp','out_hum','in_temp']:
        print(f,c,'mid/other median,mean',np.round(jump(df,c),3))
# out_temp midnight jump split by: next day same group (twin) vs not
for f in ['F13','F47']:
    df=A[A.farm==f].sort_values('t')
    last=df[df.hour==23].set_index('day').out_temp; first=df[df.hour==0].set_index('day').out_temp
    res={'same':[],'diff':[]}
    for d in last.index:
        if d+1 in first.index and (f,d) in g.index and (f,d+1) in g.index:
            k='same' if g.loc[(f,d),'mine']==g.loc[(f,d+1),'mine'] else 'diff'
            res[k].append(abs(first[d+1]-last[d]))
    print(f,{k:(len(v),np.round(np.median(v),2)) for k,v in res.items()})
