import env, common
import numpy as np, pandas as pd
tX, ty, sX = common.load_raw()
s4=pd.read_csv('submissions/submission_04.csv'); s4['farm']=s4.row_id.str[:3]; s4['day']=s4.row_id.str[4:7].astype(int)
pe=s4.groupby(['farm','day']).sub_ec.mean()
tl=ty[ty.farm.isin(['F13','F47'])].groupby(['farm','day']).sub_ec.mean()
g=pd.read_csv('local/audit3_03_groups.csv').set_index(['farm','day'])
dc=pd.read_csv('local/deep_cal_11_days.csv').set_index(['farm','day'])
for f in ['F13','F47']:
    print(f)
    rng=range(179,246)
    line=[]
    for d in rng:
        if (f,d) in pe.index: line.append(f'{d}:P{pe[(f,d)]:.2f}c{dc.chain.get((f,d),-1)}')
        elif (f,d) in tl.index: line.append(f'{d}:T{tl[(f,d)]:.2f}c{dc.chain.get((f,d),-1)}')
    print(' '.join(line))
# train: day-level EC alternation in seg2 vs seg1: lag1 vs lag2 abs diff
for f in ['F13','F47']:
    x=tl.loc[f]
    for seg,(a,b) in {'seg1':(0,178),'seg2':(179,300)}.items():
        xs=x[(x.index>=a)&(x.index<=b)]
        l1=(xs-xs.reindex(xs.index-1).values).abs().mean(); l2=(xs-xs.reindex(xs.index-2).values).abs().mean()
        print(f,seg,'lag1 %.3f lag2 %.3f'%(l1,l2))
# predicted test: lag1 vs lag2
for f in ['F13','F47']:
    x=pe.loc[f]; l1=(x-x.reindex(x.index-1).values).abs().mean(); l2=(x-x.reindex(x.index-2).values).abs().mean()
    print('pred',f,'lag1 %.3f lag2 %.3f'%(l1,l2))
# same-date twin EC difference in train vs predicted in test
def twin(ser,farm):
    out=[]
    for k,s in g.groupby('mine'):
        ds=[d for (ff,d) in s.index if ff==farm and (farm,d) in ser.index]
        if len(ds)==2: out.append(abs(ser[(farm,ds[0])]-ser[(farm,ds[1])]))
    return np.mean(out),len(out)
for f in ['F13','F47']: print(f,'twin |dEC| train',np.round(twin(tl,f),3),'pred test',np.round(twin(pe,f),3))
for f in ['F13','F47']:
    out={1:[],2:[]}
    for k,s in g.groupby('mine'):
        ds=sorted(d for (ff,d) in s.index if ff==f and (f,d) in tl.index)
        if len(ds)==2 and ds[1]-ds[0]==1: out[1 if ds[0]<179 else 2].append(abs(tl[(f,ds[0])]-tl[(f,ds[1])]))
    print(f,'adjacent twin |dEC| seg1 %.3f (n=%d) seg2 %.3f (n=%d)'%(np.mean(out[1]),len(out[1]),np.mean(out[2]) if out[2] else np.nan,len(out[2])))
# train seg2 near block3 per-day EC with chain for F47 and F13 sorted
