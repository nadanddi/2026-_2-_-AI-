# -*- coding: utf-8 -*-
"""Forensic 6: residual episodes (sub_temp vs causal in_temp model) and their input signatures."""
import env
import numpy as np, pandas as pd
G = pd.read_pickle(env.LOCAL+"/eda_forensic_5_G.pkl")
tr = G[G.set=="train"].copy()
# run length of identical in_temp values (centered), and dist to nearest NaN
for f in ["F13","F47"]:
    g = G[G.farm==f]
    v = g.in_temp.values; t=g.t.values
    rl = np.ones(len(v),int)
    # compute runs
    start=0
    for i in range(1,len(v)+1):
        if i==len(v) or not (v[i]==v[start] and t[i]-t[start]==i-start):
            rl[start:i]=i-start; start=i
    G.loc[g.index,"temp_run"]=rl
    nat = t[np.isnan(v)]
    G.loc[g.index,"dist_na"] = [np.min(np.abs(nat-x)) if len(nat) else 999 for x in t]
tr = G[G.set=="train"].copy()
print("temp_run>=6 share train/test by farm:")
print(G.assign(r6=G.temp_run>=6).groupby(["farm","set"]).r6.mean().mul(100).round(2))
print("temp_run>=4:",G.assign(r=G.temp_run>=4).groupby(["farm","set"]).r.mean().mul(100).round(2).to_dict())
# residual by dist_na bucket
tr["dna"]=pd.cut(tr.dist_na,[-1,0,1,3,6,12,24,48,1e9])
print(tr.groupby("dna").res.agg(["size",lambda s: s.abs().mean(), lambda s:(s.abs()>2).mean()]).round(3))
tr["trun"]=pd.cut(tr.temp_run,[0,1,2,3,5,8,100])
print(tr.groupby("trun").res.agg(["size",lambda s: s.abs().mean(), lambda s:(s.abs()>2).mean()]).round(3))
# episodes
eps=[]
for f in ["F13","F47"]:
    g = tr[tr.farm==f]
    big = (g.res.abs()>2.5).values; t=g.t.values
    i=0
    while i<len(g):
        if big[i]:
            j=i
            while j+1<len(g) and big[j+1] and t[j+1]==t[j]+1: j+=1
            if j-i+1>=3:
                s=g.iloc[i:j+1]
                eps.append(dict(farm=f,day=int(s.day.iloc[0]),h0=int(s.hour.iloc[0]),len=j-i+1,res=round(s.res.mean(),2),
                   min_dist_na=int(s.dist_na.min()), max_trun=int(s.temp_run.max()), na=int(s.in_temp.isna().sum())))
            i=j+1
        else: i+=1
E=pd.DataFrame(eps); print(E.to_string()); print("episodes",len(E),"near NA(<=6h)",(E.min_dist_na<=6).sum(),"flat run>=4",(E.max_trun>=4).sum())
E.to_csv(env.LOCAL+"/eda_forensic_6_episodes.csv",index=False)
G.to_pickle(env.LOCAL+"/eda_forensic_6_G.pkl")
