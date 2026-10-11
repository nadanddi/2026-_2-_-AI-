from load import *
D=load(); D['key']=D.farm+'_'+D.rec.astype(str); D=D[D.src=='train']
def runinfo(c,lin=False):
    out=[]
    for k,g in D.groupby('key'):
        s=g[c].values
        x=np.diff(s,2) if lin else np.diff(s)
        z=np.abs(x)<1e-9
        if lin: z&=np.abs(np.diff(s)[1:])>1e-9
        # longest run
        best=0;cur=0;end=0
        for i,v in enumerate(z):
            cur=cur+1 if v else 0
            if cur>best: best,end=cur,i
        if best>=(4 if lin else 5): out.append((k,best+ (2 if lin else 1),end, s[end] if not lin else None))
    return out
for c in ['sub_ec','sub_temp','in_hum','in_temp']:
    r=runinfo(c); print(c,'같은값 6h+ 기록',len(r), r[:10])
    r=runinfo(c,True); print(c,'완전 직선(보간형) 6h+ 기록',len(r), r[:10])
g=D[D.key.isin(['F47_133','F47_155','F47_158'])][['key','h','in_co2','act_circfan','act_vent','act_thermal','in_temp','in_hum']]
print(g[g.h.isin([0,1,2,3,4,5,20,21,22,23])].to_string())
