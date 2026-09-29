from common import *
R=pd.read_pickle("daykeys.pkl")
gid={k:i for i,k in enumerate(R.ot.unique())}; R["gid"]=R.ot.map(gid)
A,S=load()
day=A.groupby(["gh","day"]).agg(ec=("sub_ec","mean"),st=("sub_temp","mean"),it=("in_temp","mean"),ih=("in_hum","mean"),co2=("in_co2","mean"),
   co2min=("in_co2","min"),vent=("act_vent","mean"),circ=("act_circfan","mean"),heat=("act_heating","mean"),therm=("act_thermal","mean"),shade=("act_shade","mean"),fog=("act_fog","mean"),aco2=("act_co2","mean"),test=("is_test","max")).reset_index()
day=day.merge(R[["gh","day","gid"]],on=["gh","day"],how="left")
out=[]
for g in ["F13","F47"]:
    d=day[(day.gh==g)].sort_values("day").reset_index(drop=True)
    fh=d[d.day<178].copy()
    z=np.full(len(fh),-1); last=[None,None]
    i=0; gids=fh.gid.values; ec=fh.ec.values
    while i<len(fh):
        if i+1<len(fh) and gids[i]==gids[i+1]:
            a,b=ec[i],ec[i+1]
            if last[0] is None: z[i],z[i+1]=(0,1) if a<b else (1,0)
            else:
                c1=abs(a-last[0])+abs(b-last[1]); c2=abs(a-last[1])+abs(b-last[0])
                z[i],z[i+1]=(0,1) if c1<=c2 else (1,0)
            last[z[i]]=a; last[z[i+1]]=b; i+=2
        else:
            a=ec[i]
            if last[0] is None or last[1] is None: z[i]=0 if last[0] is None else 1
            else: z[i]=0 if abs(a-last[0])<=abs(a-last[1]) else 1
            last[z[i]]=a; i+=1
    fh["zone"]=z
    print(g, fh.groupby("zone")[["ec","st","it","ih","co2","co2min","vent","circ","heat","therm","shade","fog","aco2"]].mean().round(2))
    # pairwise differences within pairs: sign consistency zone1-zone0
    pr=fh[fh.duplicated("gid",keep=False)]
    piv=pr.pivot_table(index="gid",columns="zone",values=["it","ih","co2","vent","circ","heat","therm","st"])
    for c in ["it","ih","co2","vent","circ","heat","therm","st"]:
        dd=(piv[(c,1)]-piv[(c,0)]).dropna(); print("  ",c,"frac zone1>zone0",round((dd>0).mean(),2),"n",len(dd))
    print("  zone seq:", "".join(map(str,z)))
    out.append(fh)
pd.concat(out).to_pickle("zones_firsthalf.pkl")
