from common import *
pd.set_option("display.width",250); pd.set_option("display.max_rows",500)
R=pd.read_pickle("daykeys.pkl")
gid={k:i for i,k in enumerate(R.ot.unique())}; R["gid"]=R.ot.map(gid)
A,S=load()
co2=A.groupby(["gh","day"]).in_co2.mean().rename("co2"); R=R.join(co2,on=["gh","day"])
for g in ["F47","F13"]:
    x=R[R.gh==g].sort_values("day").reset_index(drop=True)
    out=[]
    for _,r in x.iterrows():
        tw=R[(R.gid==r.gid)&~((R.gh==g)&(R.day==r.day))]
        s=" ".join(f"{a}{b}{'T' if t else ''}(ec{e:.2f},st{st:.1f})" for a,b,t,e,st in zip(tw.gh,tw.day,tw.test,tw.ec,tw.st))
        out.append(f"{r.day:3d} g{r.gid:3d} {'T' if r.test else ' '} ec={r.ec:.3f} st={r.st:5.2f} it={r.it:5.2f} co2={r.co2:5.0f} | {s}")
    print(g); print("\n".join(out))
