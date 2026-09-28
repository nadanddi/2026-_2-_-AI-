from common import *
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
A,S=load()
for g in ["F13","F47"]:
    G=grid(A,g)
    tr=G[G.is_test==0]
    print(g, "corr sub_temp", tr[FEATS+["sub_temp","sub_ec"]].corr()["sub_temp"].round(3).dropna().to_dict())
    print(g, "corr sub_ec", tr[FEATS+["sub_temp","sub_ec"]].corr()["sub_ec"].round(3).dropna().to_dict())
    # lag relationship sub_temp vs in_temp
    for L in range(0,5):
        print(" lag",L, "corr(sub_temp, in_temp shift L)", round(G.sub_temp.corr(G.in_temp.shift(L)),4))
    r=(G.sub_temp-G.in_temp); print(" resid sub-in mean,std", r.mean().round(3), r.std().round(3))
    # EC daily
    dm=G.groupby("day").sub_ec.mean()
    for L in [1,2,3,5,7,10,12]:
        print(" daily EC autocorr lag",L, round(dm.corr(dm.shift(L)),3), "rmse persist", round(np.sqrt(((dm-dm.shift(L))**2).mean()),4))
    # within-day vs between
    print(" EC total std", G.sub_ec.std().round(4), "within-day std", (G.sub_ec-G.groupby("day").sub_ec.transform("mean")).std().round(4))
    print(" EC by hour mean", G.groupby("hr").sub_ec.mean().round(3).values)
    print(" hourly EC diff std", G.sub_ec.diff().std().round(4))
    print(" temp by hour", G.groupby("hr").sub_temp.mean().round(2).values)
