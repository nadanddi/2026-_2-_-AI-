"""Causal feature builder. Every feature for row (gh, t) uses only inputs at times <= t of the same greenhouse.
Label-derived features (twin_*) use train_y of strictly earlier days of the same greenhouse and can be masked."""
from common import *

LAGV=["in_temp","in_hum","in_co2","out_temp","out_rad","out_hum","act_heating","act_thermal","act_shade","act_vent","act_circfan"]

def _gh_features(G):
    G=G.copy()
    G["hr"]=G.index%24; G["day"]=G.index//24
    f={}
    same=lambda k: (G["day"].shift(k)==G["day"])
    for v in LAGV:
        if G[v].notna().sum()==0: continue
        for k in (1,2,3,4,6):
            s=G[v].shift(k); f[f"{v}_x{k}"]=s                       # cross-midnight lag (grid)
            f[f"{v}_d{k}"]=s.where(same(k))                        # within-day lag
        f[f"{v}_dcm"]=G.groupby("day")[v].transform(lambda s:s.expanding().mean())
    it=G["in_temp"]
    for hl in (1,2,3,4,6,8,12):
        f[f"it_ewx{hl}"]=it.ewm(halflife=hl,ignore_na=True).mean()                      # across days
        f[f"it_ewd{hl}"]=G.groupby("day")["in_temp"].transform(lambda s:s.ewm(halflife=hl,ignore_na=True).mean())  # within day
    for hl in (2,6):
        f[f"ot_ewd{hl}"]=G.groupby("day")["out_temp"].transform(lambda s:s.ewm(halflife=hl,ignore_na=True).mean())
    f["it_dmax"]=G.groupby("day")["in_temp"].cummax(); f["it_dmin"]=G.groupby("day")["in_temp"].cummin()
    f["orad_dcs"]=G.groupby("day")["out_rad"].cumsum()
    f["heat_dcs"]=G.groupby("day")["act_heating"].cumsum()
    f["it_diff1"]=it-it.shift(1)
    return pd.concat([G,pd.DataFrame(f,index=G.index)],axis=1)

def build(A, ghs):
    out=[]
    for g in ghs:
        G=grid(A,g)
        G=G[FEATS+["row_id","is_test","sub_temp","sub_ec"]]
        F=_gh_features(G); F["gh"]=g; F["t"]=F.index
        out.append(F[F.row_id.notna()])
    return pd.concat(out,ignore_index=True)

def twin_map(A, g):
    T=day_table(A,g); return dict(zip(T.day,T.twin))

def add_twin(F, A, hidden=None):
    """twin_*: labels/inputs of the same-date other-compartment day (earlier day, same gh). hidden: set of (gh,day) whose labels are masked."""
    hidden=hidden or set()
    F=F.copy()
    key=F.set_index(["gh","day","hr"])
    lab=key[["sub_temp","sub_ec","in_temp"]]
    tw=[]
    for g in F.gh.unique():
        m=twin_map(A,g)
        tw.append(pd.Series(F.loc[F.gh==g,"day"].map(m).values,index=F.index[F.gh==g]))
    F["twin_day"]=pd.concat(tw)
    idx=pd.MultiIndex.from_arrays([F.gh,F.twin_day.fillna(-1).astype(int),F.hr])
    tv=lab.reindex(idx)
    F["twin_st"]=tv.sub_temp.values; F["twin_ec"]=tv.sub_ec.values; F["twin_it"]=tv.in_temp.values
    # twin day mean EC
    dm=lab.groupby(level=[0,1]).sub_ec.mean()
    F["twin_ecd"]=dm.reindex(pd.MultiIndex.from_arrays([F.gh,F.twin_day.fillna(-1).astype(int)])).values
    mask=np.array([(g,int(d)) in hidden if d==d else False for g,d in zip(F.gh,F.twin_day)])
    for c in ["twin_st","twin_ec","twin_ecd"]: F.loc[mask,c]=np.nan
    F["twin_st_minus_it"]=F.twin_st-F.twin_it
    F["twin_st_adj"]=F.twin_st-F.twin_it+F.in_temp
    return F
