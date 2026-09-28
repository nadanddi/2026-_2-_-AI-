from common import *
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",500)
A,S=load()
for g in ["F47","F13","F02","F20"]:
    G=grid(A,g)
    d=G[["in_temp","sub_temp","out_temp","sub_ec","in_co2"]].diff()
    mid=G.hr==0
    print(g, "abs diff at midnight vs other hours:", {c:(round(d[c][mid].abs().median(),3), round(d[c][~mid].abs().median(),3)) for c in d.columns})
G=grid(A,"F47")
P=G.pivot_table(index="day",columns="hr",values="out_temp")
for a,b in [(102,103),(104,105),(100,101),(150,151),(183,184),(185,186),(220,221)]:
    if a in P.index and b in P.index: print(a,b, P.loc[a].values[:12], P.loc[b].values[:12])
# duplicate daily out_temp profiles
key=P.round(2).apply(lambda r: tuple(r.values),axis=1)
print("F47 days with duplicated full out_temp profile:", key.duplicated(keep=False).sum(), "of", len(key))
