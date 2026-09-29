from common import *
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",500)
A,S=load()
G=grid(A,"F47")
print(G.loc[(G.day>=150)&(G.day<=153),["day","hr","out_temp","in_temp","in_hum","out_rad","act_vent","act_heating","sub_temp","sub_ec"]].to_string())
dm=G.groupby("day")[["sub_ec","sub_temp","in_temp","out_temp"]].mean().round(3)
print(dm.loc[100:245].to_string())
