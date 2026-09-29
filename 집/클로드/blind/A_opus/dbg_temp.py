from cv import *
M,o=run("sub_temp",twin=True,seeds=(0,))
for (g,b),(y,p) in o.items():
    e=(p-y).reshape(-1,24)
    print(g,b,"per-day rmse",np.sqrt((e**2).mean(1)).round(2),"bias",e.mean(1).round(2))
    print("    by hour rmse", np.sqrt((e**2).mean(0)).round(1))
