import pandas as pd, numpy as np, os
E = pd.read_csv(r"C:\work\farmai\집\클로드\research\audit_temp\eval_sanity.csv")
D = r"C:\work\farmai\공용\대회자료\정형데이터\참가자_배포"
tX = pd.read_csv(os.path.join(D, "train_X.csv")); ty = pd.read_csv(os.path.join(D, "train_y.csv"))
tr = tX.merge(ty, on="row_id"); tr["farm"] = tr.row_id.str[:3]; tr["hour"] = tr.row_id.str[8:10].astype(int)
tr = tr[tr.farm.isin(["F13", "F47"])]
pd.set_option("display.width", 250)
x = E[(E.farm == "F47") & (E.day == 220)][["hour", "in_temp", "sub_temp", "p13", "base", "codex", "pfn", "exp_gap", "n_match"]]
print(x.round(2).to_string(index=False))
print("training rows with in_temp<=7:", int((tr.in_temp <= 7).sum()), " min sub_temp there %.2f, mean gap(sub-in) %.2f" % (tr[tr.in_temp <= 7].sub_temp.min(), (tr.sub_temp - tr.in_temp)[tr.in_temp <= 7].mean()))
print("eval rows in_temp<=7:", int((E.in_temp <= 7).sum()), "mean pred gap %.2f" % (E.sub_temp - E.in_temp)[E.in_temp <= 7].mean())
print("training sub_temp min %.2f ; eval pred min %.2f" % (tr.sub_temp.min(), E.sub_temp.min()))
print("cold eval rows (in_temp<=8) by hour: mean pred gap vs train gap at in_temp<=8")
a = E[E.in_temp <= 8].assign(g=lambda d: d.sub_temp - d.in_temp).groupby("hour").g.agg(["mean", "count"])
b = tr[tr.in_temp <= 8].assign(g=lambda d: d.sub_temp - d.in_temp).groupby("hour").g.agg(["mean", "count"])
print(pd.concat([a, b], axis=1, keys=["eval_pred", "train_label"]).round(2).T.to_string())
