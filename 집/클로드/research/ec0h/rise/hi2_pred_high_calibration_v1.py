# HI2 (서술): 예측 하루 평균 구간별로 실제 하루 평균은? — 모델이 '높다'고 할 때 얼마나 맞고, 실제가 더 높은지 (DIAG10 sg, EV SG2/EA2)
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
D = pd.DataFrame(dict(q=O.groupby("rec").sg.mean(), y=O.groupby("rec").sub_ec.mean()))
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
E = pd.DataFrame(dict(q=R.groupby("rec").EA2.mean(), y=R.groupby("rec").y.mean()))
bins = [0, .4, .6, .8, 1.0, 1.2, 1.5, 9]
for nm, X in (("DIAG10 sg", D), ("EV EA2", E)):
    X = X.assign(b=pd.cut(X.q, bins)); X["p2"] = X.index.str[4:7].astype(int) >= 179
    g = X.groupby("b", observed=True).agg(n=("y", "size"), pred=("q", "mean"), truth=("y", "mean"), truly_high=("y", lambda s: float((s >= 1).mean())), n_p2=("p2", "sum"))
    print(f"\n{nm}: predicted day-mean band -> actual"); print(g.round(2).to_string())
    hi = X[X.q >= 1.0]; print(f"  pred >= 1.0: n {len(hi)} | truly >= 1.0: {int((hi.y>=1).sum())} | mean truth/pred {(hi.y/hi.q).mean():.2f} | false-high (truth < .8): {int((hi.y<.8).sum())}")
    th = X[X.y >= 1.0]; print(f"  truly high n {len(th)}: predicted >= 1.0 {int((th.q>=1).sum())}, predicted < .8 {int((th.q<.8).sum())}")
