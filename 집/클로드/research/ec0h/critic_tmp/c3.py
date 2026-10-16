import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"))
print(L.columns.tolist()); print(L[L.rec.isin(["F47_137","F47_163","F13_062","F47_136"])].T.to_string())
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]
for r in ["F47_135","F47_136","F47_137","F47_138"]:
    g = O[O.rec == r].groupby("hour").agg(y=("sub_ec","first"), sg=("sg","mean"), fold=("fold","first"))
    print(r, "fold", g.fold.iloc[0], "y", np.round(g.y.values[[0,1,12,22,23]],3), "sg", np.round(g.sg.values[[0,1,12,22,23]],3))
H = os.path.join(env.ROOT, u"집", u"클로드", "research")
R = pd.read_csv(os.path.join(H, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) == "7")]
for r in ["F47_136","F47_137"]:
    g = R[R.rec == r]; print("EV", r, "anch", g.anchored.iloc[0], "EA2", np.round(g.EA2.values[[0,1,12,22,23]],3), "R3", np.round(g.R3.values[[0,23]],3))
