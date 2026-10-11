# H-HOUSE: cache OOF residuals (temp DIAG10 4 cells, EC DIAG10 CUR seed-mean). read-only on existing files.
import sys, os, importlib.util, time
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
HERE = r"C:\work\farmai\집\클로드\research"; OUT = os.path.join(HERE, "agentX_house", "local"); os.makedirs(OUT, exist_ok=True)
t0 = time.time(); sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(HERE, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
M, cells = A2.A1.frames()["DIAG10"]
T = M[["row_id", "farm", "day", "hour", "sub_temp", "fold"]].copy()
for k, q in enumerate(cells): T[f"q{k}"] = np.asarray(q)
T.to_csv(os.path.join(OUT, "temp_oof_diag10.csv"), index=False); print("temp", T.shape, time.time() - t0, flush=True)
E = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv")
print(E.columns.tolist(), E.tag.unique()[:10], E.validator.unique()[:10])
E = E[(E.tag == "CUR") & (E.validator == "DIAG10")]
print("seeds", E.seed.unique())
key = [c for c in ["row_id", "farm", "day", "hour"] if c in E.columns]
G = E.groupby(key, as_index=False).agg(sg=("sg", "mean"), sub_ec=("sub_ec", "first"))
G.to_csv(os.path.join(OUT, "ec_oof_diag10.csv"), index=False); print("ec", G.shape, key)
