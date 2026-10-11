# -*- coding: utf-8 -*-
"""MX0: dump day-level OOF residuals (temp DIAG10 4-cell mean; EC rv1 CUR DIAG10 seed mean) -> local/agentX_meta/resid.pkl"""
import os, sys, importlib.util
R0 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R0)
import env  # noqa
import numpy as np, pandas as pd
sys.argv = ["x"]
s = importlib.util.spec_from_file_location("ta2", os.path.join(R0, "ta2_anchor_debiased_v1.py")); A2 = importlib.util.module_from_spec(s); s.loader.exec_module(A2)
M, cells = A2.A1.frames()["DIAG10"]
M = M.copy(); M["pt"] = np.mean(np.vstack(cells), 0); M["rt"] = M.sub_temp - M.pt
E = pd.read_csv(r"C:\work\farmai\연구실\클로드\results\ec_rv1_preds_v1.csv")
E = E[(E.tag == "CUR") & (E.validator == "DIAG10")].groupby("row_id").agg(pe=("sg", "mean"), sub_ec=("sub_ec", "first"), fold_e=("fold", "first")).reset_index()
E["re"] = E.sub_ec - E.pe; E["rle"] = np.log(E.sub_ec.clip(.01)) - np.log(E.pe.clip(.01))
H = M[["row_id", "farm", "day", "hour", "fold", "sub_temp", "pt", "rt"]].merge(E, on="row_id", how="outer")
H["farm"] = H.row_id.str[:3]; H["day"] = H.row_id.str[4:7].astype(int); H["hour"] = H.row_id.str[8:10].astype(int)
out = os.path.join(R0, "local", "agentX_meta"); os.makedirs(out, exist_ok=True)
H.to_pickle(os.path.join(out, "resid.pkl"))
print(H.shape, H.rt.notna().sum(), H.re.notna().sum())
print(H.groupby(H.day >= 179)[["rt", "re"]].agg(lambda v: np.sqrt(np.nanmean(v ** 2))))
