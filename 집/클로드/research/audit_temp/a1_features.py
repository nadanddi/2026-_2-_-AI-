# Audit: build training (MASK) and evaluation feature frames exactly as make_submission_v13_temp.build_frames, no training.
import os, sys
os.environ["AGRI_DATA"] = r"C:\work\farmai\공용\대회자료\정형데이터\참가자_배포"
PK = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\온도\code"
sys.path.insert(0, PK)
import env
import numpy as np, pandas as pd
import make_submission_v13_temp as M
from resid_reset_features import FEATURE_COLUMNS, PHYSICS_COLUMNS
import train_flags_v6 as TF
OUT = r"C:\work\farmai\집\클로드\research\audit_temp"
lab, test, ct, phc, sX = M.build_frames()
w = TF.row_weights(lab, 0.2, w_noisy=0.2)
lab["w"] = w
cx = ["cx__" + c for c in FEATURE_COLUMNS]
keep = ["row_id", "farm", "day", "hour", "sub_temp", "in_temp", "w"]
allc = list(dict.fromkeys(ct + phc + cx))
lab[list(dict.fromkeys(keep + allc))].to_pickle(OUT + r"\lab_feats.pkl")
test[list(dict.fromkeys([k for k in keep if k not in ("sub_temp", "w")] + allc))].to_pickle(OUT + r"\test_feats.pkl")
import json
json.dump(dict(ct=ct, phc=phc, cx=cx), open(OUT + r"\cols.json", "w"))
print("lab", lab.shape, "test", test.shape, "ct", len(ct), "phc", len(phc), "cx", len(cx))
