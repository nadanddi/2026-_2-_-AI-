# -*- coding: utf-8 -*-
"""VW0 (verify_water) — 액추에이터 분포 훑기 (서술)."""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import pandas as pd
H = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hourly.csv"))
for c in ['act_fog','act_co2','act_circfan','act_vent','act_heating','act_shade','act_thermal','in_co2']:
    print(c, H[c].describe().round(2).to_dict(), 'share>0', round((H[c]>0).mean(), 3))
TX = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TX = TX[TX.row_id.str[:3].isin(["F13", "F47"])]
print('test rows', len(TX), 'NaN share per col'); print(TX.isna().mean().round(3).to_dict())
