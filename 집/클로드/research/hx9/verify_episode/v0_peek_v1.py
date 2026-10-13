# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import pandas as pd
pd.set_option('display.width', 250)
R = pd.read_csv(r"C:\work\farmai\집\클로드\research\hx9\data\records.csv"); h = R[R.ec_mean >= 1]
print(h[['rec','pass2','ec_mean','ec_h0','ec_h23','sg_mean','resid_mean','prev_rec','prev_ec_h23','a1_c1','a1_cert','ep_id','dm11_day','fold']].round(3).to_string())
