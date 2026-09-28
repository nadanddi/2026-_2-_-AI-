# -*- coding: utf-8 -*-
"""Q4 diagnostic (analysis only, uses labels of OTHER days -> never a feature):
compare the submitted test EC day-means with a calendar-neighbour estimate."""
import os
import env  # noqa
import numpy as np
import pandas as pd

S = pd.read_csv(env.LOCAL + "/deep_cal_11_days.csv")
S["second"] = (S.day > 178).astype(int)
sub = None
for n in ["submission_04.csv", "submission_03.csv"]:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submissions", n)
    if os.path.exists(p):
        sub = pd.read_csv(p); print("using", n); break
sub["farm"] = sub.row_id.str[:3]; sub["day"] = sub.row_id.str[4:7].astype(int)
sm = sub.groupby(["farm", "day"]).sub_ec.mean().rename("pred").reset_index()
T = S[S.is_test].merge(sm, on=["farm", "day"])
lab = S.dropna(subset=["ec_m"])
rows = []
for _, r in T.iterrows():
    L = lab[lab.farm == r.farm]
    ncal = L.iloc[(L.cal - r.cal).abs().argsort()[:4]]
    nday = L.iloc[(L.day - r.day).abs().argsort()[:4]]
    rows.append(dict(farm=r.farm, day=r.day, cal=r.cal, pred=r.pred, ec_calnb=ncal.ec_m.mean(), ec_daynb=nday.ec_m.mean()))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.round(3).to_string())
for f in ["F13", "F47"]:
    x = R[R.farm == f]
    print(f, "mean pred %.3f | cal-neighbour %.3f | day-neighbour %.3f | corr(pred, calnb) %.2f corr(pred, daynb) %.2f"
          % (x.pred.mean(), x.ec_calnb.mean(), x.ec_daynb.mean(), x.pred.corr(x.ec_calnb), x.pred.corr(x.ec_daynb)))
# reference: on labelled second-pass days, which neighbour estimate is closer
sp = lab[lab.second == 1]
e1, e2 = [], []
for _, r in sp.iterrows():
    L = lab[(lab.farm == r.farm) & (lab.second == 0)]
    e1.append(r.ec_m - L.iloc[(L.cal - r.cal).abs().argsort()[:4]].ec_m.mean())
    L2 = lab[(lab.farm == r.farm) & (lab.day != r.day)]
    e2.append(r.ec_m - L2.iloc[(L2.day - r.day).abs().argsort()[:4]].ec_m.mean())
print("labelled 2nd-pass days: bias cal-nb %+.3f rmse %.3f | bias day-nb %+.3f rmse %.3f"
      % (np.mean(e1), np.sqrt(np.mean(np.square(e1))), np.mean(e2), np.sqrt(np.mean(np.square(e2)))))
