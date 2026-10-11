# RZ5b (사후): 천창 0..12시 <=3시간 조건이 고EC 상태·계절과 겹친 것인지 확인
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np
F = pd.read_csv("rz5_features_v1.csv", index_col=0); e = F.up >= .10
for nm, m in (("all", F.up.notna()), ("0h>=.9", F.lvl0 >= .9), ("0h<.9", F.lvl0 < .9), ("late cal>=115 (p1)", F.cal >= 115), ("p2", F.p2)):
    c = F.hrs_vent <= 3
    print(nm, "n", int(m.sum()), "events", int((e & m).sum()), "| vent<=3h: share of events %.2f, share of all %.2f, P(rise|cond) %.2f vs P(rise|not) %.2f" % (c[m & e].mean(), c[m].mean(), e[m & c].mean(), e[m & ~c].mean()))
print("Spearman vent hours vs lvl0 %.2f, vs cal %.2f" % (F.hrs_vent.corr(F.lvl0, method="spearman"), F.hrs_vent.corr(F.cal, method="spearman")))
