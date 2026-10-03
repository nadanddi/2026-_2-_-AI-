# -*- coding: utf-8 -*-
"""AG0 (diagnostic, fixed before running; 2026-10-03 집 클로드).
High-EC days are ranked (AUC .989) but under-predicted in size (1.55 vs 1.22).  If
salts accumulate, the size should grow with the EPISODE AGE (how many consecutive
dates the same 동 has been in the high state).  Uses labels -> diagnostic only.
Episode: consecutive dates (C6.205 calendar, same pass) of the same 동 (pair role /
classifier, local/st_dong_assign_v1.csv) whose label day mean >= 0.9; age = 1, 2, ...
(gaps of unlabelled dates up to 2 allowed, age keeps counting by date).
DIAG10 R3S (seed mean) day residual vs age on high days (label >= 1).
Also an INPUT-ONLY age: consecutive previous same-동 dates whose R3S predicted day
mean >= 0.8 (what a legal feature could see), and residual vs it.
Reading (fixed): clue if Spearman(residual, age) >= .30 on high days (n >= 20) for
either age, label age reported as mechanism, input age as usability."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv"))
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy()
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
DD = O.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean")).reset_index()
rows = []
for f in ("F13", "F47"):
    G = D[D.farm == f].sort_values("day").copy()
    G["date"] = np.cumsum(G.role.values != "second") - 1
    G = G.merge(DD[DD.farm == f], on=["farm", "day"], how="left")
    for dg in ("A", "B"):
        H = G[G.dong == dg].sort_values("date")
        for ps in (0, 1):
            K = H[(H.day >= 179) == ps]
            lab_age, inp_age = 0, 0
            prev_date = None
            for _, r in K.iterrows():
                gap = None if prev_date is None else r.date - prev_date
                cont = gap is not None and gap <= 3
                # label age
                if pd.notna(r.y) and r.y >= 0.9:
                    lab_age = lab_age + (gap if cont else 1) if lab_age > 0 and cont else 1
                elif pd.notna(r.y):
                    lab_age = 0
                ia = inp_age if cont else 0     # input age known BEFORE today
                rows.append(dict(farm=f, dong=dg, day=r.day, y=r.y, p=r.p, lab_age=lab_age, inp_age_prev=ia))
                if pd.notna(r.p) and r.p >= 0.8:
                    inp_age = (inp_age + 1) if cont else 1
                elif pd.notna(r.p):
                    inp_age = 0
                prev_date = r.date
R = pd.DataFrame(rows).dropna(subset=["y", "p"])
R["res"] = R.y - R.p
H = R[R.y >= 1.0]
print("high days (label >= 1) with OOF: %d" % len(H))
print("by label episode age:")
print(H.assign(a=H.lab_age.clip(upper=6)).groupby("a").agg(n=("res", "size"), y=("y", "mean"), p=("p", "mean"), res=("res", "mean")).round(3).to_string())
print("by input-only prior age (prev same-dong dates predicted >= .8):")
print(H.assign(a=H.inp_age_prev.clip(upper=4)).groupby("a").agg(n=("res", "size"), y=("y", "mean"), p=("p", "mean"), res=("res", "mean")).round(3).to_string())
r1 = spearmanr(H.res, H.lab_age).correlation; r2 = spearmanr(H.res, H.inp_age_prev).correlation
r3 = spearmanr(R.res, R.inp_age_prev).correlation
print("Spearman high days: res~label age %.2f | res~input prior age %.2f | all days res~input prior age %.2f" % (r1, r2, r3))
print("\nAG0 clue:", (r1 >= .30 or r2 >= .30) and len(H) >= 20)
