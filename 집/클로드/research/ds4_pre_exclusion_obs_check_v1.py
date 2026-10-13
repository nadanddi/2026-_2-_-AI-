import os, sys
R = r"C:\work\farmai\집\클로드\research"; sys.path.insert(0, R); os.chdir(R)
src = open("ds1_day_state_hmm_ec_v1.py", encoding="utf-8").read(); __file__ = os.path.join(R, "ds1_day_state_hmm_ec_v1.py")
exec(src[:src.index("res = []; recrows = []")])
G = O[O.seed == O.seed.iloc[0]].sort_values(["rec", "hour"]).reset_index(drop=True)
day_sg = G.groupby("rec").sg.mean(); hidden = G.groupby("fold").rec.apply(set).to_dict()
rows = []
for k, hs in hidden.items():
    tr = [r for r in day_sg.index if r not in hs]; vis = TRAIN & ~np.array([rid(i) in hs for i in range(N)])
    ss = np.array([DM[idx[r]] >= 1.0 for r in tr]); zs = np.array([zflag(idx[r], 23, vis) for r in tr])
    piN = (zs[~ss].sum() + 1) / ((~ss).sum() + 2); piH = (zs[ss].sum() + 1) / (ss.sum() + 2)
    zt = {r: zflag(idx[r], 23, vis) for r in hs}
    p2 = [r for r in hs if NUM[idx[r]] >= 179]
    rows.append(dict(fold=k, piN=round(piN, 3), piH=round(piH, 3), z_on_test=sum(zt.values()), z_on_p2=sum(zt[r] for r in p2), n_p2=len(p2),
                     fake=[(r, zt[r]) for r in ("F13_231", "F13_232", "F13_233") if r in hs]))
import pandas as pd; print(pd.DataFrame(rows).to_string(index=False))
