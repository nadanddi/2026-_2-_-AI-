# HA2b (사후 진단, 판정 변경 아님): HA2 세 세계에서 이득 최대 기록 1·3개를 빼도 개선이 남는지 + 부호 검정(개선 기록 비율)
import sys, io, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd
from scipy.stats import binomtest
src = io.open("ha2_hard_anchor_ec_three_worlds_v1.py", encoding="utf-8").read()
g = {"__name__": "x", "__file__": os.path.abspath("ha2_hard_anchor_ec_three_worlds_v1.py")}
src = src.replace("def judge(name, base, new, Y, grp):", "JD = {}\ndef judge(name, base, new, Y, grp):\n    JD[name] = (base, new, Y)\n    return\ndef _unused(name, base, new, Y, grp):")
exec(compile(src, "ha2", "exec"), g)
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
for name, (base, new, Y) in g["JD"].items():
    recs = sorted(set.union(*[set(r for r in new[s].index if not np.allclose(new[s].loc[r].values, base[s].loc[r].values, equal_nan=True)) for s in new]))
    per = pd.Series(np.mean([(((new[s].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for s in new], 0), index=recs).sort_values()
    bt = binomtest(int((per < 0).sum()), int((per != 0).sum()), .5, alternative="greater").pvalue
    out = [f"sign test improved {int((per<0).sum())}/{len(per)} p {bt:.2g}"]
    for k in (1, 3):
        rr = [r for r in recs if r not in per.index[:k]]
        c = [rm((new[s].loc[rr] - Y.loc[rr]).values) / rm((base[s].loc[rr] - Y.loc[rr]).values) - 1 for s in new]
        out.append(f"drop top {k}: " + " ".join("%+.1f%%" % (100 * x) for x in c))
    print(name, "|", " | ".join(out))
