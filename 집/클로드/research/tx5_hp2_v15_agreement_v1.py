import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); os.chdir(r"C:\work\farmai\집\클로드\research"); sys.argv=["x"]
import env, numpy as np, pandas as pd
src = open("tb1_temp_null_agree_anchor_v1.py", encoding="utf-8").read(); __file__ = os.path.abspath("tb1_temp_null_agree_anchor_v1.py")
exec(src[:src.index("PE15 = ")])   # hp2, mirror_lab, CLEAN, DAYT, SEGT, FPA ...
PM = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_mirror.csv")).set_index("rec")
rows = []
for r in PM.index:
    b = idx[r]; lab = mirror_lab(r); h = hp2(b, lab); true = CLEAN.get(r)
    rows.append(dict(rec=r, p2=NUM[b] >= 179, true=true, true_vis=(true is not None and lab[idx[true]]), hp2=h, hp2_ok=(h == true) if (h and true) else None, v15=PM.at[r, "prev0"], p0=PM.at[r, "p0"], v15_ok=(PM.at[r, "prev0"] == true) if true else None))
T = pd.DataFrame(rows)
for nm, g in (("all", T), ("pass2", T[T.p2])):
    gv = g[g.true_vis]
    print(f"{nm}: records with visible labelled true prev {len(gv)} | HP2 fires {gv.hp2.notna().mean():.2f} acc {gv.hp2_ok.dropna().astype(bool).mean():.3f} | v15 acc (top1) {gv.v15_ok.dropna().astype(bool).mean():.3f} | v15 p0>=.8 share {(gv.p0 >= .8).mean():.2f}")
    low = gv[gv.p0 < .8]
    print(f"   v15 p0<.8 subset n {len(low)}: v15 top1 acc {low.v15_ok.dropna().astype(bool).mean():.3f} | HP2 fires {low.hp2.notna().mean():.2f} acc {low.hp2_ok.dropna().astype(bool).mean():.3f} | HP2==v15 share {(low.hp2 == low.v15).mean():.2f} with acc {low[low.hp2 == low.v15].v15_ok.dropna().astype(bool).mean():.3f} | HP2!=v15 & HP2 ok {(low[(low.hp2 != low.v15) & low.hp2.notna()].hp2_ok.dropna().astype(bool)).mean():.3f} n {int(((low.hp2 != low.v15) & low.hp2.notna()).sum())}")
for r in ["F47_191", "F47_241", "F47_239", "F13_244", "F47_178", "F47_145", "F47_141", "F47_132"]:
    x = T[T.rec == r].iloc[0]; print(f"  {r}: true {x.true} vis {x.true_vis} | HP2 {x.hp2} | v15 {x.v15} {x.p0:.2f}")
