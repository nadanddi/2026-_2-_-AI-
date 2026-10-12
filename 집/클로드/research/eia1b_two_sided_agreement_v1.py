import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env, pandas as pd, numpy as np
E = pd.read_csv(os.path.join(env.LOCAL, "eia1_eval_availability.csv"))
both = E.vb.notna() & E.vf.notna()
E["agree"] = both & ((E.vb - E.vf).abs() <= .15)
E["est"] = np.where(E.agree, (E.vb + E.vf) / 2, np.nan)
E["near"] = E.agree & (E.kb <= 2) & (E.kf <= 2)
E["d"] = E.est - E.p12
pd.set_option("display.width", 200)
print(E[E.agree].sort_values("d", key=abs, ascending=False)[["rec", "kb", "vb", "kf", "vf", "est", "p12", "d", "near"]].round(3).head(25).to_string())
print("\nagree:", int(E.agree.sum()), "near(k<=2 both):", int(E.near.sum()), "| disagree both-sided:", int((both & ~E.agree).sum()))
for m, nm in ((E.agree, "agree"), (E.near, "near")):
    print(f"{nm}: SSE change potential if est were exact day level: sum 24*d^2 = {24 * (E.d[m] ** 2).sum():.2f}  (12th total SSE 23.26; .05 needs ~3.6)")
print("\ndisagreeing records:"); print(E[both & ~E.agree][["rec", "kb", "vb", "kf", "vf", "p12"]].round(3).to_string())
