# -*- coding: utf-8 -*-
"""agentY_econset y2: truth.pkl cert 연결은 EC 23->0시 연속(|차|<=~0.01)을 요구하므로 '밤사이 EC 점프로 시작하는 고EC'를 구조적으로 못 본다.
EC를 쓰지 않는 연결(배지온도 연속 + 외기 연속 + 실내온도·습도 연속)로 이전 날짜 같은 동을 다시 찾아,
고EC 날(특히 cert 이전이 없는 날)의 직전 EC와 밤사이 점프를 본다. 진단용 오라클(배지온도 정답 사용)."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env, os, pickle, numpy as np, pandas as pd
OUTD = os.path.dirname(os.path.abspath(__file__))
R = pickle.load(open(r"C:\work\farmai\집\클로드\research\local\linker_v3\base.pkl", "rb"))
r = R["recs"].reset_index(drop=True); EC = R["EC"]; ST = R["ST"]; X = R["OUT"]; IN = R["IN"]
T = pd.read_pickle(r"C:\work\farmai\집\클로드\research\local\linker_v3\truth.pkl")
cert_prev = dict(zip(T[T.cert].b, T[T.cert].a1))
tr = np.where(r.split == "train")[0]
sdO = np.nanstd(X[tr].reshape(-1, 4), 0); sdI = np.nanstd(IN[tr].reshape(-1, 3), 0)
pe = lambda A, i: A[i, 23] + 0.5 * (A[i, 23] - A[i, 22])
pb = lambda A, j: A[j, 0] - 0.5 * (A[j, 1] - A[j, 0])
ecm = np.nanmean(EC, 1)
rows = []
for j in tr:
    cs = np.abs(pe(ST, tr) - pb(ST, j)) / 0.3
    oj = np.nansum(np.array([1, 1, .3, .5]) * np.abs(pe(X, tr) - pb(X, j)) / sdO, 1)
    ci = np.nansum(np.array([1, 1, 0]) * np.abs(pe(IN, tr) - pb(IN, j)) / sdI, 1)
    c = cs + oj + 0.5 * ci; c[tr == j] = np.inf
    o = np.argsort(c); a = tr[o[0]]
    rows.append(dict(b=j, rid=f"{r.farm[j]}_{r.day[j]:03d}", a=a, arid=f"{r.farm[a]}_{r.day[a]:03d}", c1=c[o[0]], c2=c[o[1]],
                     oj=oj[o[0]], cs=cs[o[0]], ec_b=ecm[j], ec_a=ecm[a], ec_b0=EC[j, 0], ec_a23=EC[a, 23],
                     cert_prev=cert_prev.get(j, -1), same_as_cert=(cert_prev.get(j, -1) == a), samefarm=r.farm[a] == r.farm[j]))
L = pd.DataFrame(rows)
L["jump"] = L.ec_b0 - L.ec_a23; L["margin"] = L.c2 - L.c1
conf = (L.c1 <= 1.5) & (L.margin >= 1.0)
L["conf"] = conf
print("EC-free links: confident", conf.sum(), "/", len(L), "; agree with cert where cert exists:",
      L[(L.cert_prev >= 0) & conf].same_as_cert.mean().round(3), "n", ((L.cert_prev >= 0) & conf).sum())
print("|jump 23->0| among confident EC-free links q50/90/99:", np.nanquantile(np.abs(L[conf].jump), [.5, .9, .99]).round(3))
print("confident links with |jump|>0.1:", (conf & (L.jump.abs() > 0.1)).sum(), "; >0.3:", (conf & (L.jump.abs() > 0.3)).sum())
pd.set_option("display.width", 250)
hi = L.ec_b >= 1.0
print("\nHigh days (mean>=1.0):", hi.sum(), " with cert prev:", (hi & (L.cert_prev >= 0)).sum())
cols = ["rid", "ec_b", "ec_b0", "cert_prev", "arid", "ec_a", "ec_a23", "jump", "c1", "margin", "oj", "cs", "conf", "samefarm"]
print(L[hi & (L.cert_prev < 0)][cols].round(3).to_string(index=False))
print("\nHigh days with cert prev: (prev EC means)")
print(L[hi & (L.cert_prev >= 0)][["rid", "ec_b", "ec_a", "jump"]].round(3).to_string(index=False))
# how high episodes begin, by within-record EC: does EC rise within a record (ramp) across threshold?
print("\nRecords with intra-day rise: EC23-EC0 > 0.3:", ((EC[tr, 23] - EC[tr, 0]) > 0.3).sum(),
      " drop < -0.3:", ((EC[tr, 23] - EC[tr, 0]) < -0.3).sum())
x = EC[tr, 23] - EC[tr, 0]
print(" intra-day change q01/50/99", np.nanquantile(x, [.01, .5, .99]).round(3))
# episodes on calendar? number of records
print("\nNUM of high days by farm:")
for f in ["F13", "F47"]:
    print(f, sorted(r.day[tr][(ecm[tr] >= 1.0) & (r.farm[tr] == f).values].tolist()))
L.to_csv(os.path.join(OUTD, "y2_ecfree_links.csv"), index=False)
