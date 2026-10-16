import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np
A = pd.concat([pd.read_csv(os.path.join(env.LOCAL, "fl3_train_links.csv"))[["rec", "prev", "p"]], pd.read_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv"))[["rec", "prev", "p"]]])
A = A[A.prev.notna() & (A.p >= .5)]; LK = dict(zip(A.rec, A.prev))
E = pd.read_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TE = TE[TE.row_id.str[:3].isin(["F13", "F47"])]
test = set(TE.row_id.str[:7]); mn = TE.assign(rec=TE.row_id.str[:7]).groupby("rec").in_temp.min()
def vis(r, x):   # eval rule: other-farm test records not visible; same-farm test only earlier numbers
    if x in test and (x[:3] != r[:3] or int(x[4:7]) >= int(r[4:7])): return False
    return True
rows = []
for r in sorted(test):
    d, cur, seen = 0, r, {r}
    while d < 7:
        nx = LK.get(cur)
        if nx is None or nx in seen or not vis(r, nx): break
        d += 1; seen.add(nx); cur = nx
    c1 = LK.get(r); rows.append(dict(rec=r, depth=d, c1=c1, c1_prevnum=(c1 == f"{r[:3]}_{int(r[4:7]) - 1:03d}") if isinstance(c1, str) else False, c1_lab=(isinstance(c1, str) and c1 not in test), cold=mn.get(r, 99) <= 6))
D = pd.DataFrame(rows)
print("eval records", len(D), "| depth distribution", D.depth.value_counts().sort_index().to_dict())
print("depth 0 (no history):", int((D.depth == 0).sum()), "| first link is a labelled training record:", int(D.c1_lab.sum()), "| first link == previous number:", int(D.c1_prevnum.sum()))
print("cold (day min <= 6C) records:", int(D.cold.sum()), "| of them depth 0:", int((D.cold & (D.depth == 0)).sum()), "| depth >= 2:", int((D.cold & (D.depth >= 2)).sum()))
D.to_csv(os.path.join(r"C:\work\farmai\집\클로드\research", "evcov1_eval_fl_chain_v1.csv"), index=False)
