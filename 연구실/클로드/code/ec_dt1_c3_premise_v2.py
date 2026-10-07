# -*- coding: utf-8 -*-
"""DT1 전제 검산 v2 — v1 의 '진짜 앞날 = EC 자정 차 최소(후보 400)'는 우연 일치에 휘둘림 → CH2 식 결합 비용
  cost(a→b) = (EC 점프/.02)² + 외기 날짜 점프 비용(4채널 추세보정 표준화 제곱합, 가중 1/1/.3/1)
으로 진짜 앞날을 정의(정답 사용 = 진단). 그다음 합법 질문: b 의 0시 외기만으로(외기 날짜 점프 비용만) 앞날의 '날짜 묶음'을 고르면
진짜 앞날 날짜 묶음이 몇 위인가(1차 날짜 묶음 + 2차 정답 묶음 전체 대상). 대상: C3 정답 16, 비교로 C1·C2.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
R = os.path.dirname(os.path.abspath(__file__))
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; WT = dict(zip(W, (1, 1, .3, 1)))
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + W); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"] + W)
TR["is_test"] = False; TE["is_test"] = True
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
EC = Y[Y.farm.isin(["F13", "F47"])].pivot_table(index=["farm", "day"], columns="hour", values="sub_ec")
P = {v: X.pivot_table(index=["farm", "day"], columns="hour", values=v) for v in W}
istest = X.groupby(["farm", "day"]).is_test.first(); keys = list(P[W[0]].index)
sc = {v: np.nanstd(np.diff(P[v].loc[[k for k in keys if not istest[k]]].values, axis=1)) for v in W}
def jump(df, a, b):
    A23, A22, B0, B1 = df.loc[a, 23], df.loc[a, 22], df.loc[b, 0], df.loc[b, 1]
    return (B0 - A23) - .5 * ((A23 - A22) + (B1 - B0))
cls = pd.read_csv(os.path.join(R, "..", "results", "ec_dt0_date_class_v1.csv")).set_index(["farm", "day"]).cls
# 날짜 묶음 (외기 완전 쌍둥이)
WZ = pd.concat([(P[v] - P[v].loc[[k for k in keys if not istest[k]]].values.mean()) / P[v].loc[[k for k in keys if not istest[k]]].values.std() for v in W], axis=1)
A = WZ.loc[keys].values; Dm = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
par = list(range(len(keys)))
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for i in range(len(keys)):
    for j in np.where(Dm[i] <= .05)[0]:
        par[fd(i)] = fd(j)
grp = {k: fd(i) for i, k in enumerate(keys)}
labs = [k for k in EC.index]
rows = []
for b in labs:
    if b[1] < 179:
        continue
    cand = [a for a in labs if a != b and grp[a] != grp[b]]
    dc = np.array([sum(WT[v] * (jump(P[v], a, b) / sc[v]) ** 2 for v in W) for a in cand])
    ej = np.array([EC.loc[b, 0] - EC.loc[a, 23] for a in cand])
    tot = (ej / .02) ** 2 + dc
    k = int(np.nanargmin(tot)); t = cand[k]
    # 합법: 외기 날짜 비용만으로 날짜 묶음 순위 (묶음별 최소 비용)
    G = pd.Series(dc, index=[grp[a] for a in cand]).groupby(level=0).min().sort_values()
    rank = int(np.where(G.index == grp[t])[0][0]) + 1
    rows.append(dict(farm=b[0], day=b[1], cls=cls.get(b), y=EC.loc[b].mean(), t_farm=t[0], t_day=t[1], t_cost=tot[k], t_ecj=ej[k], t_dc=dc[k],
                     date_rank=rank, n_groups=len(G), second_cost=np.sort(tot)[1]))
O = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(O[O.cls == "C3"].round(3).to_string(index=False))
for c, q in O.groupby("cls"):
    print("%s n=%d | 진짜 앞날: 1차 %d·2차 %d, 다른 ID %d | 결합비용 중앙 %.2f (2등 %.2f) | 외기만으로 날짜묶음 순위: 1위 %d, ≤3위 %d, 중앙 %d (묶음 %d개)" % (
        c, len(q), (q.t_day < 179).sum(), (q.t_day >= 179).sum(), (q.t_farm != q.farm).sum(), q.t_cost.median(), q.second_cost.median(),
        (q.date_rank == 1).sum(), (q.date_rank <= 3).sum(), q.date_rank.median(), q.n_groups.median()))
O.to_csv(os.path.join(R, "..", "results", "ec_dt1_c3_premise_v2.csv"), index=False, encoding="utf-8-sig")
