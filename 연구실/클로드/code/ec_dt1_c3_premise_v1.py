# -*- coding: utf-8 -*-
"""DT1 전제 검산 — C3(1차에 없는 날짜) 2차 기록의 전날·같은 날 정보 (진단 전용, 정답은 판정·오라클에만) — 2026-10-07 연구실 클로드
날짜 묶음 = 외기 24h 완전 쌍둥이 연결 성분(두 온실 ID, 학습+평가 기록). 2차는 각 온실 ID 안에서 달력 순서.
C3 정답 기록 b 에 대해:
 Q1 같은 날짜 묶음 구성(정답 기록/평가 기록/온실 ID)
 Q2 전날 묶음 = 같은 ID 2차 순서에서 b 의 묶음 바로 앞 묶음(다른 날짜인 첫 앞 기록의 묶음). 그 묶음 기록 중 정답 있는 것 수.
 Q3 진짜 앞날(정답 자정 연속 |EC_b(0) − EC_a(23)| 최소, 정답 기록 전체 두 ID)이 전날 묶음 / 같은 날 묶음 / 그 밖 어디에 있나
 Q4 SG2 위 상한(하루 수준): SG2 하루평균 vs 전날 묶음 정답 기록 중 오라클 선택 23시 EC, 가짜 묶음(2·3 묶음 전) 오라클, 같은 날짜 정답 기록 오라클
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd

R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + W); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=["row_id"] + W)
TR["is_test"] = False; TE["is_test"] = True
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
EC = Y[Y.farm.isin(["F13", "F47"])].pivot_table(index=["farm", "day"], columns="hour", values="sub_ec")
WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W); istest = X.groupby(["farm", "day"]).is_test.first()
trk = [k for k in WV.index if not istest[k]]; WZ = (WV - WV.loc[trk].mean()) / WV.loc[trk].std()
keys = list(WZ.index); A = WZ.values; idx = {k: i for i, k in enumerate(keys)}
Dm = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
par = list(range(len(keys)))
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for i in range(len(keys)):
    for j in np.where(Dm[i] <= .05)[0]:
        par[fd(i)] = fd(j)
grp = {k: fd(i) for k, i in idx.items()}
members = {}
for k, g in grp.items():
    members.setdefault(g, []).append(k)
cls = pd.read_csv(os.path.join(R, "..", "results", "ec_dt0_date_class_v1.csv")).set_index(["farm", "day"]).cls
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1)
sgd = S.groupby(["validator", "farm", "day"]).sg.mean()

def prev_group(f, d, steps=1):
    """같은 ID 기록 순서에서 b 의 날짜 묶음보다 steps 번째 앞 날짜 묶음 (다른 묶음이 나올 때마다 1)"""
    ds = sorted(e for ff, e in keys if ff == f and e < d); g0 = grp[(f, d)]; seen = []
    for e in reversed(ds):
        g = grp[(f, e)]
        if g != g0 and (not seen or seen[-1] != g):
            seen.append(g)
            if len(seen) == steps:
                return g
    return None
rows = []
labs = set(EC.index)
for (f, d), c in cls.items():
    if c != "C3" or (f, d) not in labs:
        continue
    same = [k for k in members[grp[(f, d)]] if k != (f, d)]
    g1 = prev_group(f, d, 1)
    pv = members.get(g1, []) if g1 is not None else []
    allc = [k for k in labs if k != (f, d)]
    j = np.array([abs(EC.loc[(f, d), 0] - EC.loc[k, 23]) for k in allc]); best = allc[int(np.argmin(j))]
    loc = "전날묶음" if best in pv else "같은날" if best in same else ("1차" if best[1] < 179 else "2차 다른곳")
    def orac(cands):
        cl = [k for k in cands if k in labs]
        if not cl:
            return np.nan
        e = np.array([EC.loc[k, 23] for k in cl]); return e[np.argmin(np.abs(e - EC.loc[(f, d), 0]))]
    row = dict(farm=f, day=d, y=EC.loc[(f, d)].mean(), n_same=len(same), n_same_lab=sum(k in labs for k in same),
               n_prev=len(pv), n_prev_lab=sum(k in labs for k in pv), n_prev_test=sum(bool(istest[k]) for k in pv),
               best_loc=loc, best_jump=j.min(), o_prev=orac(pv), o_same=orac(same),
               o_fake2=orac(members.get(prev_group(f, d, 2), [])), o_fake3=orac(members.get(prev_group(f, d, 3), [])))
    for v in ("DIAG10", "DIAG10y", "EL1"):
        row["sg_" + v] = sgd.get((v, f, d), np.nan)
    rows.append(row)
O = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("C3 정답 기록 %d" % len(O))
print(O[["farm", "day", "y", "n_same", "n_same_lab", "n_prev", "n_prev_lab", "n_prev_test", "best_loc", "best_jump", "o_prev", "o_same", "sg_DIAG10"]].round(3).to_string(index=False))
print("\n진짜 앞날 위치:", O.best_loc.value_counts().to_dict(), "| 전날 묶음에 정답 기록 있는 날 %d/%d" % ((O.n_prev_lab > 0).sum(), len(O)))
rm = lambda x: np.sqrt(np.nanmean(np.square(x)))
for v in ("DIAG10", "DIAG10y", "EL1"):
    q = O[O["sg_" + v].notna()]
    for nm, m in (("전체", q.y > -1), ("일반", q.y < 1)):
        qq = q[m]
        print("  %-7s %s n=%2d | SG2 %.3f | 전날묶음 오라클 %.3f (있는 날 %d) | 가짜 2묶음전 %.3f · 3묶음전 %.3f | 같은날 오라클 %.3f" % (
            v, nm, len(qq), rm(qq["sg_" + v] - qq.y), rm(qq.o_prev - qq.y), qq.o_prev.notna().sum(), rm(qq.o_fake2 - qq.y), rm(qq.o_fake3 - qq.y), rm(qq.o_same - qq.y)))
O.to_csv(os.path.join(R, "..", "results", "ec_dt1_c3_premise_v1.csv"), index=False, encoding="utf-8-sig")
