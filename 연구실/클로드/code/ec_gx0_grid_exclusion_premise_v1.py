# -*- coding: utf-8 -*-
"""GX0 — 출처 4 × 날짜 128 격자 '배제법' 전제 검산 (진단, 가린 기록 b 의 정답·입력 미사용) — 2026-10-07 연구실 클로드
날짜 묶음 G = 외기 24h 완전 쌍둥이 연결 성분(두 온실 ID, 학습+평가).
2차 정답 기록 b 를 가림(정답 없음 취급). 실제 평가 기록 60개도 정답 없음.
 1) 전날 묶음 P: G 안의 다른 정답 기록 c 마다, 정답 기록 전체에서 CH2 비용
      cost(a→c) = (EC 점프/.02)² + 외기 날짜 비용 + 실내 비용/4  (학습 자료라 정답 사용 가능)
    최소 a 를 찾고, 그 a 들의 묶음 중 최빈 = P.  다음날 묶음 N 도 대칭(c→a).
 2) 배제: P 의 정답 기록과 G\{b} 정답 기록을 비용 헝가리안으로 짝지음 → P 에서 짝 없는 정답 기록이 정확히 1개면 b 의 앞날 후보 a*.
    N 도 대칭 → b 의 다음날 후보 n*.
 3) 추정: v = mean(EC23(a*), EC0(n*)) (있는 것만). 둘 다 없으면 추정 없음.
판정용(정답 사용): b 의 진짜 앞날(가린 기록 b 의 정답을 써서 같은 CH2 비용 최소) 과 a* 일치율, v 하루 RMSE vs SG2 하루평균(3 검증기 시드평균).
가짜 대조: P 의 짝 있는 정답 기록 중 하나를 임의로(첫 번째) 고른 값.
주의: b 하나만 가림(평가처럼 여러 기록 동시 가림 아님) → 낙관적. 날짜 묶음 판정에 쓰는 평가 기록 외기는 같은 날짜 쌍둥이 성분 구성에만 사용.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; WT = dict(zip(W, (1, 1, .3, 1))); IN = ["in_temp", "in_hum", "in_co2"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
TR["is_test"] = False; TE["is_test"] = True
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
EC = Y[Y.farm.isin(["F13", "F47"])].pivot_table(index=["farm", "day"], columns="hour", values="sub_ec")
P_ = {v: X.pivot_table(index=["farm", "day"], columns="hour", values=v) for v in W + IN}
istest = X.groupby(["farm", "day"]).is_test.first(); keys = list(P_[W[0]].index); trk = [k for k in keys if not istest[k]]
sc = {v: np.nanstd(np.diff(P_[v].loc[trk].values, axis=1)) for v in W + IN}
labs = list(EC.index); li = {k: i for i, k in enumerate(labs)}; n = len(labs)
def J(df, idx):
    M = df.loc[idx].values; A23, A22, B0, B1 = M[:, 23], M[:, 22], M[:, 0], M[:, 1]
    return (B0[None, :] - A23[:, None]) - .5 * ((A23 - A22)[:, None] + (B1 - B0)[None, :])
DC = sum(WT[v] * (J(P_[v], labs) / sc[v]) ** 2 for v in W)
IC = sum((J(P_[v], labs) / sc[v]) ** 2 for v in IN) / 4
EJ = J(EC, labs)
COST = np.where(np.isfinite(DC + IC + (EJ / .02) ** 2), DC + IC + (EJ / .02) ** 2, 1e6); np.fill_diagonal(COST, 1e6)
# 날짜 묶음
WZ = pd.concat([(P_[v] - P_[v].loc[trk].values.mean()) / P_[v].loc[trk].values.std() for v in W], axis=1)
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
mem = {}
for k, g in grp.items():
    mem.setdefault(g, []).append(k)
cls = pd.read_csv(os.path.join(R, "..", "results", "ec_dt0_date_class_v1.csv")).set_index(["farm", "day"]).cls

def side(b, direction):
    """direction = -1 전날, +1 다음날. b 는 가림."""
    others = [c for c in mem[grp[b]] if c != b and c in li]
    if not others:
        return None, None, "같은날 정답 없음"
    avail = [k for k in labs if k != b]
    ai = np.array([li[k] for k in avail])
    picks = []
    for c in others:
        col = COST[ai, li[c]] if direction < 0 else COST[li[c], ai]
        picks.append(avail[int(np.argmin(col))])
    gs = pd.Series([grp[p] for p in picks]).value_counts()
    Pg = gs.index[0]
    Pl = [k for k in mem[Pg] if k in li and k != b]
    if not Pl:
        return None, None, "이웃묶음 정답 없음"
    C = np.array([[COST[li[p], li[c]] if direction < 0 else COST[li[c], li[p]] for c in others] for p in Pl])
    r_, c_ = linear_sum_assignment(C)
    left = [Pl[i] for i in range(len(Pl)) if i not in set(r_)]
    if len(left) != 1:
        return None, (Pl[r_[0]] if len(r_) else None), "남은 기록 %d개" % len(left)
    return left[0], Pl[r_[0]], "결정"
rows = []
for b in labs:
    if b[1] < 179:
        continue
    a, fa, sa = side(b, -1); nn, fn, sn = side(b, +1)
    vals = ([EC.loc[a, 23]] if a else []) + ([EC.loc[nn, 0]] if nn else [])
    fv = ([EC.loc[fa, 23]] if fa else []) + ([EC.loc[fn, 0]] if fn else [])
    # 진짜 앞날(정답 사용, 판정용)
    t = labs[int(np.argmin(COST[:, li[b]]))]; ts = labs[int(np.argmin(COST[li[b], :]))]
    rows.append(dict(farm=b[0], day=b[1], cls=cls.get(b), y=EC.loc[b].mean(), prev=sa, nxt=sn, hit_prev=(a == t) if a else np.nan,
                     hit_next=(nn == ts) if nn else np.nan, v=np.mean(vals) if vals else np.nan, fake=np.mean(fv) if fv else np.nan))
O = pd.DataFrame(rows)
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1)
sgd = S.groupby(["validator", "farm", "day"]).sg.mean().rename("sgd").reset_index()
print("2차 정답 %d | 전날 상태 %s | 다음날 상태 %s" % (len(O), O.prev.value_counts().to_dict(), O.nxt.value_counts().to_dict()))
print("배제 결정 시 진짜 앞날 적중 %.2f (n=%d), 진짜 다음날 적중 %.2f (n=%d) | 추정 있는 날 %d" % (
    O.hit_prev.mean(), O.hit_prev.notna().sum(), O.hit_next.mean(), O.hit_next.notna().sum(), O.v.notna().sum()))
rm = lambda x: np.sqrt(np.nanmean(np.square(x)))
for v in ("DIAG10", "DIAG10y", "EL1"):
    Q = O.merge(sgd[sgd.validator == v], on=["farm", "day"])
    for nm, m in (("추정 있는 날", Q.v.notna()), ("  그중 일반", Q.v.notna() & (Q.y < 1)), ("  그중 C3", Q.v.notna() & (Q.cls == "C3"))):
        q = Q[m]
        print("  %-7s %-12s n=%2d | SG2 %.3f | 배제 추정 %.3f | 가짜(짝 있는 기록) %.3f | 반반 %.3f" % (
            v, nm, len(q), rm(q.sgd - q.y), rm(q.v - q.y), rm(q.fake - q.y), rm((q.v + q.sgd) / 2 - q.y)))
O.to_csv(os.path.join(R, "..", "results", "ec_gx0_premise_v1.csv"), index=False, encoding="utf-8-sig")
