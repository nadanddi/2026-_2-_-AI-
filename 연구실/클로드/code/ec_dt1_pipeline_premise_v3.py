# -*- coding: utf-8 -*-
"""DT1 전제 검산 v3 — '외기로 앞날 날짜 묶음 → 실내 자정 연속으로 그 묶음 안 기록 → 그 기록 23시 EC' 하루 수준 vs SG2 (진단)
 v2 와 같은 비용 정의. 후보는 정답 기록만(평가·가림 기록이 앞날인 경우는 이번 상한에서 '정답 앞날 없음'으로 남음).
 합법 파이프라인(L): 날짜묶음 = 외기 날짜 비용(b 의 0·1시 외기, 학습 기록) 최소 묶음 → 그 묶음 정답 기록 중 실내(온·습·CO2) 자정 점프 비용 최소 → 23시 EC
 가짜(F): 외기 2위 묶음에서 같은 방식.  오라클(O): v2 진짜 앞날(정답 사용).
 SG2: 저장 OOF 3 검증기 시드평균 하루평균. 얹기: 보호 .30, 반영 1 (하루 수준 단위, 상한 확인용).
 주의(검증 공정성): 이번 검산은 검증기 폴드 가림을 후보에 반영하지 않음 → 다음 계획에서 반영 필수.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; WT = dict(zip(W, (1, 1, .3, 1))); IN = ["in_temp", "in_hum", "in_co2"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
TR["is_test"] = False; TE["is_test"] = True
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
EC = Y[Y.farm.isin(["F13", "F47"])].pivot_table(index=["farm", "day"], columns="hour", values="sub_ec")
P = {v: X.pivot_table(index=["farm", "day"], columns="hour", values=v) for v in W + IN}
istest = X.groupby(["farm", "day"]).is_test.first(); keys = list(P[W[0]].index); trk = [k for k in keys if not istest[k]]
sc = {v: np.nanstd(np.diff(P[v].loc[trk].values, axis=1)) for v in W + IN}
def jump(df, a, b):
    A23, A22, B0, B1 = df.loc[a, 23], df.loc[a, 22], df.loc[b, 0], df.loc[b, 1]
    return (B0 - A23) - .5 * ((A23 - A22) + (B1 - B0))
WZ = pd.concat([(P[v] - P[v].loc[trk].values.mean()) / P[v].loc[trk].values.std() for v in W], axis=1)
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
cls = pd.read_csv(os.path.join(R, "..", "results", "ec_dt0_date_class_v1.csv")).set_index(["farm", "day"]).cls
V2 = pd.read_csv(os.path.join(R, "..", "results", "ec_dt1_c3_premise_v2.csv")).set_index(["farm", "day"])
labs = list(EC.index)
rows = []
for b in labs:
    if b[1] < 179:
        continue
    cand = [a for a in labs if a != b and grp[a] != grp[b]]
    dc = pd.Series([sum(WT[v] * (jump(P[v], a, b) / sc[v]) ** 2 for v in W) for a in cand], index=cand)
    G = dc.groupby([grp[a] for a in cand]).min().sort_values()
    def pick(g):
        mem = [a for a in cand if grp[a] == g]
        ic = [sum((jump(P[v], a, b) / sc[v]) ** 2 for v in IN) for a in mem]
        a = mem[int(np.nanargmin(ic))]; return a, EC.loc[a, 23]
    aL, vL = pick(G.index[0]); aF, vF = pick(G.index[1])
    t = (V2.loc[b, "t_farm"], int(V2.loc[b, "t_day"]))
    rows.append(dict(farm=b[0], day=b[1], cls=cls.get(b), y=EC.loc[b].mean(), vL=vL, vF=vF, vO=EC.loc[t, 23], hitL=(aL == t), hit_date=(grp[aL] == grp[t])))
O = pd.DataFrame(rows)
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1)
sgd = S.groupby(["validator", "farm", "day"]).sg.mean().rename("sgd").reset_index()
rm = lambda x: np.sqrt(np.nanmean(np.square(x)))
print("2차 정답 %d | 합법 파이프라인: 날짜 적중 %.2f, 기록 적중 %.2f (C3: %.2f / %.2f)" % (len(O), O.hit_date.mean(), O.hitL.mean(),
      O[O.cls == "C3"].hit_date.mean(), O[O.cls == "C3"].hitL.mean()))
for v in ("DIAG10", "DIAG10y", "EL1"):
    Q = O.merge(sgd[sgd.validator == v], on=["farm", "day"])
    def guard(x):
        return np.where(np.abs(x - Q.sgd) <= .30, x, Q.sgd)
    for nm, m in (("전체", Q.y > -1), ("일반", Q.y < 1), ("C3", Q.cls == "C3"), ("C3 일반", (Q.cls == "C3") & (Q.y < 1))):
        q = Q[m]; gL, gF, gO = guard(Q.vL)[m.values], guard(Q.vF)[m.values], guard(Q.vO)[m.values]
        print("  %-7s %-7s n=%2d | SG2 %.3f | 합법 L 그대로 %.3f, 보호 %.3f | 가짜 F 보호 %.3f | 오라클 O 보호 %.3f, 그대로 %.3f" % (
            v, nm, len(q), rm(q.sgd - q.y), rm(q.vL - q.y), rm(gL - q.y), rm(gF - q.y), rm(gO - q.y), rm(q.vO - q.y)))
O.to_csv(os.path.join(R, "..", "results", "ec_dt1_pipeline_v3.csv"), index=False, encoding="utf-8-sig")
