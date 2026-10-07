# -*- coding: utf-8 -*-
"""GX2 — 격자 배제 v2 (GX1 중간 비평 반영, 실행 전 고정) — 2026-10-07 연구실 클로드
GX1 결과: 결정 시 앞날 식별 강함(EC 경계 차 .000~.007, 우연 ~1e-7)이나 결정 6~13/46, 전체 효과로는 판정 불가.
변경:
 C1 가림 = 대상 b + 실제 평가 60 (평가와 같은 조건; LOCK 포함 학습 정답은 모두 참조). 대상 = SG2 OOF 가 있는 2차 46일.
 C2 attach: b 자신을 묶음 후보에서 제외하고 0..h(h=5 고정) 외기로 붙임 → 모든 시각 행에 같은 결정(5시 이후 적용).
 C3 결정 조건 일반화: 이웃 묶음 P 의 보이는 정답 수 = G 의 보이는 정답 수 + 1 → 남은 1개 a*. 단 b 자신의 0시 입력 검증:
    a*→b 외기+실내/4 자정 비용 ≤ Q90 (Q90 = 정답 기록 CH2 연결(EC 점프 <.05)의 외기+실내/4 비용 90% 분위, 학습 자료로 계산).
    다음날도 대칭(b 의 23시 입력 사용 → 23시 행에만 쓸 수 있으므로 다음날 쪽은 값 계산에서 제외: 앞날만 사용).
 C4 추정 두 안(k=2): A1 0시 맞춤 pred_h = sg_h − sg_0 + EC23(a*) (h ≥ 5),  A2 누적 맞춤 pred_h = sg_h − pm_h + EC23(a*) (GX1 방식).
 C5 판정 단위 = 결정 기록. 기준선 SG2 OOF 3 검증기 × 시드 3 = 9칸.
[고정 기준] 각 안: 결정 기록 시간 RMSE 가 9칸 모두 SG2 보다 낮고, DIAG10 시드평균에서 결정 기록 단위 부트스트랩 2000 P(worse) < .025/2,
  결정 기록 ≥ 15. 함께 보고: |EC23(a*) − SG2 하루평균| > .15 인 '큰 불일치' 기록에서 누가 맞나(하루 평균 기준), 무작위 대조(P 의 아무 정답 기록),
  실제 평가 60 중 결정 수와 큰 불일치 수(정답 없이).
"""
import os, sys, json
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
PV = {v: X.pivot_table(index=["farm", "day"], columns="hour", values=v) for v in W + IN}
istest = X.groupby(["farm", "day"]).is_test.first(); keys = list(PV[W[0]].index); trk = [k for k in keys if not istest[k]]
sc = {v: np.nanstd(np.diff(PV[v].loc[trk].values, axis=1)) for v in W + IN}
mu = {v: np.nanmean(PV[v].loc[trk].values) for v in W}; sd = {v: np.nanstd(PV[v].loc[trk].values) for v in W}
labs = list(EC.index); li = {k: i for i, k in enumerate(labs)}
def J(df, idx):
    M = df.loc[idx].values; A23, A22, B0, B1 = M[:, 23], M[:, 22], M[:, 0], M[:, 1]
    return (B0[None, :] - A23[:, None]) - .5 * ((A23 - A22)[:, None] + (B1 - B0)[None, :])
CST = sum(WT[v] * (J(PV[v], labs) / sc[v]) ** 2 for v in W) + sum((J(PV[v], labs) / sc[v]) ** 2 for v in IN) / 4 + (J(EC, labs) / .02) ** 2
CST = np.where(np.isfinite(CST), CST, 1e6); np.fill_diagonal(CST, 1e6)
Z = {k: np.concatenate([(PV[v].loc[k].values - mu[v]) / sd[v] for v in W]) for k in keys}   # 4×24
# 학습 기록끼리 날짜 묶음
L = np.array([Z[k] for k in labs]); Dm = np.sqrt(np.nanmean((L[:, None, :] - L[None, :, :]) ** 2, axis=2))
par = list(range(len(labs)))
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for i in range(len(labs)):
    for j in np.where(Dm[i] <= .05)[0]:
        par[fd(i)] = fd(j)
grp = {k: fd(i) for i, k in enumerate(labs)}
mem = {}
for k, g in grp.items():
    mem.setdefault(g, []).append(k)
def attach(b, h=23):
    """b 를 0..h시 외기로 학습 묶음에 붙임(h ≥ 5). 여러 묶음이면 가장 가까운 것."""
    cols = np.concatenate([np.arange(24) + 24 * i for i in range(4)]); cols = cols[(cols % 24) <= h]
    d = {g: min(np.sqrt(np.nanmean((Z[k][cols] - Z[b][cols]) ** 2)) for k in m) for g, m in mem.items()}
    g = min(d, key=d.get)
    return g if d[g] <= .05 else None
LOCK = {(s["farm"], int(s["day"])) for s in json.load(open(os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json"), encoding="utf-8"))["selected"]}


CN = [k for k in labs]
# Q90: 정답 연결(EC 점프 < .05, CST 최소 앞 기록)의 외기+실내 비용 분위
NOEC = sum(WT[v] * (J(PV[v], labs) / sc[v]) ** 2 for v in W) + sum((J(PV[v], labs) / sc[v]) ** 2 for v in IN) / 4
NOEC = np.where(np.isfinite(NOEC), NOEC, 1e6)
pre = np.argmin(CST, axis=0); ok = np.abs(J(EC, labs)[pre, np.arange(len(labs))]) < .05
Q90 = np.quantile(NOEC[pre, np.arange(len(labs))][ok], .9); print("Q90 = %.3f (연결 %d)" % (Q90, ok.sum()), flush=True)
# 평가 기록 포함 비용(b 쪽 입력만 쓰는 외기+실내) 계산 함수
def noec(a, b):
    c = 0.0
    for v in W + IN:
        A23, A22, B0, B1 = PV[v].loc[a, 23], PV[v].loc[a, 22], PV[v].loc[b, 0], PV[v].loc[b, 1]
        j = (B0 - A23) - .5 * ((A23 - A22) + (B1 - B0))
        c += (WT[v] if v in W else .25) * (j / sc[v]) ** 2
    return c
def attach5(b):
    cols = np.concatenate([np.arange(6) + 24 * i for i in range(4)])
    best, bd = None, 9.0
    for g, m in mem.items():
        for k in m:
            if k == b:
                continue
            d = np.sqrt(np.nanmean((Z[k][cols] - Z[b][cols]) ** 2))
            if d < bd:
                best, bd = g, d
    return best if bd <= .05 else None
rng = np.random.default_rng(0)
def decide(b, ref):
    g = attach5(b)
    if g is None:
        return None, None, "붙임 실패"
    G = [c for c in mem[g] if c in ref and c != b]
    if not G:
        return None, None, "G 정답 0"
    avail = [k for k in labs if k in ref]; ai = np.array([li[k] for k in avail])
    picks = [avail[int(np.argmin(CST[ai, li[c]]))] for c in G]
    Pg = pd.Series([grp[p] for p in picks]).value_counts().index[0]
    Pl = [k for k in mem[Pg] if k in ref]
    if len(Pl) != len(G) + 1:
        return None, None, "P 정답 수 ≠ G+1"
    C = np.array([[CST[li[p], li[c]] for c in G] for p in Pl])
    r_, _ = linear_sum_assignment(C)
    a = [Pl[i] for i in range(len(Pl)) if i not in set(r_)][0]
    if noec(a, b) > Q90:
        return None, Pl[int(rng.integers(len(Pl)))], "0시 검증 탈락"
    return a, Pl[int(rng.integers(len(Pl)))], "결정"
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); SGS = [23, 808, 9090]
tests = {k for k in keys if istest[k]}
out = []
for b in sorted({(f, d) for f, d in zip(S.farm, S.day)}):
    ref = {x for x in labs if x != b}
    a, pa, st = decide(b, ref)
    out.append(dict(farm=b[0], day=b[1], st=st, a=a, v=EC.loc[a, 23] if a else np.nan, pv=EC.loc[pa, 23] if pa else np.nan,
                    hit=abs(EC.loc[b, 0] - EC.loc[a, 23]) < .05 if a else np.nan, y=EC.loc[b].mean()))
O = pd.DataFrame(out); print("결정 상태:", O.st.value_counts().to_dict(), "| 적중 %.2f (n=%d)" % (O.hit.mean(), O.hit.notna().sum()))
rm = lambda x: np.sqrt(np.mean(np.square(x)))
res = {}
for anc in ("A1", "A2"):
    cells = []; seedmean = {}
    for vn in ("DIAG10", "DIAG10y", "EL1"):
        T = S[S.validator == vn].merge(O[["farm", "day", "v", "pv"]], on=["farm", "day"]).sort_values(["farm", "day", "hour"])
        T = T[T.v.notna()]
        preds = []; sgs = []
        for s in SGS:
            sg = T["sg_%d" % s]; g0 = sg.groupby([T.farm, T.day]).transform("first"); pm = sg.groupby([T.farm, T.day]).transform(lambda x: x.expanding().mean())
            base = g0 if anc == "A1" else pm
            pred = np.where(T.hour >= 5, sg - base + T.v, sg)
            cells.append(rm(pred - T.sub_ec) < rm(sg - T.sub_ec)); preds.append(pred); sgs.append(sg.values)
            print("  %s %-7s SG2 %4d | 결정 %d일 | SG2 %.4f → %.4f" % (anc, vn, s, T.groupby(["farm", "day"]).ngroups, rm(sg - T.sub_ec), rm(pred - T.sub_ec)))
        seedmean[vn] = (T.assign(p=np.mean(preds, 0), s=np.mean(sgs, 0)))
    T = seedmean["DIAG10"]
    B = T.groupby(["farm", "day"])[["p", "s", "sub_ec"]].apply(lambda g: pd.Series({"a": ((g.s - g.sub_ec) ** 2).sum(), "b": ((g.p - g.sub_ec) ** 2).sum()}))
    idx = np.random.default_rng(1).integers(0, len(B), (2000, len(B))); pw = np.mean(B.b.values[idx].sum(1) > B.a.values[idx].sum(1))
    n_dec = T.groupby(["farm", "day"]).ngroups
    passed = all(cells) and pw < .0125 and n_dec >= 15
    print("  %s: 9칸 개선 %d/9, DIAG10 P(worse) %.4f, 결정 %d → %s" % (anc, sum(cells), pw, n_dec, "통과" if passed else "불합격"))
    res[anc] = passed
# 큰 불일치 (DIAG10 SG2 시드평균 하루 평균 기준)
D = S[S.validator == "DIAG10"].assign(sg=lambda x: x[["sg_23", "sg_808", "sg_9090"]].mean(axis=1)).groupby(["farm", "day"]).agg(sgd=("sg", "mean"), y=("sub_ec", "mean")).reset_index()
Q = O.merge(D.drop(columns="y"), on=["farm", "day"])
Q = Q[Q.v.notna()]; big = Q[(Q.v - Q.sgd).abs() > .15]
print("\n큰 불일치(|v−SG2|>.15) %d건: 격자 하루오차 %.3f vs SG2 %.3f, 격자가 더 가까운 건 %d | 대조(무작위) %.3f" % (
    len(big), rm(big.v - big.y), rm(big.sgd - big.y), int(((big.v - big.y).abs() < (big.sgd - big.y).abs()).sum()), rm(big.pv - big.y)))
print(Q[["farm", "day", "y", "sgd", "v", "pv", "hit"]].round(3).to_string(index=False))
# 실제 평가 60 (모든 학습 정답 참조)
tout = []
s14 = pd.read_csv(os.path.join(env.SUBMIT, u"09회차_2026-10-06(팀)", "submission_14.csv")); s14["farm"], s14["day"] = s14.row_id.str[:3], s14.row_id.str[4:7].astype(int)
sd14 = s14.groupby(["farm", "day"]).sub_ec.mean()
for b in sorted(tests):
    a, _, st = decide(b, set(labs))
    tout.append(dict(farm=b[0], day=b[1], st=st, v=EC.loc[a, 23] if a else np.nan, sub14=sd14[b]))
TO = pd.DataFrame(tout)
print("\n실제 평가 60: 상태 %s | 결정 %d, |v − 제출14 하루평균| > .15 인 기록 %d" % (TO.st.value_counts().to_dict(), TO.v.notna().sum(), ((TO.v - TO.sub14).abs() > .15).sum()))
print(TO[TO.v.notna()].round(3).to_string(index=False))
print("\n[GX2 고정 기준] A1 %s / A2 %s" % ("통과" if res["A1"] else "불합격", "통과" if res["A2"] else "불합격"))
O.to_csv(os.path.join(R, "..", "results", "ec_gx2_val_v1.csv"), index=False, encoding="utf-8-sig"); TO.to_csv(os.path.join(R, "..", "results", "ec_gx2_test_v1.csv"), index=False, encoding="utf-8-sig")
