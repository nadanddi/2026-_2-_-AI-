# -*- coding: utf-8 -*-
"""GX1 — 격자 배제법, 평가형 동시 가림 검증 (계획 v1 + 계획 비평 반영, 실행 전 고정) — 2026-10-07 연구실 클로드
계획: 연구실/클로드/문서/GX1_격자배제_계획_v1.md. 비평 반영(실행 전):
 R1 결정 조건: b 의 날짜 묶음 G 에서 '보이는 정답 기록'이 정확히 3개(4자리 = 3 정답 + b)일 때만. 그 밖은 결정 안 함.
    날짜 묶음은 학습 기록끼리 외기 쌍둥이 연결 성분으로 먼저 만들고, b 는 0..h시 외기(h ≥ 5)로 그 성분에 붙인다(평가 기록은 다리로 쓰지 않음).
 R2 같은 이웃(앞날 또는 다음날)이 한 폴드·한 시점에서 두 대상에 배정되면 둘 다 그 쪽은 결정 안 함.
 R3 폴드 가림 = 그 폴드 기록 전체(1차 포함) + 실제 평가 60 + 잠금 40일(SG2와 같게 참조 제외).
 R4 판정 p 는 DIAG10 (프로젝트 규칙). DIAG10y·EL1 은 칸 방향 조건에 포함.
 R5 대조: 전날 묶음 P 의 정답 기록 중 무작위 1개(시드 0)로 같은 규칙.
 R6 보호 변형 없음(단일 변형).
절차(대상 = 폴드의 2차 기록 b):
 전날 묶음 P = G 의 다른 정답 기록 c 마다 참조 정답 기록 중 CH2 비용(EC 자정 점프/.02)² + 외기 날짜 + 실내/4 최소인 앞 기록 → 그 묶음 최빈.
 배제 = P 의 참조 정답 기록 ↔ G 정답 기록 헝가리안, 짝 없는 정답 기록이 정확히 1개면 앞날 a*. 다음날 N 도 대칭(n*).
 v = 평균(EC23(a*), EC0(n*)) 있는 것. 예측: h ≥ 5 행에서 pred_h = sg_h − pm_h + v (pm_h = SG2 0..h 누적평균), 그 밖 sg_h.
[고정 기준] 9칸(검증기 3 × SG2 시드 3) 모두 2차 시간 RMSE < SG2, 시드평균 두 온실 모두 개선, 일반 날·고EC 날 각각 악화 ≤ 1%,
  DIAG10 온실×5기록 묶음 부트스트랩 2000 P(worse) < .025. 함께: 결정 수, 비순환 적중(|EC0(b) − EC23(a*)| < .05), 대조, 고EC 상위 날 제외 민감도.
사전 관문: 실제 평가 60 중 R1 로 결정 가능한 기록(정답 기록 3개 묶음, 이웃 결정 여부 무관) 수 < 10 이면 종료.
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

# ---- 사전 관문: 실제 평가 60
gate = 0
for b in keys:
    if istest[b]:
        g = attach(b)
        if g is not None and len(mem[g]) == 3:
            gate += 1
print("사전 관문: 평가 60 중 '정답 기록 3개 묶음'에 붙는 기록 %d (기준 ≥ 10)" % gate, flush=True)
if gate < 10:
    print("종료"); sys.exit(0)

rng = np.random.default_rng(0)
def decide(b, ref, direction, placebo=False):
    g = attach(b)
    if g is None:
        return None
    G = [c for c in mem[g] if c in ref]
    if len(G) != 3:
        return None
    avail = [k for k in labs if k in ref]; ai = np.array([li[k] for k in avail])
    picks = [avail[int(np.argmin(CST[ai, li[c]] if direction < 0 else CST[li[c], ai]))] for c in G]
    Pg = pd.Series([grp[p] for p in picks]).value_counts().index[0]
    Pl = [k for k in mem[Pg] if k in ref]
    if len(Pl) < 2:
        return None
    C = np.array([[CST[li[p], li[c]] if direction < 0 else CST[li[c], li[p]] for c in G] for p in Pl])
    r_, _ = linear_sum_assignment(C)
    left = [Pl[i] for i in range(len(Pl)) if i not in set(r_)]
    if placebo:
        return Pl[int(rng.integers(len(Pl)))]
    return left[0] if len(left) == 1 else None

S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); SGS = [23, 808, 9090]
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); f10 = DP[DP.validator == "DIAG10"].groupby(["farm", "day"]).validation_fold.first()
FOLD = {"DIAG10": f10.to_dict(), "DIAG10y": {k: ((k[1] + 2) // 6) % 10 for k in labs if k not in LOCK}}
el = S[S.validator == "EL1"].groupby(["farm", "day"]).validation_fold.first()
FOLD["EL1"] = el.to_dict()
V = {}
for vname in ("DIAG10", "DIAG10y", "EL1"):
    tgt = S[S.validator == vname].groupby(["farm", "day"]).validation_fold.first()
    for k in sorted(tgt.unique()):
        hid = {key for key, kk in FOLD[vname].items() if kk == k} | {key for key, kk in tgt.items() if kk == k}
        ref = {x for x in labs if x not in hid and x not in LOCK}
        dec = {}
        for b in [key for key, kk in tgt.items() if kk == k]:
            dec[b] = (decide(b, ref, -1), decide(b, ref, +1), decide(b, ref, -1, True), decide(b, ref, +1, True))
        for side in (0, 1):   # R2 중복 배정 제거
            cnt = pd.Series([d[side] for d in dec.values() if d[side] is not None]).value_counts()
            for b in dec:
                if dec[b][side] is not None and cnt[dec[b][side]] > 1:
                    dec[b] = tuple(None if i == side else x for i, x in enumerate(dec[b]))
        for b, (a, n, pa, pn) in dec.items():
            vv = [EC.loc[a, 23]] if a else []; vv += [EC.loc[n, 0]] if n else []
            pv = [EC.loc[pa, 23]] if pa else []; pv += [EC.loc[pn, 0]] if pn else []
            V[(vname, b)] = dict(v=np.mean(vv) if vv else np.nan, pv=np.mean(pv) if (pv and vv) else np.nan,
                                 hit=(abs(EC.loc[b, 0] - EC.loc[a, 23]) < .05) if a else np.nan)
    print("  %s 완료" % vname, flush=True)
rm = lambda x: np.sqrt(np.mean(np.square(x)))
verdict = []
for vname in ("DIAG10", "DIAG10y", "EL1"):
    T = S[S.validator == vname].sort_values(["farm", "day", "hour"]).copy()
    T["v"] = [V[(vname, (f, d))]["v"] for f, d in zip(T.farm, T.day)]; T["pv"] = [V[(vname, (f, d))]["pv"] for f, d in zip(T.farm, T.day)]
    T["dy"] = T.groupby(["farm", "day"]).sub_ec.transform("mean")
    vd = pd.DataFrame([dict(b=b, **x) for (vn, b), x in V.items() if vn == vname])
    print("\n=== %s: 결정 %d/46일, 비순환 적중(앞날) %.2f (n=%d) ===" % (vname, vd.v.notna().sum(), vd.hit.mean(), vd.hit.notna().sum()))
    cells = []; P = []; PL = []
    for s in SGS:
        sg = T["sg_%d" % s]; pm = sg.groupby([T.farm, T.day]).transform(lambda x: x.expanding().mean())
        use = T.v.notna() & (T.hour >= 5)
        pred = np.where(use, sg - pm + T.v, sg); plc = np.where(use & T.pv.notna(), sg - pm + T.pv, sg)
        a, b2 = rm(sg - T.sub_ec), rm(pred - T.sub_ec)
        print("  SG2 %4d: %.4f → %.4f (%+.1f%%) | 대조 %.4f" % (s, a, b2, 100 * (b2 / a - 1), rm(plc - T.sub_ec)))
        cells.append(b2 < a); P.append(pred); PL.append(sg)
    T["pred"] = np.mean(P, axis=0); T["sg"] = np.mean(PL, axis=0)
    ok = all(cells)
    for nm, m in (("일반", T.dy < 1), ("고EC", T.dy >= 1), ("F13", T.farm == "F13"), ("F47", T.farm == "F47")):
        a, b2 = rm(T.sg[m] - T.sub_ec[m]), rm(T.pred[m] - T.sub_ec[m]); ch = 100 * (b2 / a - 1)
        print("  시드평균 %-4s %.4f → %.4f (%+.1f%%)" % (nm, a, b2, ch))
        if nm in ("일반", "고EC") and ch > 1: ok = False
        if nm in ("F13", "F47") and ch >= 0: ok = False
    top = T.groupby(["farm", "day"]).apply(lambda g: ((g.sg - g.sub_ec) ** 2).sum() - ((g.pred - g.sub_ec) ** 2).sum()).sort_values(ascending=False)
    ex = ~T.set_index(["farm", "day"]).index.isin(top.index[:3])
    print("  민감도: 이득 상위 3일 %s 제외 시 %.4f → %.4f" % ([f"{f}_{d}" for f, d in top.index[:3]], rm(T.sg[ex] - T.sub_ec[ex]), rm(T.pred[ex] - T.sub_ec[ex])))
    if vname == "DIAG10":
        T["blk"] = T.farm + "_" + (T.day // 5).astype(str)
        B = T.groupby("blk")[["sg", "pred", "sub_ec"]].apply(lambda g: pd.Series({"a": ((g.sg - g.sub_ec) ** 2).sum(), "b": ((g.pred - g.sub_ec) ** 2).sum()}))
        idx = np.random.default_rng(1).integers(0, len(B), (2000, len(B)))
        pw = np.mean(B.b.values[idx].sum(1) > B.a.values[idx].sum(1)); print("  DIAG10 P(worse) %.4f" % pw)
        ok = ok and pw < .025
    verdict.append(ok); print("  → %s" % ("충족" if ok else "미충족"))
print("\n[GX1 고정 기준] %s" % ("통과" if all(verdict) else "불합격"))
pd.DataFrame([dict(V=vn, farm=b[0], day=b[1], **x) for (vn, b), x in V.items()]).to_csv(os.path.join(R, "..", "results", "ec_gx1_decisions_v1.csv"), index=False, encoding="utf-8-sig")
