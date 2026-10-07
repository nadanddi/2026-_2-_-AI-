# -*- coding: utf-8 -*-
"""GX3 — 격자 배제법 + 안전장치 (계획 v1 + 계획 단계 비평 8개 반영, 실행 전 고정) — 2026-10-07 연구실 클로드
계획: 연구실/클로드/문서/GX3_격자배제_안전장치_계획_v1.md. 비평 반영:
 F1 NaN: S1/S2 계산에 NaN 이 끼면 결정하지 않음.
 F2 유일성: 같은 ID 에서 b 보다 앞번호인 정답 없는 기록이 이미 같은 a* 를 받았으면 결정하지 않음.
 F3 S2 문턱 = 학습 진짜 연결에서 '진짜 앞날 vs 같은 묶음 다른 기록'의 0시 비용 차(2위 − 진짜) 분포 10% 분위(최소 0). 학습 자료로 계산.
 F4 M3: b + 평가 60 + b 의 날짜 묶음 G 에 붙는 다른 ID 기록과 그 같은 ID ±2 기록. 오염률(진단: G∪P 안의 정답 없는 기록 존재 비율)을
    M1/M2/M3/실제 평가에서 보고. M3 오염률 < 실제 평가 오염률이면 '판정 불가'(불충족).
 F5 독립 기준: 2차 46일 밖 학습 정답 기록(1차 등)에서 M1 방식 결정 시 앞날 적중(|EC0(b) − EC23(a*)| < .05) Wilson 95% 하한 ≥ .90.
    무작위 대조(P 정답 기록 중 임의) 적중률 함께.
 F6 두 건 빼기: M1 에서 이득 상위 2 기록을 빼도 9칸 개선 & DIAG10 P(worse) < .00625.
 F7 규정: 제출 전 운영위 7절 문의 필수('남은 1개' 추론이 학습/평가 구성에 기대므로 3절 경계).
 F8 M2 오프셋·길이: 같은 ID 연속 L 기록 가림, (L=5, 오프셋 0..4), (L=10, 오프셋 0, 4, 9) 모두 모아 집계.
결정 규칙: GX2 와 같은 CST(EC 자정 점프/.02)² + 외기 날짜 + 실내/4, attach = 0..5시 외기 z-RMSE ≤ .05 로 '보이는 정답 기록' 묶음에 붙임(b 제외).
 S1 G 의 보이는 정답 기록마다 최소 비용 앞 기록(pick)이 모두 같은 묶음 P, pick 비용·헝가리안 짝 비용 모두 ≤ q95(학습 진짜 연결 CST, EC 항 포함).
    |P 보이는 정답| = |G 보이는 정답| + 1 → 남은 1개 a*.
 S2 a* 가 P 정답 기록 중 b 의 0시 비용(외기+실내/4, EC 없음) 최소이고 2위와 차 ≥ F3 문턱.
 S3 같은 ID 앞번호 '정답 없는'(평가·가림) 기록이 0..5시 외기로 G 또는 P 에 붙으면 결정 안 함(다른 ID 평가 입력은 보지 않음).
추정 A2(사후 선택 명시): h ≥ 5 행 pred_h = sg_h − pm_h + EC23(a*), 그 밖 sg_h.
[고정 기준] C1 M1 결정 ≥ 15 (46일 중) / C2 M1·M2·M3 각 결정 기록에서 9칸(검증기 3 × SG2 시드 3) 모두 SG2 보다 낮음(M2·M3 결정 < 10 이면 불충족)
 / C3 M1 DIAG10 결정 기록 부트스트랩 2000 P(worse) < .00625 (누적 본페로니 4) / C4 F6 / C5 F5 / C6 M3 오염률 ≥ 실제 평가 오염률.
"""
import os, sys, json, math
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
tests = [k for k in keys if istest[k]]
sc = {v: np.nanstd(np.diff(PV[v].loc[trk].values, axis=1)) for v in W + IN}
mu = {v: np.nanmean(PV[v].loc[trk].values) for v in W}; sd = {v: np.nanstd(PV[v].loc[trk].values) for v in W}
labs = list(EC.index); li = {k: i for i, k in enumerate(labs)}; ki = {k: i for i, k in enumerate(keys)}
def J(df, idx):
    M = df.loc[idx].values; A23, A22, B0, B1 = M[:, 23], M[:, 22], M[:, 0], M[:, 1]
    return (B0[None, :] - A23[:, None]) - .5 * ((A23 - A22)[:, None] + (B1 - B0)[None, :])
CST = sum(WT[v] * (J(PV[v], labs) / sc[v]) ** 2 for v in W) + sum((J(PV[v], labs) / sc[v]) ** 2 for v in IN) / 4 + (J(EC, labs) / .02) ** 2
CST = np.where(np.isfinite(CST), CST, np.nan); np.fill_diagonal(CST, np.nan)
NOE = sum(WT[v] * (J(PV[v], keys) / sc[v]) ** 2 for v in W) + sum((J(PV[v], keys) / sc[v]) ** 2 for v in IN) / 4   # 모든 기록 쌍, EC 없음
Z = np.vstack([np.concatenate([(PV[v].loc[k].values - mu[v]) / sd[v] for v in W]) for k in keys])
FULL = Z; E6 = Z[:, np.concatenate([np.arange(6) + 24 * i for i in range(4)])]
# 학습 기록 날짜 묶음(전체 외기)
L = FULL[[ki[k] for k in labs]]; Dm = np.sqrt(np.nanmean((L[:, None, :] - L[None, :, :]) ** 2, axis=2))
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
LE6 = E6[[ki[k] for k in labs]]
# q95 / S2 문턱 (학습 진짜 연결)
pre = np.nanargmin(np.where(np.isnan(CST), np.inf, CST), axis=0); jec = J(EC, labs)
okl = [(pre[j], j) for j in range(len(labs)) if np.isfinite(CST[pre[j], j]) and abs(jec[pre[j], j]) < .05]
Q95 = np.quantile([CST[i, j] for i, j in okl], .95)
marg = []
for i, j in okl:
    a, b = labs[i], labs[j]; oth = [o for o in mem[grp[a]] if o != a]
    if oth:
        c0 = NOE[ki[a], ki[b]]; co = [NOE[ki[o], ki[b]] for o in oth]
        if np.isfinite(c0) and np.all(np.isfinite(co)):
            marg.append(min(co) - c0)
THR = max(0.0, float(np.quantile(marg, .10)))
print("q95 %.3f (진짜 연결 %d) | S2 문턱 %.3f (차 분포 n=%d, 중앙 %.2f)" % (Q95, len(okl), THR, len(marg), np.median(marg)), flush=True)

def attach(b, refset):
    """b 의 0..5시 외기로 '보이는 정답 기록' 묶음에 붙임."""
    d = np.sqrt(np.nanmean((LE6 - E6[ki[b]]) ** 2, axis=1))
    m = np.array([k in refset and k != b for k in labs]); d = np.where(m, d, np.inf)
    j = int(np.argmin(d))
    return grp[labs[j]] if d[j] <= .05 else None

def decide(b, refset, unl_same_earlier, taken):
    """반환 (a*, 무작위대조, 상태)."""
    g = attach(b, refset)
    if g is None:
        return None, None, "붙임 실패"
    G = [c for c in mem[g] if c in refset and c != b]
    if not G:
        return None, None, "G 정답 0"
    ai = np.array([li[k] for k in labs if k in refset])
    picks, pc = [], []
    for c in G:
        col = CST[ai, li[c]]
        if np.all(np.isnan(col)):
            return None, None, "S1 NaN"
        j = int(np.nanargmin(col)); picks.append(labs[ai[j]]); pc.append(col[j])
    if len({grp[p] for p in picks}) != 1 or max(pc) > Q95:
        return None, None, "S1 pick"
    Pg = grp[picks[0]]
    Pl = [k for k in mem[Pg] if k in refset]
    if len(Pl) != len(G) + 1:
        return None, None, "P 수 ≠ G+1"
    C = np.array([[CST[li[p], li[c]] for c in G] for p in Pl])
    if np.isnan(C).any():
        return None, None, "S1 NaN"
    r_, c_ = linear_sum_assignment(C)
    if C[r_, c_].max() > Q95:
        return None, None, "S1 짝"
    a = [Pl[i] for i in range(len(Pl)) if i not in set(r_)][0]
    rnd = Pl[int(np.random.default_rng(ki[b]).integers(len(Pl)))]
    nc = np.array([NOE[ki[p], ki[b]] for p in Pl])
    if not np.all(np.isfinite(nc)):
        return None, rnd, "S2 NaN"
    o = np.argsort(nc)
    if Pl[o[0]] != a or (nc[o[1]] - nc[o[0]]) < THR:
        return None, rnd, "S2"
    for u in unl_same_earlier:
        gu = attach(u, refset)
        if gu is not None and gu in (g, Pg):
            return None, rnd, "S3"
    if a in taken:
        return None, rnd, "유일성"
    return a, rnd, "결정"

def run_mode(targets, hidden_fn):
    """targets: 대상 b 목록. hidden_fn(b) → 가림 집합(정답 없음). 반환 dict b → (a*, rnd, 상태, 오염)."""
    out = {}
    for b in targets:
        hid = hidden_fn(b) | {b} | set(tests)
        refset = set(k for k in labs if k not in hid)
        same = sorted([u for u in hid if u[0] == b[0] and u[1] < b[1]], key=lambda x: x[1])
        taken = set()
        for u in same:   # 앞번호 정답 없는 기록의 결정(유일성용)
            au, _, su = decide(u, refset, [w for w in same if w[1] < u[1]], set())
            if au is not None:
                taken.add(au)
        a, rnd, st = decide(b, refset, same, taken)
        # 오염(진단): b 의 진짜 묶음/앞날 묶음에 정답 없는 기록 존재
        g = attach(b, refset); cont = np.nan
        if g is not None and a is not None:
            allg = [k for k in keys if np.sqrt(np.nanmean((FULL[ki[k]] - FULL[ki[mem[g][0]]]) ** 2)) <= .05]
            allp = [k for k in keys if np.sqrt(np.nanmean((FULL[ki[k]] - FULL[ki[a]]) ** 2)) <= .05]
            cont = any((k in hid and k != b) for k in allg + allp)
        out[b] = (a, rnd, st, cont)
    return out

S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); SGS = [23, 808, 9090]
T46 = sorted({(f, d) for f, d in zip(S.farm, S.day)})
def block(b, Lb, off):
    ds = sorted(d for f, d in keys if f == b[0]); i = ds.index(b[1]); s = i - off
    return {(b[0], ds[j]) for j in range(max(0, s), min(len(ds), s + Lb))}
def m3_hide(b):
    g = attach(b, set(k for k in labs if k != b))
    h = set()
    if g is not None:
        for k in keys:
            if k[0] != b[0] and np.sqrt(np.nanmean((FULL[ki[k]] - FULL[ki[mem[g][0]]]) ** 2)) <= .05:
                ds = sorted(d for f, d in keys if f == k[0]); i = ds.index(k[1])
                h |= {(k[0], ds[j]) for j in range(max(0, i - 2), min(len(ds), i + 3))}
    return h
modes = {"M1": [(b, b) for b in T46]}
print("M1 실행", flush=True)
RES = {"M1": run_mode(T46, lambda b: set())}
RES["M2"] = {}
for Lb, offs in ((5, range(5)), (10, (0, 4, 9))):
    for off in offs:
        r = run_mode(T46, lambda b, Lb=Lb, off=off: block(b, Lb, off))
        for b, v in r.items():
            RES["M2"][(b, Lb, off)] = v
print("M2 실행 완료", flush=True)
RES["M3"] = run_mode(T46, m3_hide); print("M3 실행 완료", flush=True)
TEST = run_mode(sorted(tests), lambda b: set())
rm = lambda x: np.sqrt(np.mean(np.square(x)))

def evaluate(name, dec):
    """dec: list of (b, a*) 결정. 9칸 비교 + DIAG10 부트스트랩."""
    rows = pd.DataFrame([dict(farm=b[0], day=b[1], v=EC.loc[a, 23], key=i) for i, (b, a) in enumerate(dec)])
    cells = []; d10 = None
    for vn in ("DIAG10", "DIAG10y", "EL1"):
        T = S[S.validator == vn].merge(rows, on=["farm", "day"]).sort_values(["key", "hour"])
        ps, ss = [], []
        for s in SGS:
            sg = T["sg_%d" % s]; pm = sg.groupby(T.key).transform(lambda x: x.expanding().mean())
            pred = np.where(T.hour >= 5, sg - pm + T.v, sg); cells.append(rm(pred - T.sub_ec) < rm(sg - T.sub_ec)); ps.append(pred); ss.append(sg.values)
        if vn == "DIAG10":
            d10 = T.assign(p=np.mean(ps, 0), s=np.mean(ss, 0))
    B = d10.groupby("key")[["p", "s", "sub_ec"]].apply(lambda g: pd.Series({"a": ((g.s - g.sub_ec) ** 2).sum(), "b": ((g.p - g.sub_ec) ** 2).sum()}))
    idx = np.random.default_rng(1).integers(0, len(B), (2000, len(B))); pw = float(np.mean(B.b.values[idx].sum(1) > B.a.values[idx].sum(1)))
    gain = (B.a - B.b).sort_values(ascending=False)
    print("  %s: 결정 %d | 9칸 개선 %d/9 | DIAG10 결정기록 RMSE %.4f → %.4f | P(worse) %.4f" % (
        name, len(B), sum(cells), np.sqrt(B.a.sum() / len(d10)), np.sqrt(B.b.sum() / len(d10)), pw))
    return all(cells), pw, list(gain.index[:2]), rows

verdict = {}
d1 = [(b, v[0]) for b, v in RES["M1"].items() if v[0] is not None]
print("\n=== M1 상태:", pd.Series([v[2] for v in RES["M1"].values()]).value_counts().to_dict())
ok1, p1, top2, rows1 = evaluate("M1", d1)
verdict["C1"] = len(d1) >= 15; verdict["C3"] = p1 < .00625
d1b = [x for i, x in enumerate(d1) if i not in top2]
ok1b, p1b, _, _ = evaluate("M1 두 건 빼기", d1b) if len(d1b) >= 3 else (False, 1.0, None, None)
verdict["C4"] = ok1b and p1b < .00625
d2 = [(k[0], v[0]) for k, v in RES["M2"].items() if v[0] is not None]
print("=== M2 상태(모든 오프셋 합):", pd.Series([v[2] for v in RES["M2"].values()]).value_counts().to_dict())
ok2 = False
if len(d2) >= 10:
    ok2, _, _, _ = evaluate("M2", d2)
d3 = [(b, v[0]) for b, v in RES["M3"].items() if v[0] is not None]
print("=== M3 상태:", pd.Series([v[2] for v in RES["M3"].values()]).value_counts().to_dict())
ok3 = False
if len(d3) >= 10:
    ok3, _, _, _ = evaluate("M3", d3)
verdict["C2"] = ok1 and ok2 and ok3
for nm, res in (("M1", RES["M1"]), ("M2", RES["M2"]), ("M3", RES["M3"]), ("평가", TEST)):
    c = [v[3] for v in res.values() if v[0] is not None and v[3] == v[3]]
    print("  오염률 %s: %d/%d = %.2f" % (nm, sum(c), len(c), np.mean(c) if c else np.nan))
cm3 = np.mean([v[3] for v in RES["M3"].values() if v[0] is not None and v[3] == v[3]] or [np.nan])
cte = np.mean([v[3] for v in TEST.values() if v[0] is not None and v[3] == v[3]] or [np.nan])
verdict["C6"] = bool(cm3 >= cte) if np.isfinite(cm3) and np.isfinite(cte) else False
# 적중(46일) 과 독립 기준(46일 밖)
for nm, res in (("M1", RES["M1"]), ("M3", RES["M3"])):
    h = [abs(EC.loc[b, 0] - EC.loc[v[0], 23]) < .05 for b, v in res.items() if v[0] is not None]
    print("  적중 %s: %d/%d" % (nm, sum(h), len(h)))
outside = [k for k in labs if k not in set(T46)]
IND = run_mode(outside, lambda b: set())
h = [abs(EC.loc[b, 0] - EC.loc[v[0], 23]) < .05 for b, v in IND.items() if v[0] is not None]
hr = [abs(EC.loc[b, 0] - EC.loc[v[1], 23]) < .05 for b, v in IND.items() if v[0] is not None and v[1] is not None]
n, k = len(h), sum(h); z = 1.96
wl = ((k / n + z * z / (2 * n)) - z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n))) / (1 + z * z / n) if n else 0
print("  독립(46일 밖 %d 기록): 결정 %d, 적중 %d → Wilson 하한 %.3f | 무작위 대조 적중 %.2f" % (len(outside), n, k, wl, np.mean(hr) if hr else np.nan))
verdict["C5"] = wl >= .90
s14 = pd.read_csv(os.path.join(env.SUBMIT, u"09회차_2026-10-06(팀)", "submission_14.csv")); s14["farm"], s14["day"] = s14.row_id.str[:3], s14.row_id.str[4:7].astype(int)
sd14 = s14.groupby(["farm", "day"]).sub_ec.mean()
TT = pd.DataFrame([dict(farm=b[0], day=b[1], st=v[2], v=EC.loc[v[0], 23] if v[0] else np.nan, sub14=sd14[b]) for b, v in TEST.items()])
print("\n실제 평가 60: %s | 결정 %d, 큰 불일치(|v−제출14|>.15) %d" % (TT.st.value_counts().to_dict(), TT.v.notna().sum(), ((TT.v - TT.sub14).abs() > .15).sum()))
print(TT[TT.v.notna()].round(3).to_string(index=False))
print("\n판정:", verdict, "→ [GX3 고정 기준] %s" % ("통과" if all(verdict.values()) else "불합격"))
TT.to_csv(os.path.join(R, "..", "results", "ec_gx3_test_v1.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame([dict(mode=m, farm=(b[0][0] if m == "M2" else b[0]), day=(b[0][1] if m == "M2" else b[1]), a=str(v[0]), st=v[2], cont=v[3])
              for m in ("M1", "M2", "M3") for b, v in RES[m].items()]).to_csv(os.path.join(R, "..", "results", "ec_gx3_val_v1.csv"), index=False, encoding="utf-8-sig")
