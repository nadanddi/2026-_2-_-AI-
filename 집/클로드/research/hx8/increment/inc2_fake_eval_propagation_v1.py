# -*- coding: utf-8 -*-
"""INC2 (2026-10-13 집 클로드, hx8/increment).  서술(채택 판정 아님; 사전 고정 아님 — 결과는 '탐색'으로만 씀).
가짜 평가 사례: 학습 고EC 에피소드(진단 연결 a1->b + 학습 전용 달력 dmt3e 로 하루 차 1 확인)에서
  앞 앵커 a(하루 평균 >= 1, 23시 정답 A23 공개) -> k 걸음 뒤 대상 t(정답 숨김, 사이 날 정답도 숨김)   [평가 F13_221(k2)·223(k3)·225(k4) 흉내]
  뒤 앵커 b(하루 평균 >= 1, 0시 정답 B0 공개)  -> k 걸음 앞 대상 t                                 [평가 F47_227(k1)…221(k4) 흉내]
대상 선택: (S1) 모델 OOF 하루 평균 >= .9 (평가처럼 '모델이 고EC 로 본' 날),  (S2) 정답 하루 평균 >= 1 (참고, 결과로 고른 표본).
추정량(대상 하루 평균):
  M   현재 모델 OOF(DIAG10 3시드 평균; DIAG10 없는 2차 기록은 EL1·P2LOO 평균)
  P   앵커 그대로(A23 또는 B0)
  D   앵커 + k 걸음 평균 이동(사례의 에피소드 밖 학습 고EC 쌍에서 계산한 mean(y_t - A23) 또는 mean(y_t - B0)) — '상수 증분' 전파
  Dm  같은 것을 중앙값으로(급락에 덜 끌림)
  I   앵커 + 날마다 입력 증분 예측(INC1 릴지 α100, 에피소드 밖 학습 고EC W 로 적합)을 앞으로 누적, 지운 날(앞 앵커 k>=2 의 첫 사이 날)은 상수 증분  [앞 방향만; 뒤 방향은 규정상 뒤 번호 입력 불가]
  B*  w·X + (1-w)·M, w 는 에피소드 밖 사례로 최소제곱(0..1 자름), k 별
교차적합 단위 = 에피소드(진단 연결 성분).  판정 지표 = 하루 수준 RMSE, M 대비 더 가까운 사례 비율, 에피소드 부트스트랩 RMSE 차 95% 구간.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
from feat4 import EC, FARM, NUM, N, TRAIN, S3
import numpy as np, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
here = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
F = pd.read_csv(os.path.join(here, "inc1_features_v1.csv"), index_col=0); FEATS = list(F.columns)
T1 = pd.read_csv(os.path.join(here, "inc1_targets_v1.csv"), index_col=0)
y, W, comp = T1.y, T1.W, T1.comp
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
tr = list(y.index)
def consec(a, b): return a in CAL.index and b in CAL.index and CAL.seg[a] == CAL.seg[b] and CAL.day[b] == CAL.day[a] + 1
links = [(rid(int(a)), rid(int(b)), bool(c)) for a, b, c in zip(TL.a1, TL.b, TL.cert)]
links = [(a, b, c) for a, b, c in links if a in y.index and b in y.index and consec(a, b)]
succ = {}; prev = {}; certL = {}
for a, b, c in links: succ.setdefault(a, []).append(b); prev.setdefault(b, []).append(a); certL[(a, b)] = c
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"]; O["rec"] = O.row_id.str[:7]
Md = O[O.validator == "DIAG10"].groupby("rec").sg.mean(); Me = O[O.validator != "DIAG10"].groupby("rec").sg.mean()
M = Md.combine_first(Me); MSRC = pd.Series({r: ("DIAG10" if r in Md.index else "EL1/P2LOO") for r in M.index})
print(f"links (consecutive, train-train) {len(links)} (cert {sum(c for *_, c in links)}); model day means {len(M)} (DIAG10 {len(Md)})")

def paths(start, k, nxt):
    out = [[start]]
    for _ in range(k):
        out = [p + [q] for p in out for q in nxt.get(p[-1], [])]
    return out
cases = []
for direction, nxt in (("fwd", succ), ("bwd", prev)):
    for a in tr:
        if y[a] < 1.0: continue
        for k in range(1, 6):
            for p in paths(a, k, nxt):
                t = p[-1]
                if t not in M.index: continue
                anc = EC[idx[a], 23] if direction == "fwd" else EC[idx[a], 0]
                cases.append(dict(dir=direction, k=k, a=a, t=t, path=p, anc=anc, yt=y[t], m=M[t], src=MSRC[t], ep=comp[a],
                                  allcert=all(certL.get((p[j], p[j + 1]) if direction == "fwd" else (p[j + 1], p[j]), False) for j in range(k))))
C = pd.DataFrame(cases)
print(f"cases: {len(C)} | fwd {int((C.dir=='fwd').sum())}, bwd {int((C.dir=='bwd').sum())} | episodes {C.ep.nunique()}")
# ---------- increment model for I (ridge alpha 100 on high-EC W, fitted outside episode)
H = [r for r in tr if y[r] >= 1.0]
def ridge_fit(recs):
    Xm = F.loc[recs, FEATS].values.astype(float); t = W[recs].values
    mu, sd = Xm.mean(0), Xm.std(0) + 1e-9; Z = (Xm - mu) / sd; Zt = np.c_[np.ones(len(Z)), Z]; Ie = np.eye(Zt.shape[1]); Ie[0, 0] = 0
    w = np.linalg.solve(Zt.T @ Zt + 100 * Ie, Zt.T @ t)
    return lambda r: float(np.r_[1, (F.loc[r, FEATS].values.astype(float) - mu) / sd] @ w)
jumps = {(a, b): EC[idx[b], 0] - EC[idx[a], 23] for a, b, _ in links}

def estimate(Cs, ep_out):
    """estimators for cases Cs using statistics from cases/records not in episode ep_out"""
    Tr = C[(C.ep != ep_out)]
    out = []
    Hin = [r for r in H if comp[r] != ep_out]
    f = ridge_fit(Hin); cW = W[Hin].mean()
    cj = np.mean([v for (a, b), v in jumps.items() if comp[a] != ep_out and y[a] >= 1])
    for _, c in Cs.iterrows():
        tk = Tr[(Tr.dir == c.dir) & (Tr.k == c.k)]
        d_mean = (tk.yt - tk.anc).mean() if len(tk) else 0.0
        d_med = (tk.yt - tk.anc).median() if len(tk) else 0.0
        e = dict(M=c.m, P=c.anc, D=c.anc + d_mean, Dm=c.anc + d_med)
        if c.dir == "fwd":
            inter = c.path[1:-1]; s = 0.0
            for j, r in enumerate(inter):
                s += (cW if j == 0 and c.k >= 2 else f(r)) + cj     # first intermediate day = 'deleted' (no inputs)
            e["I"] = c.anc + s + cj + f(c.t) / 2                     # target: midnight jump + half of its own within-day change (day mean)
            # same with constant increments only (I0) — isolates input contribution
            e["I0"] = c.anc + len(inter) * (cW + cj) + cj + cW / 2
        out.append(e)
    return pd.DataFrame(out, index=Cs.index)

E = pd.concat([estimate(C[C.ep == ep], ep) for ep in C.ep.unique()]).loc[C.index]
C = pd.concat([C, E.drop(columns="M")], axis=1)
# blends with cross-fit weight per (dir,k)
for X_ in ["P", "D", "Dm", "I", "I0"]:
    col = np.full(len(C), np.nan)
    for (d, k), g in C.groupby(["dir", "k"]):
        if X_ not in g or g[X_].isna().all(): continue
        for ep in g.ep.unique():
            trn = g[g.ep != ep]; te = g.ep == ep
            u = (trn[X_] - trn.m).values; v = (trn.yt - trn.m).values
            w = float(np.clip((u @ v) / (u @ u + 1e-12), 0, 1)) if len(trn) >= 3 else 0.0
            col[g.index[te.values]] = (g.m + w * (g[X_] - g.m))[te].values
    C["B" + X_] = col

def summarize(Cs, label):
    rows = []
    for (d, k), g in Cs.groupby(["dir", "k"]):
        r = dict(dir=d, k=k, n=len(g), eps=g.ep.nunique(), y=g.yt.mean(), M=np.sqrt(np.mean((g.m - g.yt) ** 2)))
        for X_ in ["P", "D", "Dm", "I", "I0", "BP", "BD", "BDm", "BI"]:
            if X_ in g and g[X_].notna().all():
                r[X_] = np.sqrt(np.mean((g[X_] - g.yt) ** 2)); r["win_" + X_] = np.mean(np.abs(g[X_] - g.yt) < np.abs(g.m - g.yt))
        rows.append(r)
    R = pd.DataFrame(rows)
    print(f"\n--- {label}: day-level RMSE by direction/k (M = current model OOF), win_X = share of cases closer than M")
    cols = ["dir", "k", "n", "eps", "y", "M", "P", "D", "Dm", "I", "I0", "BP", "BD", "BDm", "BI"]
    print(R[[c for c in cols if c in R]].round(3).to_string(index=False))
    print(R[["dir", "k"] + [c for c in R if c.startswith("win_")]].round(2).to_string(index=False))
    return R

S1 = C[C.m >= 0.9]; S2 = C[C.yt >= 1.0]
R1 = summarize(S1, "S1 target model-pred >= .9 (eval-like)")
R2 = summarize(S2, "S2 target truth >= 1 (outcome-selected, reference)")
summarize(S1[S1.allcert], "S1 restricted to all-certain links")
# episode bootstrap of RMSE difference (BD - M, D - M) for the eval-relevant cells: fwd k2..4, bwd k1..4
rng = np.random.default_rng(5)
print("\n--- episode bootstrap 95% CI of RMSE(X) - RMSE(M), S1")
for d, ks in (("fwd", [2, 3, 4]), ("bwd", [1, 2, 3, 4])):
    g = S1[(S1.dir == d) & S1.k.isin(ks)]
    eps = g.ep.unique()
    for X_ in ["P", "D", "Dm", "BD", "BP"] + (["I", "BI"] if d == "fwd" else []):
        base = np.sqrt(np.mean((g[X_] - g.yt) ** 2)) - np.sqrt(np.mean((g.m - g.yt) ** 2))
        bs = []
        for _ in range(2000):
            pick = rng.choice(eps, len(eps)); gg = pd.concat([g[g.ep == e] for e in pick])
            bs.append(np.sqrt(np.mean((gg[X_] - gg.yt) ** 2)) - np.sqrt(np.mean((gg.m - gg.yt) ** 2)))
        print(f"   {d} k{ks} {X_:3s}: n {len(g)}, episodes {len(eps)}, diff {base:+.3f}  CI [{np.quantile(bs,.025):+.3f}, {np.quantile(bs,.975):+.3f}]  win {np.mean(np.abs(g[X_]-g.yt) < np.abs(g.m-g.yt)):.2f}")
# case listing for fwd k2..4 / bwd k1..2 (S1)
print("\n--- S1 cases (fwd k<=4, bwd k<=4): anchor, target, truth, model, P, D, I")
L = S1[((S1.dir == "fwd") & (S1.k <= 4)) | ((S1.dir == "bwd") & (S1.k <= 4))]
print(L[["dir", "k", "a", "t", "anc", "yt", "m", "src", "P", "D", "I", "BD"]].round(3).to_string(index=False))
C.drop(columns="path").to_csv(os.path.join(here, "inc2_cases_v1.csv"), index=False)
# ---------- empirical k-step drift table (all training, used for eval preview in INC3)
print("\n--- k-step drift on all high-anchor cases (no selection on target): mean / median (y_t - anchor), sd")
for (d, k), g in C.groupby(["dir", "k"]):
    dd = g.yt - g.anc
    print(f"   {d} k{k}: n {len(g)}, eps {g.ep.nunique()}, mean {dd.mean():+.3f}, median {dd.median():+.3f}, sd {dd.std():.3f}, share |d|>.3 {np.mean(np.abs(dd)>.3):.2f}")
