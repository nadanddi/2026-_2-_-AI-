# -*- coding: utf-8 -*-
"""INC3 (2026-10-13 집 클로드, hx8/increment).  서술: INC2 결과의 견고성 + 평가 8기록 미리보기(제출물 아님).
(a) 시드별 모델(M) 로 다시 계산 — 방향이 3시드 모두 같은가 (뒤 방향 k1..4, 앞 방향 k2..4; 추정량 P, D, Dm, BD).
(b) 에피소드 하나씩 빼고 RMSE 차 방향 유지 여부.
(c) k 별 에피소드 부트스트랩 CI (뒤 k1·k2·k3·k4 따로; F47_227·225·223·221 에 대응).
(d) M 의 치우침(평균 y - M) 과 D 의 이득이 '치우침 고치기'인지 '사례별 정보'인지: M 에 사례 밖 평균 치우침을 더한 M+bias 와 비교.
(e) 앞 방향 k1 입력 증분(I)이 왜 좋아 보였나 — 사례별 기여, 에피소드 E(F13_172~178) 빼면?
(f) 보통 앵커(하루 평균 < 1) -> 모델이 고EC 로 본 대상(k=2) 사례 (F13_237 ← F13_233 .634 흉내).
(g) 평가 미리보기: F47_229 0시(1.716) 뒤 방향 k1..4 → F47_227·225·223·221, F13_217 23시(1.663) 앞 방향 k2..4 → F13_221·223·225,
    F13_233 23시(.643) 앞 방향 k2 → F13_237. 계수는 학습 가짜 사례 전부에서(교차적합 아님, 미리보기). 13회차 하루 평균과 비교.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
from feat4 import EC, FARM, NUM, N, TRAIN, S3
import numpy as np, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
here = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
C = pd.read_csv(os.path.join(here, "inc2_cases_v1.csv"))
T1 = pd.read_csv(os.path.join(here, "inc1_targets_v1.csv"), index_col=0); y, comp = T1.y, T1.comp
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
Ms = O.groupby(["seed", "rec"]).sg.mean().unstack(0)
seeds = list(Ms.columns)
S1 = C[C.m >= 0.9].copy()
rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def drift_est(Cs, Call, col_m):
    """D (mean drift) and BD (blend with model column col_m) cross-fit by episode, within (dir,k)."""
    D = np.full(len(Cs), np.nan); BD = np.full(len(Cs), np.nan)
    for j, (ix, c) in enumerate(Cs.iterrows()):
        tk = Call[(Call.dir == c.dir) & (Call.k == c.k) & (Call.ep != c.ep)]
        D[j] = c.anc + (tk.yt - tk.anc).mean()
        trn = Cs[(Cs.dir == c.dir) & (Cs.k == c.k) & (Cs.ep != c.ep)]
        if len(trn) >= 3:
            dtr = np.array([r.anc + (Call[(Call.dir == r.dir) & (Call.k == r.k) & (Call.ep != c.ep) & (Call.ep != r.ep)].yt -
                                    Call[(Call.dir == r.dir) & (Call.k == r.k) & (Call.ep != c.ep) & (Call.ep != r.ep)].anc).mean() for _, r in trn.iterrows()])
            u = dtr - trn[col_m].values; v = trn.yt.values - trn[col_m].values
            w = float(np.clip(u @ v / (u @ u + 1e-12), 0, 1))
        else:
            w = 0.0
        BD[j] = c[col_m] + w * (D[j] - c[col_m])
    return D, BD

print("(a) per-seed model: RMSE(X) - RMSE(M_seed), S1, by direction/k-group")
for s in seeds:
    S1[f"m{s}"] = [Ms[s].get(t, np.nan) for t in S1.t]
for d, ks in (("bwd", [1]), ("bwd", [2]), ("bwd", [3]), ("bwd", [4]), ("bwd", [1, 2, 3, 4]), ("fwd", [2]), ("fwd", [3]), ("fwd", [4]), ("fwd", [2, 3, 4])):
    g = S1[(S1.dir == d) & S1.k.isin(ks)]
    line = []
    for s in seeds:
        D, BD = drift_est(g, C, f"m{s}")
        line.append(f"seed{s}: M {rm(g[f'm{s}'], g.yt):.3f} P {rm(g.anc, g.yt) - rm(g[f'm{s}'], g.yt):+.3f} D {rm(D, g.yt) - rm(g[f'm{s}'], g.yt):+.3f} BD {rm(BD, g.yt) - rm(g[f'm{s}'], g.yt):+.3f}")
    print(f"   {d} k{ks} n {len(g)} eps {g.ep.nunique()} | " + " | ".join(line))

print("\n(b) leave-one-episode-out: RMSE(D) - RMSE(M) and RMSE(BD) - RMSE(M) (seed-mean M), S1")
for d, ks in (("bwd", [1, 2, 3, 4]), ("bwd", [3, 4]), ("fwd", [2, 3, 4])):
    g = S1[(S1.dir == d) & S1.k.isin(ks)]
    out = []
    for ep in g.ep.unique():
        gg = g[g.ep != ep]; D, BD = drift_est(gg, C[C.ep != ep], "m")
        out.append((ep, len(g) - len(gg), rm(D, gg.yt) - rm(gg.m, gg.yt), rm(BD, gg.yt) - rm(gg.m, gg.yt)))
    print(f"   {d} k{ks}: " + "; ".join(f"-{e}({n}) D {a:+.3f} BD {b:+.3f}" for e, n, a, b in out))

print("\n(c) per-k episode bootstrap CI of RMSE(X)-RMSE(M), S1 (X = D, Dm, BD from INC2)")
rng = np.random.default_rng(9)
for d, k in [("bwd", 1), ("bwd", 2), ("bwd", 3), ("bwd", 4), ("fwd", 2), ("fwd", 3), ("fwd", 4)]:
    g = S1[(S1.dir == d) & (S1.k == k)]; eps = g.ep.unique(); line = []
    for X_ in ["P", "D", "BD"]:
        base = rm(g[X_], g.yt) - rm(g.m, g.yt); bs = []
        for _ in range(2000):
            gg = pd.concat([g[g.ep == e] for e in rng.choice(eps, len(eps))]); bs.append(rm(gg[X_], gg.yt) - rm(gg.m, gg.yt))
        line.append(f"{X_} {base:+.3f} [{np.quantile(bs,.025):+.3f},{np.quantile(bs,.975):+.3f}] win {np.mean(np.abs(g[X_]-g.yt)<np.abs(g.m-g.yt)):.2f}")
    print(f"   {d} k{k} n {len(g)} eps {len(eps)} | " + " | ".join(line))

print("\n(d) is the gain just bias correction? M+bias (bias = mean(y-M) over S1 cases of same dir,k outside episode)")
for d, k in [("bwd", 1), ("bwd", 2), ("bwd", 3), ("bwd", 4), ("fwd", 2), ("fwd", 3), ("fwd", 4)]:
    g = S1[(S1.dir == d) & (S1.k == k)]
    mb = np.array([c.m + (g[g.ep != c.ep].yt - g[g.ep != c.ep].m).mean() if (g.ep != c.ep).any() else c.m for _, c in g.iterrows()])
    print(f"   {d} k{k}: n {len(g)} mean y {g.yt.mean():.3f}, mean M {g.m.mean():.3f}, mean anchor {g.anc.mean():.3f} | RMSE M {rm(g.m,g.yt):.3f}, M+bias {rm(mb,g.yt):.3f}, P {rm(g.anc,g.yt):.3f}, D {rm(g.D,g.yt):.3f}, BD {rm(g.BD,g.yt):.3f} | corr(y,anchor) {np.corrcoef(g.yt,g.anc)[0,1]:+.2f}, corr(y,M) {np.corrcoef(g.yt,g.m)[0,1]:+.2f}")

print("\n(e) forward k1 input-increment I vs constant I0, per case (S1)")
g = S1[(S1.dir == "fwd") & (S1.k == 1)].copy(); g["eI"] = (g.I - g.yt).abs(); g["eI0"] = (g.I0 - g.yt).abs()
print(g[["a", "t", "ep", "anc", "yt", "I", "I0", "eI", "eI0"]].round(3).to_string(index=False))
epE = comp.get("F13_175")
gg = g[g.ep != epE]
print(f"   all: RMSE I {rm(g.I,g.yt):.3f} vs I0 {rm(g.I0,g.yt):.3f} | without episode of F13_175 ({epE}): I {rm(gg.I,gg.yt):.3f} vs I0 {rm(gg.I0,gg.yt):.3f} (n {len(gg)})")

# (f) normal anchors -> model-high targets (k=2 forward), using all train consecutive links
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
def consec(a, b): return a in CAL.index and b in CAL.index and CAL.seg[a] == CAL.seg[b] and CAL.day[b] == CAL.day[a] + 1
succ = {}
for a, b in zip(TL.a1, TL.b):
    a, b = rid(int(a)), rid(int(b))
    if a in y.index and b in y.index and consec(a, b): succ.setdefault(a, []).append(b)
Mm = Ms.mean(1)
rows = []
for a in y.index:
    if y[a] >= 1.0: continue
    for b in succ.get(a, []):
        for t in succ.get(b, []):
            if t in Mm.index and Mm[t] >= 0.9:
                rows.append(dict(a=a, t=t, anc=EC[idx[a], 23], yt=y[t], m=Mm[t], ep=comp[a]))
F2 = pd.DataFrame(rows)
if len(F2):
    print(f"\n(f) normal anchor (<1) -> model-high target (>=.9), forward k=2: n {len(F2)}, episodes {F2.ep.nunique()}, truly high (>=1) {int((F2.yt>=1).sum())}")
    print(f"   RMSE M {rm(F2.m,F2.yt):.3f}, P(anchor) {rm(F2.anc,F2.yt):.3f}, half-half {rm((F2.m+F2.anc)/2,F2.yt):.3f}; mean y {F2.yt.mean():.3f}, mean M {F2.m.mean():.3f}, mean anchor {F2.anc.mean():.3f}")
    print(F2.round(3).to_string(index=False))

# (g) eval preview (not a submission): drift & blend weights from ALL training fake cases (S1 for weights, all cases for drift)
S = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv")); S["rec"] = S.row_id.str[:7]
p13 = S.groupby("rec").sub_ec.mean()
print("\n(g) eval preview (drift = mean(y_t - anchor) over all high-anchor training cases; w = LS blend weight on S1 cases; 13th = submission_22 day mean)")
for d, anc_rec, hour, anc, targets in (("bwd", "F47_229", 0, EC[idx["F47_229"], 0], [("F47_227", 1), ("F47_225", 2), ("F47_223", 3), ("F47_221", 4)]),
                                         ("fwd", "F13_217", 23, EC[idx["F13_217"], 23], [("F13_221", 2), ("F13_223", 3), ("F13_225", 4)])):
    for t, k in targets:
        ck = C[(C.dir == d) & (C.k == k)]; dr = (ck.yt - ck.anc).mean(); sdv = (ck.yt - ck.anc).std()
        g = S1[(S1.dir == d) & (S1.k == k)]; u = (g.D - g.m).values; v = (g.yt - g.m).values; w = float(np.clip(u @ v / (u @ u + 1e-12), 0, 1))
        Dv = anc + dr
        print(f"   {t} ({d} k{k} from {anc_rec} {hour}h {anc:.3f}): drift {dr:+.3f} (sd {sdv:.3f}, n {len(ck)}) -> D {Dv:.3f} | 13th {p13[t]:.3f} | w {w:.2f} -> blend {p13[t] + w*(Dv - p13[t]):.3f} (change {w*(Dv - p13[t]):+.3f})")
print(f"   F13_237 (fwd k2 from F13_233 23h {EC[idx['F13_233'],23]:.3f}, normal anchor): 13th {p13['F13_237']:.3f}  — see (f) for what normal->high cases say")
