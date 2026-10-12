# -*- coding: utf-8 -*-
# hx8/forensics fx9 v1: fx5 정정판 — fx5 의 진단 연결 사슬(truth.pkl)은 c 자신의 0시·23시 정답 연속으로 a·d 를 고른 것이라
#   (d 0시 ≈ c 23시가 연결 정의) 보간이 '맞게 나오도록' 새어 있음 → 증거로 못 씀. 여기서는 c 의 정답을 전혀 안 쓰는 경로만 쓴다.
# 경로(F13_237 에 쓰는 것과 같음, c 의 정답 미사용):
#   다음날: c 의 쌍둥이(외기 24h 동일 학습 기록) t 의 정답 연속 후속 s_t(진단 연결 c1 ≤ 2, 여유 c2/c1 ≥ 10; t·s_t 정답만 사용) →
#     묶음(s_t) 에서 쌍둥이들의 후속을 빼고 c 의 구동기 지문 최근접 = d. 앞날: 같은 방식으로 쌍둥이의 선행 p_t → a.
# 사전 고정 규칙:
#   대상: 학습 기록 c, 검증기 OOF 하루평균 sg ≥ 1.0(주; 민감도 .9), d 가 있는 것.
#   E1(주): d 0시 EC < .9 이면(모델 고EC 인데 다음날 같은 동이 보통) c 를 d 0시 값(하루 평탄)으로 대체, 아니면 모델 유지.
#   E2(보조): a·d 둘 다 있고 a23·d0 < .9 이면 a23→d0 직선 보간(k_p = k_n = 1), 아니면 모델.
#   Eβ: E1 발동 사례에서 sg + β(d0 − sg), β 는 사례 하나 빼기.
#   판정: 발동 사례의 하루 수준 RMSE·개선 비율, DIAG10·P2LOO·EL1 × 3시드 같은 방향 + DIAG10 사례 부트스트랩 p_worse < .025;
#     발동 서로 다른 기록 < 10 이면 '검증 부족(판정 불가)'.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3

OUT = os.path.dirname(os.path.abspath(__file__))
rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)]); ix = {r: i for i, r in enumerate(rid)}
tr = np.where(TRAIN)[0]
sdA = np.nanstd(AC[tr].reshape(-1, 5), 0) + 1e-6
def acd(x, y): return np.nanmean(np.abs(AC[x] - AC[y]) / sdA)
key = {}
for i in tr: key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
bund = {i: v for v in key.values() for i in v}
t = pd.read_pickle(S3 + r"\truth.pkl")
t = t[t.a1.map(lambda a: TRAIN[a]) & t.b.map(lambda b: TRAIN[b])].copy()
t["marg"] = t.c2 / t.c1.clip(lower=.05)
good = t[(t.c1 <= 2.0) & (t.marg >= 10)]
succ = {}; prev = {}
for _, r in good.iterrows(): succ.setdefault(int(r.a1), []).append(int(r.b)); prev[int(r.b)] = int(r.a1)
truth_next = {int(r.a1): int(r.b) for _, r in t[t.c1 <= 1.0].iterrows()}
truth_prev = {int(r.b): int(r.a1) for _, r in t[t.c1 <= 1.0].iterrows()}
def route(c):
    tw = [k for k in bund.get(c, []) if k != c]
    S = [s for k in tw for s in succ.get(k, [])]; Pv = [prev[k] for k in tw if k in prev]
    d = a = None
    if S:
        B = bund.get(S[0], [S[0]]); rest = [k for k in B if k not in S and k != c]
        if rest: d = min(rest, key=lambda k: acd(c, k))
    if Pv:
        B = bund.get(Pv[0], [Pv[0]]); rest = [k for k in B if k not in Pv and k != c]
        if rest: a = min(rest, key=lambda k: acd(c, k))
    return a, d
RT = {c: route(c) for c in tr}
P = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
P = P[P.tag == "CUR"].copy(); P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:].astype(int)
rows = []
for (v, s), S in P.groupby(["validator", "seed"]):
    for r, g in S.groupby("rec"):
        if len(g) != 24: continue
        c = ix[r]; g = g.sort_values("h"); sg = g.sg.values; y = EC[c]
        if np.isnan(y).any(): continue
        a, d = RT[c]
        if d is None: continue
        ip = None
        if a is not None:
            ip = EC[a, 23] + (EC[d, 0] - EC[a, 23]) * (24 + np.arange(24) - 23) / (48 - 23)
        rows.append(dict(validator=v, seed=s, rec=r, sgm=sg.mean(), y=y.mean(), d=rid[d], d0=EC[d, 0],
                         d_true=truth_next.get(c) == d, a=rid[a] if a is not None else "", a23=EC[a, 23] if a is not None else np.nan,
                         a_true=(truth_prev.get(c) == a) if a is not None else np.nan,
                         sse_m=((sg - y) ** 2).sum(), sse_e1=((EC[d, 0] - y) ** 2).sum(),
                         sse_e2=((ip - y) ** 2).sum() if ip is not None else np.nan, ipm=ip.mean() if ip is not None else np.nan))
A = pd.DataFrame(rows)
A.to_csv(os.path.join(OUT, "fx9_cases_v1.csv"), index=False)
print("records with a next-day route anchor (DIAG10 seed 47):", int(((A.validator == "DIAG10") & (A.seed == 47)).sum()),
      "| route d == diagnostic next:", round(A[(A.validator == "DIAG10") & (A.seed == 47)].d_true.mean(), 3))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
out = []
for tm in (1.0, 0.9):
    for (v, s), g in A.groupby(["validator", "seed"]):
        h = g[g.sgm >= tm]; f1 = h[h.d0 < .9]
        f2 = h[(h.a != "") & (h.d0 < .9) & (h.a23 < .9)]
        bet = []
        for k in range(len(f1)):
            o = f1.drop(f1.index[k]); u = (o.d0 - o.sgm).values; e = (o.y - o.sgm).values
            bet.append(np.clip((u * e).sum() / max((u * u).sum(), 1e-9), 0, 1) if len(o) else 0.0)
        eb = (f1.sgm + np.array(bet) * (f1.d0 - f1.sgm) - f1.y) if len(f1) else pd.Series(dtype=float)
        out.append(dict(tau_m=tm, validator=v, seed=s, n_high=len(h), e1_fire=len(f1),
                        e1_day_model=np.sqrt(((f1.sgm - f1.y) ** 2).mean()) if len(f1) else np.nan,
                        e1_day_anchor=np.sqrt(((f1.d0 - f1.y) ** 2).mean()) if len(f1) else np.nan,
                        e1_day_beta=np.sqrt((eb ** 2).mean()) if len(f1) else np.nan,
                        e1_improved=((f1.d0 - f1.y).abs() < (f1.sgm - f1.y).abs()).mean() if len(f1) else np.nan,
                        e1_hr_model=np.sqrt(f1.sse_m.sum() / 24 / max(len(f1), 1)), e1_hr_anchor=np.sqrt(f1.sse_e1.sum() / 24 / max(len(f1), 1)),
                        e2_fire=len(f2), e2_hr_model=np.sqrt(f2.sse_m.sum() / 24 / max(len(f2), 1)), e2_hr_interp=np.sqrt(f2.sse_e2.sum() / 24 / max(len(f2), 1)),
                        beta=np.mean(bet) if bet else np.nan, d_true=f1.d_true.mean() if len(f1) else np.nan))
O = pd.DataFrame(out); O.to_csv(os.path.join(OUT, "fx9_summary_v1.csv"), index=False)
print(O.round(3).to_string(index=False))
g = A[(A.validator == "DIAG10") & (A.sgm >= 1.0)]
L = g.groupby("rec").agg(n=("seed", "size"), y=("y", "mean"), sg=("sgm", "mean"), d=("d", "first"), d0=("d0", "first"), d_true=("d_true", "first"),
                          a=("a", "first"), a23=("a23", "first"))
print("\nDIAG10 model-high (sg>=1.0) records with route anchor (seed-mean):"); print(L.round(3).to_string())
for tm in (1.0,):
    f = A[(A.validator == "DIAG10") & (A.sgm >= tm) & (A.d0 < .9)]
    pc = f.groupby("rec").agg(m=("sse_m", "mean"), e=("sse_e1", "mean"))
    if len(pc):
        rng = np.random.default_rng(11); dif = np.array([(lambda s: s.e.sum() - s.m.sum())(pc.sample(len(pc), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(20000)])
        print(f"bootstrap DIAG10 E1 tau {tm}: distinct fired records {len(pc)}, SSE model {pc.m.sum():.2f} vs anchor {pc.e.sum():.2f}, p_worse {(dif >= 0).mean():.4f}")
c = ix["F13_237"]; a, d = RT[c]
print("\nF13_237 route: a =", rid[a] if a is not None else None, "| d =", rid[d] if d is not None else None, f"d0 {EC[d, 0]:.3f}" if d is not None else "")
