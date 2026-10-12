# -*- coding: utf-8 -*-
# hx8/forensics fx5 v1: '모델은 고EC 라는데 같은 물리 동의 앞·뒤 학습 정답은 보통'(양쪽 보통 괄호) 검증.
# 포렌식 근거(fx1·fx4): 순수 창에서 ID(F13/F47)는 물리 동이 아니라 기록 꼬리표 — 84/85일에 동이 ID 를 바꿔 탐.
#   구동기 지문(차광·보온·유동팬)이 물리 동을 가름(fx4 쌍 선택 AUC .95, ID 전환 연결에서도 .93).
#   8기록 중 7개(F13_221·223·225, F47_221·223·225·227)는 SIG 고EC 동(6.690), F13_237 만 다른 동('차광 53' 동:
#   F13_231 .62 → 233 .63 → [235·237 평가] → F13_128 .78 → 132 .87 …). F13_237 은 앞(88일 F13_233 23시 .643)·
#   뒤(91일 F13_128 0시 .721) 같은 동 학습 정답이 2일·1일 거리에 있는 유일한 기록 = 이 검증의 표적.
#   6.570 W2k 는 앞쪽(r−4)만 썼고 검증 1건; 여기서는 뒤쪽 앵커를 더하고 '모델 고EC + 양쪽 보통' 상황을 학습에서 흉내.
# 실행 전 고정(사전 규칙):
#   사례: 학습 기록 c 중 OOF 하루평균 sg ≥ 1.0(주; 민감도 .9) 이고, 진단 연결(c1 ≤ 1.0)로 같은 동 앞 a(k_p일 전)·뒤 d(k_n일 후)
#     학습 정답이 이어지는 것. 장면 S3 = (k_p,k_n) ∈ {(2,1),(1,2)} (F13_237 처럼 두 앵커 사이 3일), 없으면 S2 = (1,1) 따로 보고.
#     장면 안쪽 기록(a 와 d 사이)은 정답 없음으로 간주.
#   규칙 FB: a 23시 EC < .9 이고 d 0시 EC < .9(양쪽 보통)이면 c 를 a23→d0 직선(절대 시각) 보간으로 대체, 아니면 모델 유지.
#   보조: β 수축 est = sg + β(보간 − sg), β 는 사례 하나 빼기(사례 단위 교차적합).
#   식별 두 판: (T) 진단 연결 그대로(식별 완벽 상한), (F) 규정형 — a·d 의 같은 날짜 묶음(외기 24h 동일 학습 기록) 안에서
#     c 의 자기 구동기 지문과 가장 가까운 기록을 고름(c 자기 입력 + 학습 기록만).
#   판정: 하루 수준 RMSE·개선 사례 비율, 검증기(DIAG10·P2LOO·EL1) × 시드(3) 모두 같은 방향 + DIAG10 사례 부트스트랩 p_worse < .025.
#   표본이 작으면(사례 < 10) 판정 불가로 기록.
# 평가 적용(서술): F13_237 의 앞 앵커(같은 ID 앞 번호 구조 + 지문)와 뒤 앵커(자기 23시 외기 → 다음날 학습 묶음 0시 외기 연속성, 그 안 지문 최근접).
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3

OUT = os.path.dirname(os.path.abspath(__file__))
rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)]); ix = {r: i for i, r in enumerate(rid)}
tr_idx = np.where(TRAIN)[0]
sdA = np.nanstd(AC[tr_idx].reshape(-1, 5), 0) + 1e-6
def acd(x, y): return np.nanmean(np.abs(AC[x] - AC[y]) / sdA)
key = {}
for i in tr_idx: key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
bundle = {i: v for v in key.values() for i in v}
t = pd.read_pickle(S3 + r"\truth.pkl")
L = t[(t.c1 <= 1.0)]
pred = {int(r.b): int(r.a1) for _, r in L.iterrows() if TRAIN[int(r.a1)] and TRAIN[int(r.b)]}
succ = {}
for b, a in pred.items(): succ.setdefault(a, []).append(b)
succ = {a: v[0] for a, v in succ.items() if len(v) == 1}
def back(c, k):
    x = c
    for _ in range(k):
        x = pred.get(x)
        if x is None: return None
    return x
def fwd(c, k):
    x = c
    for _ in range(k):
        x = succ.get(x)
        if x is None: return None
    return x
P = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
P = P[P.tag == "CUR"].copy(); P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:].astype(int)

def interp(a, d, kp, kn):
    t0, t1 = 23.0, 24.0 * (kp + kn); v0, v1 = EC[a, 23], EC[d, 0]
    tt = 24.0 * kp + np.arange(24)
    return v0 + (v1 - v0) * (tt - t0) / (t1 - t0)

def pick(true_rec, c):
    cands = bundle.get(true_rec, [true_rec])
    return min(cands, key=lambda k: acd(c, k)), len(cands)

def run(validator, seed, tau_m=1.0, tau_a=0.9):
    S = P[(P.validator == validator) & (P.seed == seed)]
    G = {r: g.sort_values("h") for r, g in S.groupby("rec")}
    rows = []
    for r, g in G.items():
        c = ix[r]
        if len(g) != 24: continue
        sg = g.sg.values; y = EC[c]
        if np.isnan(y).any() or sg.mean() < tau_m: continue
        for scen, opts in (("S3", [(2, 1), (1, 2)]), ("S2", [(1, 1)])):
            got = None
            for kp, kn in opts:
                a, d = back(c, kp), fwd(c, kn)
                if a is not None and d is not None: got = (kp, kn, a, d); break
            if got is None: continue
            kp, kn, a, d = got
            for idm in ("T", "F"):
                if idm == "T": aa, dd, na, nd = a, d, 1, 1
                else:
                    aa, na = pick(a, c); dd, nd = pick(d, c)
                fire = (EC[aa, 23] < tau_a) and (EC[dd, 0] < tau_a)
                ip = interp(aa, dd, kp, kn)
                rows.append(dict(validator=validator, seed=seed, scen=scen, idm=idm, rec=r, kp=kp, kn=kn,
                                 a=rid[aa], d=rid[dd], a_ok=aa == a, d_ok=dd == d, ncand_a=na, ncand_d=nd,
                                 a23=EC[aa, 23], d0=EC[dd, 0], fire=fire, y=y.mean(), sg=sg.mean(), ip=ip.mean(),
                                 sse_m=((sg - y) ** 2).sum(), sse_i=((ip - y) ** 2).sum()))
            break  # one scenario per case: S3 preferred
    return pd.DataFrame(rows)

allr = []
for v in ("DIAG10", "P2LOO", "EL1"):
    for s in sorted(P.seed.unique()):
        for tm in (1.0, 0.9):
            d = run(v, s, tau_m=tm); d["tau_m"] = tm; allr.append(d)
A = pd.concat(allr, ignore_index=True)
A.to_csv(os.path.join(OUT, "fx5_cases_v1.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print("cases per (validator, tau_m, scen, idm), seed 47:")
print(A[A.seed == 47].groupby(["validator", "tau_m", "scen", "idm"]).agg(n=("rec", "size"), fire=("fire", "sum"),
      a_ok=("a_ok", "mean"), d_ok=("d_ok", "mean")).to_string())
print("\nDIAG10 seed 47 tau_m 1.0 case list (T identification):")
cl = A[(A.validator == "DIAG10") & (A.seed == 47) & (A.tau_m == 1.0) & (A.idm == "T")]
print(cl[["scen", "rec", "kp", "kn", "a", "d", "a23", "d0", "fire", "y", "sg", "ip"]].round(3).to_string(index=False))

def summarize(D):
    out = []
    for (v, tm, sc, idm, s), g in D.groupby(["validator", "tau_m", "scen", "idm", "seed"]):
        f = g[g.fire]
        if len(f) == 0:
            out.append(dict(validator=v, tau_m=tm, scen=sc, idm=idm, seed=s, n=len(g), nfire=0)); continue
        dm = f.sg - f.y; di = f.ip - f.y
        # beta leave-one-case-out (shrink toward interp)
        bet = []
        for k in range(len(f)):
            o = f.drop(f.index[k]); u = (o.ip - o.sg).values; e = (o.y - o.sg).values
            bet.append(np.clip((u * e).sum() / max((u * u).sum(), 1e-9), 0, 1) if len(o) else 0.0)
        eb = f.sg + np.array(bet) * (f.ip - f.sg) - f.y
        out.append(dict(validator=v, tau_m=tm, scen=sc, idm=idm, seed=s, n=len(g), nfire=len(f),
                        rmse_day_model=np.sqrt((dm ** 2).mean()), rmse_day_interp=np.sqrt((di ** 2).mean()),
                        rmse_day_beta=np.sqrt((eb ** 2).mean()), improved=(di.abs() < dm.abs()).mean(),
                        rmse_hr_model=np.sqrt(f.sse_m.sum() / (24 * len(f))), rmse_hr_interp=np.sqrt(f.sse_i.sum() / (24 * len(f))),
                        beta_mean=np.mean(bet)))
    return pd.DataFrame(out)
Sm = summarize(A)
Sm.to_csv(os.path.join(OUT, "fx5_summary_v1.csv"), index=False)
print("\nsummary over fired cases (day-level RMSE):")
print(Sm.round(3).to_string(index=False))
# bootstrap p_worse (DIAG10, tau_m 1.0, S3+S2 pooled, both idm), cases resampled, seeds averaged per case
for idm in ("T", "F"):
    g = A[(A.validator == "DIAG10") & (A.tau_m == 1.0) & (A.idm == idm) & A.fire]
    if len(g) == 0: print("bootstrap", idm, "no fired cases"); continue
    pc = g.groupby("rec").agg(sse_m=("sse_m", "mean"), sse_i=("sse_i", "mean"))
    rng = np.random.default_rng(7); diff = []
    for _ in range(20000):
        s = pc.sample(len(pc), replace=True, random_state=int(rng.integers(1e9)))
        diff.append(s.sse_i.sum() - s.sse_m.sum())
    diff = np.array(diff)
    print(f"bootstrap DIAG10 tau_m 1.0 idm {idm}: cases {len(pc)}, SSE model {pc.sse_m.sum():.3f} vs interp {pc.sse_i.sum():.3f}, p_worse {(diff >= 0).mean():.4f}")
# non-fired control: model-high cases whose bracket is high (interp would be the BI1 situation) — report only
g = A[(A.validator == "DIAG10") & (A.seed == 47) & (A.tau_m == 1.0) & (A.idm == "T") & ~A.fire]
if len(g): print(f"\nnon-fired (bracket high) DIAG10 s47: n {len(g)}, day RMSE model {np.sqrt(((g.sg - g.y) ** 2).mean()):.3f} vs interp {np.sqrt(((g.ip - g.y) ** 2).mean()):.3f}")

# ---------- eval application (descriptive): F13_237 ----------
c = ix["F13_237"]
print("\nF13_237 own AC means:", np.round(np.nanmean(AC[c], 0), 1))
# previous side: same-ID earlier numbers r-2 (eval, input only) and r-4 (train)
for r in ("F13_235", "F13_233", "F13_231"):
    print(f"  prev cand {r} train={TRAIN[ix[r]]} acd={acd(c, ix[r]):.2f} EC23={EC[ix[r], 23]:.3f}")
# next side: training bundle whose 0h outdoor is closest to F13_237 own 23h outdoor (weather continuity), then fingerprint
sdO = np.nanstd(X[tr_idx].reshape(-1, 4), 0)
o23 = X[c, 23]
cand_b = sorted({tuple(v) for v in key.values()}, key=lambda v: np.nanmean(np.abs(X[v[0], 0] - o23) / sdO))
for v in cand_b[:3]:
    dist = np.nanmean(np.abs(X[v[0], 0] - o23) / sdO)
    print(f"  next-day bundle cand (0h outdoor dist {dist:.3f}): " + ", ".join(f"{rid[k]}(acd {acd(c, k):.2f}, EC0 {EC[k, 0]:.3f})" for k in v))
a, d = ix["F13_233"], None
v0 = cand_b[0]; d = min(v0, key=lambda k: acd(c, k))
ip = interp(a, d, 2, 1)
print(f"  FB estimate F13_237 (a=F13_233 23h {EC[a, 23]:.3f}, d={rid[d]} 0h {EC[d, 0]:.3f}, kp=2 kn=1): day mean {ip.mean():.3f} (13th 1.256)")
