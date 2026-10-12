# -*- coding: utf-8 -*-
"""CMP3 고EC 2단계(고EC 학습끼리 입력 이웃) 보정의 '규정 안 인과판' + 같은 날짜 제외 + 평가 8기록 추정
(2026-10-13 집 클로드, hx8/compress; 이전 에이전트가 CMP2 후 한도로 멈춘 지점에서 이어감).

[CMP2 의 남은 문제 — 이 스크립트가 고치는 것]
 (i) CMP2 의 KH 는 하루 전체 24시간 입력(S2)으로 이웃을 찾아 0시 행에도 같은 보정폭을 더함 → 평가 h시 행이
     자기 h시 이후 입력을 쓰는 꼴(규정 (1) 위반 소지). 여기서는 h시 행은 자기 0..h시 입력만으로 이웃을 찾음.
 (ii) F13·F47 은 같은 장소라 같은 날짜 기록의 외기 24시간이 완전히 같음(6.510). 입력 이웃 1위가 '같은 날짜 다른 동'이
     되기 쉬워, 평가와 다른 정보 구조가 생김 → 이웃 후보에서 같은 날짜 기록(외기 4종 24시간 평균 절대차 < 1e-6)을 뺌.
 (iii) CMP2 는 평가 8기록에 KH 를 적용하지 않음 → 여기서 추정치를 냄.

[사전 고정 — 실행 전에 적음]
가짜 평가 사례: DIAG10 저장 OOF(CUR) 하루 평균 pm >= .9 학습 기록(헛경보 포함), 시드 47/1414/6464 × {sg, nosg} = 6칸.
에피소드(교차적합 단위): 학습 기록 중 (정답 하루 평균 >= 1 또는 시드평균 OOF >= .9) 를 진단 연결(c1 <= 2.0)로 묶은 덩어리.
시각 h 의 보정(인과): cm_h = 모델 시간 예측 0..h 누적 평균. cm_h >= .9 일 때만 hp_h + Δ_h, 아니면 hp_h.
  SH   : Δ = a (다른 에피소드 표지 행의 평균 잔차)                                  [기준선, 닫힌 HG3/U 계열]
  HC   : Δ_h = w (μ_hi − cm_h), μ_hi = 후보 고EC 학습 기록 하루 평균 정답의 평균        [이웃 없이 고EC 평균으로 당기기]
  KHk  : Δ_h = w (knn_h − cm_h), knn_h = 후보 중 0..h 시 표준화 입력(14채널) 거리 최근접 k 의 하루 평균 정답
  w    : [0,1] 로 자른 최소제곱, 다른 에피소드 표지 행(y_h − hp_h ~ w z_h)에서만.
후보(이웃 풀) = 정답 하루 평균 >= 1 학습 기록(35) 중
  P1(엄격): 대상 에피소드 전체 제외 + 대상과 같은 날짜 제외 + 자기 자신 제외
  P2(평가형): 대상과 진단 연결 ±1 기록 제외 + 같은 날짜 제외 + 자기 제외(같은 에피소드라도 2일+ 떨어진 기록은 허용 —
              평가 F13_221 이 2일 앞 F13_217 을 풀에 가지는 상황 흉내). 단 w 맞춤 기록 jj 의 풀에서는 대상 j 를 늘 제외.
변형(본페로니 5): HC_P1, KH3_P1, KH5_P1, KH10_P1, KH5_P2. 1순위 = KH5_P1.
판정(사전): 1순위가 (a) DIAG10 6칸 모두 하루 수준 RMSE(시간 보정의 하루 평균 오차) M0 보다 낮고,
  (b) 6칸 모두 에피소드 부트스트랩 P(M0 보다 나쁨) < .025/5 = .005, (c) P2LOO·EL1 의 예측 고EC 기록(계수는 DIAG10 사례에서,
  대상 에피소드 제외)에서도 6칸 모두 하루 수준 SSE 감소 → '유망'. (a)만 → '약함'. (a) 실패 → '종료'.
  다른 변형은 같은 기준의 보조(본페로니 .005).
평가 8기록: 시간 예측 = 13회차(submission_22) 그 기록 값, 이웃 = 그 기록 0..h 시 입력, 풀 = 고EC 학습 35 중 같은 날짜 제외,
  계수 = DIAG10 시드평균 sg 사례 전체(평가 입력·예측에 아무것도 맞추지 않음). 서술용 추정치이며 제출물 아님.
"""
import sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cm_common import *

TAU = 0.9; HIT = 1.0; NB = 5
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

pa = {int(b): int(a) for a, b, c in zip(T.a1, T.b, T.c1) if c <= 2.0}
nb1 = defaultdict(set)
for b, a in pa.items():
    nb1[a].add(b); nb1[b].add(a)

# 같은 날짜(외기 4종 24시간 동일)
O4 = np.nan_to_num(A[:, :, 3:7])
SD = np.zeros((N, N), bool)
for i in range(N):
    SD[i] = np.abs(O4 - O4[i]).mean((1, 2)) < 1e-6
np.fill_diagonal(SD, False)

# 표준화 입력과 0..h 누적 거리 (훈련 통계로 표준화)
trn = np.where(TRAIN)[0]
mu = np.nanmean(A[trn].reshape(-1, 14), 0); sdv = np.nanstd(A[trn].reshape(-1, 14), 0) + 1e-9
Z = np.nan_to_num((A - mu) / sdv)
Dc = np.zeros((24, N, N), np.float32)
acc = np.zeros((N, N))
for t in range(24):
    zt = Z[:, t, :]; sq = (zt ** 2).sum(1)
    acc += np.maximum(sq[:, None] + sq[None, :] - 2 * zt @ zt.T, 0)
    Dc[t] = acc / (t + 1)

# 시드평균 OOF(그룹용): DIAG10 우선, 없으면 EL1/P2LOO
sgD, _ = oof("DIAG10"); sgE, _ = oof("EL1"); sgP, _ = oof("P2LOO")
pmall = {}
for k in r.rid[TRAIN]:
    v = sgD.get(k, sgE.get(k, sgP.get(k)))
    if v is not None: pmall[rid2i[k]] = float(np.mean(v))
Fset = {i for i in range(N) if TRAIN[i] and (ydm[i] >= HIT or pmall.get(i, 0) >= TAU)}
par = {i: i for i in range(N)}
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for b, a in pa.items():
    if b in Fset and a in Fset: par[fd(b)] = fd(a)
grp = np.array([fd(i) for i in range(N)])
HI = np.array([i for i in range(N) if TRAIN[i] and ydm[i] >= HIT])
P(f"고EC 학습(풀) {len(HI)}, 에피소드 수(풀 기준) {len(set(grp[HI]))}")

_cache = {}
def pool(i, rule, excl_j=None):
    key = (i, rule, excl_j)
    if key in _cache: return _cache[key]
    m = np.ones(len(HI), bool)
    m &= HI != i
    m &= ~SD[i, HI]
    if rule == "P1":
        m &= grp[HI] != grp[i]
    elif rule == "P2":
        m &= ~np.isin(HI, list(nb1[i]))
    if excl_j is not None:
        if rule == "P1":
            m &= grp[HI] != grp[excl_j]
        m &= HI != excl_j
    _cache[key] = HI[m]
    return HI[m]

_kc = {}
def knn_curve(i, rule, K, excl_j=None):
    """24시간 각 h 의 고EC 이웃 정답 평균 (h시: 0..h 입력 거리)."""
    key = (i, rule, K, excl_j)
    if key in _kc: return _kc[key]
    cand = pool(i, rule, excl_j)
    if K == 0:  # HC
        v = np.full(24, ydm[cand].mean())
    else:
        v = np.array([ydm[cand[np.argsort(Dc[h, i, cand])[:K]]].mean() for h in range(24)])
    _kc[key] = v
    return v

METHODS = {"SH": ("SH", None, None), "HC_P1": ("W", "P1", 0), "KH3_P1": ("W", "P1", 3), "KH5_P1": ("W", "P1", 5),
           "KH10_P1": ("W", "P1", 10), "KH5_P2": ("W", "P2", 5)}


def case_table(validator, col, seed):
    sg, _ = oof(validator, col, seed)
    rows = []
    for k, v in sg.items():
        i = rid2i[k]
        if not TRAIN[i]: continue
        hp = np.asarray(v, float)
        if hp.mean() < TAU: continue
        cm = np.cumsum(hp) / np.arange(1, 25)
        rows.append(dict(i=i, rid=k, hp=hp, cm=cm, yh=Ytr[i], y=ydm[i], pm=hp.mean()))
    return rows


def coef(method, fitrows, excl_j):
    kind, rule, K = METHODS[method]
    zs, ts = [], []
    for d in fitrows:
        g = d["cm"] >= TAU
        if not g.any(): continue
        t = (d["yh"] - d["hp"])[g]
        if kind == "SH":
            ts.append(t); continue
        z = (knn_curve(d["i"], rule, K, excl_j) - d["cm"])[g]
        zs.append(z); ts.append(t)
    if kind == "SH":
        return float(np.concatenate(ts).mean())
    z = np.concatenate(zs); t = np.concatenate(ts)
    return float(np.clip((z * t).sum() / (z * z).sum(), 0, 1))


def apply(method, d, c, excl_none=True):
    kind, rule, K = METHODS[method]
    g = d["cm"] >= TAU
    if kind == "SH":
        dl = np.full(24, c)
    else:
        dl = c * (knn_curve(d["i"], rule, K, None) - d["cm"])
    return np.where(g, d["hp"] + dl, d["hp"])


def boot_p(e_new, e_ref, gg, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(gg)
    dv = np.array([(e_new[gg == u] ** 2).sum() - (e_ref[gg == u] ** 2).sum() for u in ug])
    bs = np.array([dv[rng.integers(0, len(ug), len(ug))].sum() for _ in range(B)])
    return float((bs >= 0).mean()), float(dv.sum())


summ = []
for col in ["sg", "nosg"]:
    for seed in SEEDS:
        D10 = case_table("DIAG10", col, seed)
        y = np.array([d["y"] for d in D10]); gg = grp[[d["i"] for d in D10]]
        hi = y >= HIT
        base_h = np.stack([d["hp"] for d in D10]); yh = np.stack([d["yh"] for d in D10])
        e0 = (yh - base_h).mean(1)
        summ.append(dict(V="DIAG10", col=col, seed=seed, model="M0", n=len(D10), n_hi=int(hi.sum()), rmse=rmse(e0),
                         rmse_hi=rmse(e0[hi]), rmse_fp=rmse(e0[~hi]), rmse_hour=rmse(yh - base_h)))
        for mname in METHODS:
            preds = []; cs = []
            for j, d in enumerate(D10):
                fit = [dd for jj, dd in enumerate(D10) if gg[jj] != gg[j]]
                c = coef(mname, fit, d["i"]); cs.append(c)
                preds.append(apply(mname, d, c))
            ph = np.stack(preds); e = (yh - ph).mean(1)
            p, dsse = boot_p(e, e0, gg)
            summ.append(dict(V="DIAG10", col=col, seed=seed, model=mname, n=len(D10), n_hi=int(hi.sum()), rmse=rmse(e),
                             rmse_hi=rmse(e[hi]), rmse_fp=rmse(e[~hi]), rmse_hour=rmse(yh - ph),
                             impr=float(np.mean(np.abs(e) < np.abs(e0) - 1e-12)), chg=int(np.sum(np.abs(e - e0) > 1e-9)),
                             P_vs_M0=p, dSSE=dsse, coef_med=float(np.median(cs))))
        # 다른 검증기: 계수는 같은 칸 DIAG10 사례(대상 에피소드 제외)
        for V in ["P2LOO", "EL1"]:
            R2 = case_table(V, col, seed)
            if not R2: continue
            y2 = np.array([d["y"] for d in R2]); yh2 = np.stack([d["yh"] for d in R2]); b2 = np.stack([d["hp"] for d in R2])
            e20 = (yh2 - b2).mean(1)
            summ.append(dict(V=V, col=col, seed=seed, model="M0", n=len(R2), n_hi=int((y2 >= HIT).sum()), rmse=rmse(e20)))
            for mname in METHODS:
                preds = []
                for d in R2:
                    fit = [dd for dd in D10 if grp[dd["i"]] != grp[d["i"]] and dd["i"] != d["i"]]
                    preds.append(apply(mname, d, coef(mname, fit, d["i"])))
                e2 = (yh2 - np.stack(preds)).mean(1)
                summ.append(dict(V=V, col=col, seed=seed, model=mname, n=len(R2), n_hi=int((y2 >= HIT).sum()), rmse=rmse(e2),
                                 dSSE=float((e2 ** 2).sum() - (e20 ** 2).sum()), impr=float(np.mean(np.abs(e2) < np.abs(e20) - 1e-12))))
        P(f"[{col} {seed}] 완료: DIAG10 사례 {len(D10)} (실제 고EC {int(hi.sum())})")

S = pd.DataFrame(summ)
S.to_csv(os.path.join(HERE, "cmp3_causal_kh_v1_summary.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
P("\n=== DIAG10 예측 고EC 학습 기록: 인과 시간 보정의 하루 수준 RMSE ===")
d10 = S[S.V == "DIAG10"]
P(d10[["col", "seed", "model", "n", "n_hi", "rmse", "rmse_hi", "rmse_fp", "rmse_hour", "impr", "chg", "P_vs_M0", "dSSE", "coef_med"]].round(4).to_string())
P("\n=== 변형별 판정 요약 ===")
m0 = d10[d10.model == "M0"].set_index(["col", "seed"]).rmse
for V in ["P2LOO", "EL1"]:
    s = S[(S.V == V) & (S.model == "M0")]
    P(f"  {V} 예측 고EC 사례 수(칸별) {s.n.tolist()}, 실제 고EC {s.n_hi.tolist()}")
for mname in METHODS:
    s = d10[d10.model == mname].set_index(["col", "seed"])
    a_ = int((s.rmse < m0).sum()); pmax = s.P_vs_M0.max()
    oth = S[(S.V != "DIAG10") & (S.model == mname)]
    c_ = int((oth.dSSE < 0).sum()); n_ = len(oth)
    verdict = "유망" if (a_ == 6 and pmax < .005 and c_ == n_) else ("약함" if a_ == 6 else "종료")
    P(f"  {mname:8s} DIAG10 RMSE {s.rmse.mean():.4f} (M0 {m0.mean():.4f}) | 나은 칸 {a_}/6 | P 최대 {pmax:.3f} | 개선 기록 비율 {s.impr.mean():.2f} | "
      f"고EC {s.rmse_hi.mean():.3f} 헛경보 {s.rmse_fp.mean():.3f} | P2LOO·EL1 SSE 감소 {c_}/{n_} → {verdict}")
P(f"  M0 고EC {d10[d10.model=='M0'].rmse_hi.mean():.3f}, 헛경보 {d10[d10.model=='M0'].rmse_fp.mean():.3f}")

# 같은 날짜 고EC 학습 기록이 있는 사례 수 (정보 구조 점검)
D10 = case_table("DIAG10", "sg", None)
nsd = sum(1 for d in D10 if any(SD[d["i"], HI]))
P(f"\n가짜 사례 중 같은 날짜 고EC 학습 기록이 있는 사례 {nsd}/{len(D10)}")

# ---------------- 평가 8기록 ----------------
P("\n=== 평가 8기록 추정 (계수: DIAG10 시드평균 sg 사례 전체) ===")
D10 = case_table("DIAG10", "sg", None)
cf = {m: coef(m, D10, None) for m in METHODS}
P("계수: " + ", ".join(f"{m} {v:+.3f}" for m, v in cf.items()))
rows = []
for k in EVAL8:
    i = rid2i[k]; hp = np.asarray(sub22[k], float); cm = np.cumsum(hp) / np.arange(1, 25)
    d = dict(i=i, hp=hp, cm=cm)
    rec = dict(rec=k, sub22=hp.mean(), gated_h=int((cm >= TAU).sum()))
    for m in METHODS:
        rec[m] = apply(m, d, cf[m]).mean()
    rec["knn5_23h"] = knn_curve(i, "P2", 5)[23]
    rec["knn5_0h"] = knn_curve(i, "P2", 5)[0]
    nbr = pool(i, "P2"); o = nbr[np.argsort(Dc[23, i, nbr])[:5]]
    rec["nbr5"] = " ".join(f"{r.rid[j]}:{ydm[j]:.2f}" for j in o)
    rec["same_date_hi"] = " ".join(f"{r.rid[j]}:{ydm[j]:.2f}" for j in HI if SD[i, j])
    rows.append(rec)
EV = pd.DataFrame(rows)
EV.to_csv(os.path.join(HERE, "cmp3_eval8_v1.csv"), index=False)
P(EV[["rec", "sub22", "gated_h", "SH", "HC_P1", "KH5_P1", "KH5_P2", "KH10_P1", "knn5_0h", "knn5_23h"]].round(3).to_string())
for _, rr in EV.iterrows():
    P(f"  {rr.rec} 23시 최근접 5: {rr.nbr5} | 같은 날짜 고EC 학습: {rr.same_date_hi or '-'}")
open(os.path.join(HERE, "cmp3_causal_kh_v1.log"), "w", encoding="utf-8").write("\n".join(out))
