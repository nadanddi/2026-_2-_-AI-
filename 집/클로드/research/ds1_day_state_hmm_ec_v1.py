# -*- coding: utf-8 -*-
"""DS1 (2026-10-13 집 클로드; 사용자: 'RNN 을 이 문제에 맞게 — 날짜 단위 동 상태 모델 진행').  사전 고정(커밋 후 실행).  EC.  1단계 = 상한(진단).
날짜 단위 2상태 순환 모델(HMM 꼴 작은 RNN): 숨은 상태 s ∈ {보통 N, 고EC H}(정답 하루 평균 ≥ 1.0), 하루 = 한 단계.
  전이 P(s_d | s_{d−1}) — 학습 기록의 '깨끗한 같은 동 연속일'(진단 연결) 쌍에서 셈.
  관측 o_d = (모델 예측 누적평균 c, 같은 날 학습 고EC 쌍둥이 표시 z):  c | s ~ 정규(μ_s, σ_s),  z | s ~ 베르누이(π_s).
  출력: 사슬 이전 날(최대 3단계, 각 날의 하루 전체 예측 평균·z 를 관측으로; 정답은 안 씀 = 평가 순수 창 흉내)과 오늘(0..h 누적평균 c_h, z_h)로 앞으로 걸러
    P(H | 과거·오늘) → ŷ_h = Σ_s P(s)·(a_s + b_s·c_h),  pred_h = clip(sg_h + (ŷ_h − c_h)).
  매개변수(전이 2·관측 6·회귀 4 = 12): 폴드 밖 학습 기록으로 적합(DIAG10 OOF sg 시드 평균 사용).
1단계(이 파일) = 상한: 사슬 = 진단용 깨끗한 연결(truth a1 & DM11 차 1; 평가 입력 달력 사용 — 규정판 아님).  사슬 없는 기록은 오늘 관측만(= 정적 혼합).
판정(상한 관문, 미리 정함): DIAG10 3시드 모두 2차 행·전체 행 개선 AND 고EC(정답 ≥ 1.0) 2차 기록 RMSE 개선 → '규정판(2단계) 진행할 가치 있음'.  아니면 방향 종료.
비교 안: DS1 사슬 있음 / DS1-static(사슬 없이 오늘 관측만) — 사슬(연속성) 몫을 분리.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
PREV = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
XR = np.round(X, 1)
def zflag(i, h, vis):
    c = np.where(vis)[0]; c = c[c != i]; t = c[np.all(np.abs(XR[c, :h + 1] - XR[i, :h + 1]) < 1e-6, axis=(1, 2))]; t = t[np.isfinite(DM[t])]
    return float((DM[t] >= 1.2).any())
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
res = []; recrows = []
for sd, G in O.groupby("seed"):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179
    G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()); cm = G.cm.values
    day_sg = G.groupby("rec").sg.mean(); fold_of = G.groupby("rec").fold.first(); hidden = G.groupby("fold").rec.apply(set).to_dict()
    new = {"DS1": sg.copy(), "STATIC": sg.copy()}
    for k, hs in hidden.items():
        tr = [r for r in day_sg.index if r not in hs]; st = {r: DM[idx[r]] >= 1.0 for r in tr}
        vis = TRAIN & ~np.array([rid(i) in hs for i in range(N)])
        # transitions from clean consecutive pairs among training (non-fold) records
        pr = [(st[PREV[r]], st[r]) for r in tr if r in PREV and PREV[r] in st]
        A = np.array([[sum(1 for a, b in pr if a == s0 and b == s1) + 1 for s1 in (0, 1)] for s0 in (0, 1)], float); A /= A.sum(1, keepdims=True)
        prior = np.array([1 - np.mean(list(st.values())), np.mean(list(st.values()))])
        cs = np.array([day_sg[r] for r in tr]); ss = np.array([st[r] for r in tr]).astype(int); ys = np.array([DM[idx[r]] for r in tr])
        zs = np.array([zflag(idx[r], 23, vis) for r in tr])
        mu = [cs[ss == s].mean() for s in (0, 1)]; sdv = [max(cs[ss == s].std(), .05) for s in (0, 1)]; pi = [(zs[ss == s].sum() + 1) / ((ss == s).sum() + 2) for s in (0, 1)]
        reg = [np.polyfit(cs[ss == s], ys[ss == s], 1) for s in (0, 1)]
        lik = lambda c, z: np.array([norm.pdf(c, mu[s], sdv[s]) * (pi[s] if z else 1 - pi[s]) for s in (0, 1)])
        for r in hs:
            if r not in day_sg.index: continue
            ix = G.index[G.rec == r].values; i = idx[r]
            chain = []; cur = r
            for _ in range(3):
                pv = PREV.get(cur)
                if pv is None or pv not in day_sg.index: break
                chain.append(pv); cur = pv
            alpha = prior.copy()
            for pv in reversed(chain):
                alpha = (alpha @ A) if pv != chain[-1] or True else alpha
                alpha = alpha * lik(day_sg[pv], zflag(idx[pv], 23, vis)); alpha /= alpha.sum()
            for j in ix:
                h = int(G.hour.values[j]); c = cm[j]; z = zflag(i, h, vis)
                for nm, a0 in (("DS1", (alpha @ A) if chain else prior), ("STATIC", prior)):
                    post = a0 * lik(c, z); post /= post.sum()
                    yh = sum(post[s] * np.polyval(reg[s], c) for s in (0, 1))
                    new[nm][j] = np.clip(sg[j] + (yh - c), lo_, hi_)
    hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    for nm, n_ in new.items():
        res.append(dict(seed=sd, arm=nm, all=100 * (rm(n_ - y) / rm(sg - y) - 1), p2=100 * (rm((n_ - y)[p2]) / rm((sg - y)[p2]) - 1),
                        p2_high=100 * (rm((n_ - y)[p2 & hi]) / rm((sg - y)[p2 & hi]) - 1), p2_norm=100 * (rm((n_ - y)[p2 & ~hi]) / rm((sg - y)[p2 & ~hi]) - 1)))
        D = pd.DataFrame(dict(rec=G.rec, e0=(sg - y) ** 2, e1=(n_ - y) ** 2, p2=p2)).groupby("rec").agg(d=("e1", "sum"), b=("e0", "sum"), p2=("p2", "first")); D["dsse"] = D.d - D.b
        for r_, row in D[D.p2].iterrows(): recrows.append(dict(seed=sd, arm=nm, rec=r_, dsse=row.dsse))
S = pd.DataFrame(res); pd.set_option("display.width", 200); print(S.round(2).to_string(index=False))
R = pd.DataFrame(recrows); g = R.groupby(["arm", "rec"]).dsse.mean()
for nm in ("DS1", "STATIC"):
    gg = g[nm]; print(f"\n{nm} pass-2 record dSSE top gains:", (-gg).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-gg).sort_values().head(6).round(3).to_dict())
ok = all(S[S.arm == "DS1"][c].lt(0).all() for c in ("all", "p2", "p2_high"))
print("\nDS1 upper-bound gate:", "PASS -> build rule-safe version" if ok else "FAIL -> stop")
