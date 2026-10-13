# -*- coding: utf-8 -*-
"""DS5 (2026-10-13 집 클로드; 사용자: '개선될 것 같은 방향으로 고쳐봐').  사전 고정(커밋 후 실행).  EC.  상한 관문(사슬 = 진단용 깨끗한 연결, DS1 과 같음).
DS1X(6.816) 설계 결함: 관측 c | s 를 정규분포로 두어 보통 상태 꼬리가 얇음 → 예측 ≥ 1 이면 P(H) .97~1.0 과신(실제 고EC 비율은 1.0~1.3 구간 55%).
고침: c 관측 우도를 판별 보정으로 — 폴드 밖 학습 기록에서 로지스틱 P(H | c)(c = 예측 하루 평균, sklearn 기본 C=1) 를 맞추고
  우도비 = [p(c)/(1−p(c))] / [사전 오즈] 로 HMM 전진 계산에 넣음(손으로 정한 경계 없음).  z 관측·전이·사슬·상태별 회귀는 DS1 그대로.
안: DS5(사슬 있음), DS5-static(사슬 없음, 비교).  판정(미리 정함, DS1 과 같음): DIAG10 3시드 모두 DS5 의 전체 행·2차 행·2차 고EC 개선 → 규정판 진행 가치.
아래는 DS1 원문을 고친 것."""
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
    new = {"DS5": sg.copy(), "DS5-static": sg.copy()}
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
        from sklearn.linear_model import LogisticRegression
        LR = LogisticRegression().fit(cs.reshape(-1, 1), ss); po = prior[1] / prior[0]
        def lik(c, z):
            p = float(np.clip(LR.predict_proba([[c]])[0, 1], 1e-4, 1 - 1e-4)); lr_ = (p / (1 - p)) / po
            return np.array([(pi[0] if z else 1 - pi[0]), lr_ * (pi[1] if z else 1 - pi[1])])
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
                for nm, a0 in (("DS5", (alpha @ A) if chain else prior), ("DS5-static", prior)):
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
for nm in ("DS5", "DS5-static"):
    gg = g[nm]; print(f"\n{nm} pass-2 record dSSE top gains:", (-gg).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-gg).sort_values().head(6).round(3).to_dict())
ok = all(S[S.arm == "DS5"][c].lt(0).all() for c in ("all", "p2", "p2_high"))
print("\nDS5 upper-bound gate:", "PASS -> build rule-safe version" if ok else "FAIL -> stop")
