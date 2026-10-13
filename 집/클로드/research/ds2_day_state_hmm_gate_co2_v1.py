# -*- coding: utf-8 -*-
"""DS2 (2026-10-13 집 클로드; 사용자: 'DS2 로 이어가봐').  사전 고정(커밋 후 실행).  EC.  상한 관문(사슬 = 진단용 깨끗한 연결, DS1 과 같음).
DS1(6.806): 2차 고EC −34% 이나 보통 동 +25%(혼합 기대값이 애매한 보통 동·가짜 고EC 를 끌어올림) → 두 장치:
  (1) 확신 문턱: max_s P(s) ≥ .8 일 때만 보정, 아니면 그대로(ŷ = c).
  (2) 동 유형 관측: CO₂ 공급(act_co2) 0..h 누적평균 q | s ~ 정규(ν_s, τ_s) (6.793: 가짜 고EC 는 CO₂ 공급이 진짜 고EC 보다 +2.47 SD 많음).
안: DS2b = DS1 + (1);  DS2a = DS1 + (1) + (2).   판정(미리 정함): DIAG10 3시드 모두 전체 행·2차 행·2차 고EC 개선 → '규정판 진행 가치 있음'.
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
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
CO2 = _X.pivot(index="rec", columns="h", values="act_co2").reindex(index=[rid(i) for i in range(N)], columns=range(24)).T.expanding().mean().T.values
def zflag(i, h, vis):
    c = np.where(vis)[0]; c = c[c != i]; t = c[np.all(np.abs(XR[c, :h + 1] - XR[i, :h + 1]) < 1e-6, axis=(1, 2))]; t = t[np.isfinite(DM[t])]
    return float((DM[t] >= 1.2).any())
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
res = []; recrows = []
for sd, G in O.groupby("seed"):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179
    G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()); cm = G.cm.values
    day_sg = G.groupby("rec").sg.mean(); fold_of = G.groupby("rec").fold.first(); hidden = G.groupby("fold").rec.apply(set).to_dict()
    new = {"DS2a": sg.copy(), "DS2b": sg.copy()}
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
        qs = np.array([np.nan_to_num(CO2[idx[r], 23]) for r in tr]); nu = [qs[ss == s].mean() for s in (0, 1)]; tau = [max(qs[ss == s].std(), 1.0) for s in (0, 1)]
        lik = lambda c, z: np.array([norm.pdf(c, mu[s], sdv[s]) * (pi[s] if z else 1 - pi[s]) for s in (0, 1)])
        likq = lambda q: np.array([norm.pdf(q, nu[s], tau[s]) for s in (0, 1)])
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
                a0 = (alpha @ A) if chain else prior; q = np.nan_to_num(CO2[i, h])
                for nm, useq in (("DS2a", True), ("DS2b", False)):
                    post = a0 * lik(c, z) * (likq(q) if useq else 1.0); post /= post.sum()
                    if post.max() < .8: continue
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
for nm in ("DS2a", "DS2b"):
    gg = g[nm]; print(f"\n{nm} pass-2 record dSSE top gains:", (-gg).sort_values(ascending=False).head(6).round(3).to_dict(), "| worst:", (-gg).sort_values().head(6).round(3).to_dict())
ok = all(S[S.arm == "DS1"][c].lt(0).all() for c in ("all", "p2", "p2_high"))
print("\nDS1 upper-bound gate:", "PASS -> build rule-safe version" if ok else "FAIL -> stop")
