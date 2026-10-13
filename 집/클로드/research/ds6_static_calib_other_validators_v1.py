# -*- coding: utf-8 -*-
"""DS6 (2026-10-13 집 클로드; 사용자: '다른 검증기로 사슬 없는 판 판정해봐').  사전 고정(커밋 후 실행).  EC.  규정판(사슬 없음).
DS5-static(6.817: DIAG10 3시드 모두 전체·2차·2차 고EC 개선, 그러나 사후 선택) 을 DIAG10 이 아닌 검증기에서 새로 판정.
보정(DS5-static 그대로): 상태 N/H(하루 평균 ≥ 1.0), 사전 = 학습 기록 H 비율, c 우도비 = 로지스틱 P(H|c)(기본 C=1) 오즈 / 사전 오즈,
  z(같은 날 학습 쌍둥이 고EC ≥ 1.2, 0..h) 베르누이, 상태별 직선 회귀 c → 실제;  pred_h = clip(sg_h + Σ P(s)(a_s + b_s c_h) − c_h),  c_h = 0..h 누적평균.
검증기·매개변수 적합:
  P2LOO, EL1(ec_rv1 CUR, 시드 47/1414/6464, 2차 46기록): 매개변수는 같은 시드 DIAG10 OOF 예측의 학습 기록에서, 그 폴드 시험 기록 및 같은 농장 ±1 번호 기록 제외.
  EV(v2_ev_ec_rows BASE, 시드 7/101/2024, 400기록, 기준 = SG2): 매개변수는 같은 셀의 그 폴드 val∪del 아닌 기록에서.  z 의 보이는 정답도 같은 숨김 기준.
판정(미리 정함): 9칸(3 검증기 × 3시드) 모두 그 검증기 전체 행 RMSE 개선  AND  EV·P2LOO 각각 농장×5일 블록 부트스트랩(시드 평균 ΔSSE) P(악화) < .025.
서술: 2차 고EC·2차 보통 행 변화.  한계: P2LOO 와 EL1 은 같은 46기록(독립 아님), EV 는 DIAG10 기록을 포함.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, FARM, NUM, N, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN]); rng = np.random.default_rng(20261127)
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
XR = np.round(X, 1)
def zflag(i, h, vis):
    c = np.where(vis)[0]; c = c[c != i]; t = c[np.all(np.abs(XR[c, :h + 1] - XR[i, :h + 1]) < 1e-6, axis=(1, 2))]; t = t[np.isfinite(DM[t])]
    return float((DM[t] >= 1.2).any())
def fit(train_daysg, vis):
    tr = [r for r in train_daysg.index if TRAIN[idx[r]] and np.isfinite(DM[idx[r]])]
    cs = np.array([train_daysg[r] for r in tr]); ss = np.array([DM[idx[r]] >= 1.0 for r in tr]).astype(int); ys = np.array([DM[idx[r]] for r in tr])
    zs = np.array([zflag(idx[r], 23, vis) for r in tr]); prior = np.array([1 - ss.mean(), ss.mean()])
    pi = [(zs[ss == s].sum() + 1) / ((ss == s).sum() + 2) for s in (0, 1)]; reg = [np.polyfit(cs[ss == s], ys[ss == s], 1) for s in (0, 1)]
    LR = LogisticRegression().fit(cs.reshape(-1, 1), ss); po = prior[1] / prior[0]
    def corr(c, z):
        p = float(np.clip(LR.predict_proba([[c]])[0, 1], 1e-4, 1 - 1e-4)); l = np.array([(pi[0] if z else 1 - pi[0]), (p / (1 - p)) / po * (pi[1] if z else 1 - pi[1])])
        post = prior * l; post /= post.sum(); return sum(post[s] * np.polyval(reg[s], c) for s in (0, 1)) - c
    return corr
def apply(G, recs, corr, vis):
    out = G.sg.values.copy()
    for r in recs:
        ix = G.index[G.rec == r].values; i = idx[r]
        for j in ix: out[j] = np.clip(G.sg.values[j] + corr(G.cm.values[j], zflag(i, int(G.hour.values[j]), vis)), lo_, hi_)
    return out
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy()
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidEV = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
res, blocks = [], {}
def score(v, sd, G, new):
    y = G.y.values; sg = G.sg.values; p2 = G.day.values >= 179; hi = np.array([DM[idx[r]] >= 1.0 for r in G.rec])
    f = lambda m: 100 * (rm((new - y)[m]) / rm((sg - y)[m]) - 1) if m.any() else np.nan
    res.append(dict(val=v, seed=sd, all=f(np.ones(len(y), bool)), p2=f(p2), p2_high=f(p2 & hi), p2_norm=f(p2 & ~hi)))
    bk = (G.farm + "_" + (G.day // 5).astype(str)).values; blocks.setdefault(v, []).append(pd.Series((new - y) ** 2 - (sg - y) ** 2).groupby(bk).mean())
for sd, D in O.groupby("seed"):
    Dg = D[D.validator == "DIAG10"]; dsg = Dg.groupby("rec").sg.mean()
    for v in ("P2LOO", "EL1"):
        G = D[D.validator == v].sort_values(["rec", "hour"]).reset_index(drop=True).rename(columns={"sub_ec": "y"})
        G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()); new = G.sg.values.copy()
        for k, Gk in G.groupby("fold"):
            te = set(Gk.rec); excl = {r for r in dsg.index for t in te if FARM[idx[r]] == FARM[idx[t]] and abs(int(NUM[idx[r]]) - int(NUM[idx[t]])) <= 1}
            vis = TRAIN & ~np.array([rid(i) in te for i in range(N)])
            corr = fit(dsg.drop(index=list(excl | te), errors="ignore"), vis)
            o = apply(G, te, corr, vis); m = G.rec.isin(te).values; new[m] = o[m]
        score(v, sd, G, new); print("done", v, sd, flush=True)
for sd, G in R.groupby(R.seed.astype(str)):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True).rename(columns={"SG2": "sg"})
    G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()); new = G.sg.values.copy(); esg = G.groupby("rec").sg.mean()
    for k, Gk in G.groupby("fold"):
        te = set(Gk.rec); hs = hidEV[int(k)] | te; vis = TRAIN & ~np.array([rid(i) in hs for i in range(N)])
        corr = fit(esg.drop(index=[r for r in esg.index if r in hs]), vis)
        o = apply(G, te, corr, vis); m = G.rec.isin(te).values; new[m] = o[m]
    score("EV", sd, G, new); print("done EV", sd, flush=True)
S = pd.DataFrame(res); pd.set_option("display.width", 200); print(S.round(3).to_string(index=False))
def bootP(v):
    s = pd.concat(blocks[v], axis=1).mean(axis=1).values; ix = rng.integers(0, len(s), (20000, len(s))); return float((s[ix].mean(1) >= 0).mean())
pw = {v: bootP(v) for v in ("P2LOO", "EL1", "EV")}; print("block bootstrap P(worse):", {k: round(x, 4) for k, x in pw.items()})
ok = bool((S["all"] < 0).all()) and pw["EV"] < .025 and pw["P2LOO"] < .025
print(f"cells improved {int((S['all'] < 0).sum())}/{len(S)}"); print("DS6 DS5-static on other validators:", "ADOPT" if ok else "FAIL")
