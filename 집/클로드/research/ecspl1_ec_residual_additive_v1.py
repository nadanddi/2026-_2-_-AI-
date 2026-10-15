# -*- coding: utf-8 -*-
"""ECSPL1 (2026-10-15 집 클로드; 사용자 '조사 안 한 모델·함수·구성 방식' — EC).  사전 고정(커밋 후 실행).  EC.
새 구성원: 가법 스플라인 잔차 모형(GAM 꼴) — NH(27번 EC 기본)의 남은 오차 r = y − NH 를 그날 0..h 누적 평균 입력으로 설명.
  특징(자기 기록·인과, 5점 F): cm_in_temp, cm_in_hum, cm_out_rad, cm_act_vent, cm_act_thermal, cm_act_shade, cm_act_co2, cm_in_co2, hr_sin, hr_cos
  각 특징 SplineTransformer(knots 5, degree 3, extrapolation 'linear') → StandardScaler → Ridge(alpha 10).
  적합: 검증기 v 의 폴드 k 를 예측할 때 같은 검증기의 '다른 폴드' 기록 행(그들의 NH 는 폴드 밖 예측 → 정당한 학습 자료)으로만, 3시드 NH 평균 잔차.
  섞기: pred = NH + λ·r̂ (+ 27번 FL K8 앵커는 그 뒤에 그대로),  λ ∈ {.25, .5} (본페로니 k 2 → α .0125).
기준 REF27 = NH + FL K8 앵커.
판정(EC 분야 규칙): P2LOO·EL1 3시드 2차 dSSE < 0 & (P2LOO 블록 P < .0125 OR 손해 기록-칸 0) & DIAG10 전체 손해 아님.  서술: 고EC/보통, 상위 기록 몫.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ecspl1_ec_residual_additive_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]; sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "cb2_ec_anchor_shaping_v1.py"), encoding="utf-8").read()
exec(src[:src.index("WS = tuple(")])
__file__ = os.path.join(HERE, "ecspl1_ec_residual_additive_v1.py")
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
RAW = ["in_temp", "in_hum", "out_rad", "act_vent", "act_thermal", "act_shade", "act_co2", "in_co2"]
XA = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); XA = XA[XA.row_id.str[:3].isin(["F13", "F47"])].copy()
XA["rec"] = XA.row_id.str[:7]; XA["hour"] = XA.row_id.str[8:10].astype(int); XA = XA.sort_values(["rec", "hour"])
for c in RAW: XA["cm_" + c] = XA.groupby("rec")[c].transform(lambda s: s.expanding().mean())
XA["hr_sin"], XA["hr_cos"] = np.sin(2 * np.pi * XA.hour / 24), np.cos(2 * np.pi * XA.hour / 24)
FE = ["cm_" + c for c in RAW] + ["hr_sin", "hr_cos"]
R = R.merge(XA[["rec", "hour"] + FE], on=["rec", "hour"], how="left").sort_values(["validator", "fold", "rec", "hour"]).reset_index(drop=True)
GRP = [(key, g.index.values) for key, g in R.groupby(["validator", "fold", "rec"])]
NHs = {s: (R[f"A1_{s}"].values - .1 * mt_) / .9 for s, mt_ in [(s, np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])) for s in SEEDS]}
nh_mean = np.mean([NHs[s] for s in SEEDS], 0); resid = R.y.values - nh_mean
RH = np.zeros(len(R))
for v, gv in R.groupby("validator"):
    for k in gv.fold.unique():
        tr = (R.validator == v).values & (R.fold != k).values; te = (R.validator == v).values & (R.fold == k).values
        Xt = R.loc[tr, FE].to_numpy(float); med = np.nanmedian(Xt, 0); Xt = np.where(np.isnan(Xt), med, Xt); Xv = R.loc[te, FE].to_numpy(float); Xv = np.where(np.isnan(Xv), med, Xv)
        m = make_pipeline(SplineTransformer(n_knots=5, degree=3, extrapolation="linear"), StandardScaler(), Ridge(alpha=10.0)); m.fit(Xt, resid[tr]); RH[te] = m.predict(Xv)
print("residual model fitted | corr(r_hat, resid) by validator:", {v: round(float(np.corrcoef(RH[R.validator == v], resid[R.validator == v])[0, 1]), 3) for v in R.validator.unique()}, flush=True)
P = {}
for s in SEEDS:
    for lam in (0.0, .25, .5):
        x0 = NHs[s] + lam * RH; out = x0.copy()
        for (v, k, r), ii in GRP:
            if r not in FL3 or FL3[r][0] in FOLDREC[(v, k)] or FL3[r][1] < .8 or DT[r] > 1.5: continue
            x = x0[ii]; pm = np.cumsum(x) / np.arange(1, len(x) + 1); out[ii] = x + (EC_[idx_[FL3[r][0]], 23] - pm)
        P[(lam, s)] = out
for lam in (.25, .5):
    rows, blocks, recd = [], [], []
    for v in ("P2LOO", "EL1", "DIAG10"):
        g = (R.validator == v).values; y = R.y.values[g]; p2 = R.p2.values[g]; key = (R.farm[g] + "_" + (R.day[g] // 5).astype(str)).values; hi = R.hi.values[g] if "hi" in R else np.zeros(int(g.sum()), bool)
        for s in SEEDS:
            dd = (P[(lam, s)][g] - y) ** 2 - (P[(0.0, s)][g] - y) ** 2
            rows.append(dict(val=v, seed=s, dSSE_p2=float(dd[p2].sum()), dSSE_all=float(dd.sum()), rel_p2=100 * (rm((P[(lam, s)][g] - y)[p2]) / rm((P[(0.0, s)][g] - y)[p2]) - 1),
                             rel_hi=100 * (rm((P[(lam, s)][g] - y)[p2 & hi]) / rm((P[(0.0, s)][g] - y)[p2 & hi]) - 1) if (p2 & hi).any() else np.nan,
                             rel_norm=100 * (rm((P[(lam, s)][g] - y)[p2 & ~hi]) / rm((P[(0.0, s)][g] - y)[p2 & ~hi]) - 1)))
            if v == "P2LOO": blocks.append(pd.Series(dd[p2]).groupby(key[p2]).mean())
            if v != "DIAG10":
                for r_, xx in pd.Series(dd[p2]).groupby(R.rec.values[g][p2]).sum().items(): recd.append(dict(val=v, seed=s, rec=r_, d=xx))
    S = pd.DataFrame(rows); print(f"\nlambda {lam}:"); print(S.round(3).to_string(index=False))
    main = S[S.val != "DIAG10"]; xb = pd.concat(blocks, axis=1).mean(axis=1).values
    pP = float((xb[np.random.default_rng(20270142).integers(0, len(xb), (20000, len(xb)))].mean(1) >= 0).mean()); RD_ = pd.DataFrame(recd)
    per = RD_.groupby("rec").d.sum().sort_values(); nh_ = int((RD_.groupby(["val", "seed", "rec"]).d.sum() > 1e-6).sum())
    c1 = bool((main.dSSE_p2 < 0).all()); c3 = not (S[S.val == "DIAG10"].dSSE_all.mean() > 0)
    print("  top gain records:", per.head(4).round(3).to_dict(), "| total", round(per.sum(), 3), "| without top", round(per.sum() - per.iloc[0], 3))
    print(f"ECSPL1 lambda {lam}: cells {int((main.dSSE_p2 < 0).sum())}/6 | P2LOO P {pP:.4f} | harmed {nh_} | DIAG10 {S[S.val == 'DIAG10'].dSSE_all.mean():+.3f} -> {'PASS' if c1 and (pP < .0125 or nh_ == 0) and c3 else 'FAIL'}")
