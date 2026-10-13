# -*- coding: utf-8 -*-
"""EE1 (2026-10-13 집 클로드; 사용자: '에피소드 끝 신호 찾아봐').  서술·탐색(판정 아님).  EC.
DS1Z(6.821): HMM 의 보통 동 손해 = 고EC 에피소드 끝 직후(F13_156·159, F47_161)·시작 직전.  끝을 알리는 신호가 입력에 있나?
쌍: 학습 기록의 깨끗한 같은 동 연속일(진단용 연결 truth a1 & DM11 차 1 — 상한, 규정판 아님), 전날 실제 고EC(하루 평균 ≥ 1.0).
  END = 오늘 < 1.0,  CONT = 오늘 ≥ 1.0.
특징(모두 오늘 + 그 동 전날 — 평가에선 같은 ID 앞 기록·학습 기록으로 가능한 종류): 입력 14종 × 시간 창(0–5, 6–11, 12–17, 18–23, 하루) 오늘 값과 '오늘 − 전날',
  EC 모델 DIAG10 OOF 예측(시드 평균) 하루 평균·0–11 평균·23시−0시·오늘−전날.  하루 전체 창은 비인과(상한) — 표시.
통계: 특징별 AUC(END vs CONT), 다중 비교 기준 = 라벨 섞기 2000회의 최대 |AUC−.5| 95% 분위.  HMM 손해 기록의 값 표시.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CPREV = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
VARS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
WIN = {"h00_05": range(0, 6), "h06_11": range(6, 12), "h12_17": range(12, 18), "h18_23": range(18, 24), "day*": range(24)}
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
CUBE = {v: _X.pivot(index="rec", columns="h", values=v).reindex(columns=range(24)) for v in VARS}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; O["rec"] = O.row_id.str[:7]
SG = O.groupby(["rec", "hour"]).sg.mean().unstack().reindex(columns=range(24))
def feats(r):
    f = {}
    for v in VARS:
        row = CUBE[v].loc[r].values if r in CUBE[v].index else np.full(24, np.nan)
        for w, hs in WIN.items():
            with np.errstate(all="ignore"): f[f"{v}|{w}"] = np.nanmean(row[list(hs)])
    s = SG.loc[r].values if r in SG.index else np.full(24, np.nan)
    f["sg|day*"] = np.nanmean(s); f["sg|h00_11"] = np.nanmean(s[:12]); f["sg|slope23-0*"] = s[23] - s[0]; f["sg|h00"] = s[0]
    return f
rows = []
for r, p in CPREV.items():
    i, j = idx[r], idx[p]
    if not (TRAIN[i] and TRAIN[j] and np.isfinite(DM[i]) and np.isfinite(DM[j])): continue
    if DM[j] < 1.0: continue
    ft, fp = feats(r), feats(p)
    d = {**{k: v for k, v in ft.items()}, **{f"Δ{k}": ft[k] - fp[k] for k in ft}}
    rows.append(dict(rec=r, prev=p, end=int(DM[i] < 1.0), y=DM[i], yprev=DM[j], **d))
P = pd.DataFrame(rows); F = [c for c in P.columns if "|" in c]
print(f"pairs with previous day truly high: {len(P)} | END {int(P.end.sum())} | CONT {int((1 - P.end).sum())}")
print("END pairs:", P[P.end == 1][["rec", "prev", "yprev", "y", "sg|day*", "Δsg|day*"]].round(3).to_string(index=False))
au = {}
for c in F:
    m = P[c].notna()
    if m.sum() >= 8 and P.end[m].nunique() == 2: au[c] = roc_auc_score(P.end[m], P[c][m])
A = pd.Series(au); rng = np.random.default_rng(20261129); null = []
X = P[A.index].values; yv = P.end.values
for _ in range(2000):
    yp = rng.permutation(yv); best = 0
    for k in range(X.shape[1]):
        m = np.isfinite(X[:, k])
        if yp[m].min() == yp[m].max(): continue
        best = max(best, abs(roc_auc_score(yp[m], X[m, k]) - .5))
    null.append(best)
thr = float(np.quantile(null, .95)); print(f"\nfeatures {len(A)} | null 95% of max |AUC-.5| = {thr:.3f}  (* = full-day / non-causal window)")
top = (A - .5).abs().sort_values(ascending=False).head(15)
print(pd.DataFrame(dict(AUC=A[top.index], passes=(top > thr))).round(3).to_string())
for r in ("F13_156", "F13_159", "F47_161", "F13_233"):
    print(f"\n{r}: prev {CPREV.get(r)} | prev truly high {r in set(P.rec)}", "" if r not in set(P.rec) else P[P.rec == r][list(top.index[:6])].round(3).to_dict("records"))
P.to_csv(os.path.join(HERE, "ee1_pairs_v1.csv"), index=False)
