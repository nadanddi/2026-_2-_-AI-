# -*- coding: utf-8 -*-
"""HR2 (2026-10-14 집 클로드; 사용자: 'RNN 이 EC 크기만을 목표로 학습할 수 없나').  서술(비인과 상한, 판정 아님).  EC.
HR1(6.886): 고EC 31일만 → GRU ≈ 상수.  여기선 모든 학습 정답 날(잠금 제외)로, 목표 = 하루 수준 잔차 d = 실제 하루 평균 − 기본(cp2 base_oof) 하루 평균
  (기본이 못 맞힌 '크기'만).  비평 6.870 이 '안 해 본 형태'로 짚은 잔차 목표 GRU.
입력: 자기 기록 0~23시 입력 14종 + 기본 예측 24시간(전 시각 = 비인과 상한).  폴드: 농장 x 번호 10개 묶음(DIAG10 꼴), 묶음 안 기록 + 같은 농장 ±1 숨김.
모델: GRU 1층 숨김 16·32 (시드 3 평균, 하루 평균 잔차 1개 출력, weight decay), 비교: 0 보정, 릿지(하루 요약 + 기본 평균), TabPFN v2(같은 요약).
보고: 잔차 RMSE(전체/보통/고EC/2차), 보정 후 하루 평균 RMSE 변화.  상한에서도 0 보정을 못 이기면 이 방향 닫음.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import torch
from sklearn.linear_model import RidgeCV
USE = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
A = pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz")); A = A[A.is_train & ~A.lock].sort_values(["rec", "hour"])
g = A.groupby("rec"); ok = g.base_oof.apply(lambda x: np.isfinite(x).all() and len(x) == 24) & g.y.apply(lambda x: np.isfinite(x).sum() >= 12)
RECS = sorted(ok[ok].index); A = A[A.rec.isin(RECS)]
YD = A.groupby("rec").y.mean().reindex(RECS).values; BD = A.groupby("rec").base_oof.mean().reindex(RECS).values; D = YD - BD
BASE = np.stack([A[A.rec == r].base_oof.values for r in RECS]).astype(np.float32)
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int); X = X[X.rec.isin(RECS)]
CUBE = np.stack([X.pivot(index="rec", columns="h", values=v).reindex(index=RECS, columns=range(24)).values for v in USE] + [BASE], 2).astype(np.float32)
farm = np.array([r[:3] for r in RECS]); num = np.array([int(r[4:]) for r in RECS]); hi = YD >= 1.0; p2 = num >= 179
blk = np.array([f"{f}_{n // 10}" for f, n in zip(farm, num)]); print("records", len(RECS), "high", int(hi.sum()), "folds", len(set(blk)))
with np.errstate(all="ignore"):
    S = np.concatenate([np.nanmean(CUBE, 1), np.nanmean(CUBE[:, 0:8], 1), np.nanmean(CUBE[:, 9:17], 1)], 1)
S = np.where(np.isfinite(S), S, np.nanmean(S, 0))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
class Net(torch.nn.Module):
    def __init__(s, d, hd):
        super().__init__(); s.g = torch.nn.GRU(d, hd, batch_first=True); s.o = torch.nn.Linear(hd, 1)
    def forward(s, x): return s.o(s.g(x)[0][:, -1]).squeeze(-1)
def gru(tr, te, hd):
    mu = np.nanmean(CUBE[tr].reshape(-1, CUBE.shape[2]), 0); sd = np.nanstd(CUBE[tr].reshape(-1, CUBE.shape[2]), 0) + 1e-6
    f = lambda I: torch.tensor(np.nan_to_num((CUBE[I] - mu) / sd), device=DEV)
    xt, xe = f(tr), f(te); yt = torch.tensor(D[tr], dtype=torch.float32, device=DEV); out = []
    for s in (1, 2, 3):
        torch.manual_seed(s); net = Net(CUBE.shape[2], hd).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-3); gg = torch.Generator().manual_seed(s)
        for _ in range(60):
            perm = torch.randperm(len(xt), generator=gg).to(DEV)
            for b in range(0, len(xt), 32):
                j = perm[b:b + 32]; loss = ((net(xt[j]) - yt[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(xe).cpu().numpy())
    return np.mean(out, 0)
def pfn(tr, te):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEV, random_state=0, ignore_pretraining_limits=True); m.fit(S[tr], D[tr]); return m.predict(S[te])
P = {k: np.zeros(len(RECS)) for k in ("zero", "ridge", "tabpfn", "gru16", "gru32")}
for b in sorted(set(blk)):
    te = np.where(blk == b)[0]; f_ = farm[te[0]]; lo, hi_ = num[te].min() - 1, num[te].max() + 1
    tr = np.where(~((farm == f_) & (num >= lo) & (num <= hi_)))[0]
    Z = (S - S[tr].mean(0)) / (S[tr].std(0) + 1e-9)
    P["ridge"][te] = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Z[tr], D[tr]).predict(Z[te])
    P["tabpfn"][te] = pfn(tr, te); P["gru16"][te] = gru(tr, te, 16); P["gru32"][te] = gru(tr, te, 32)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
print(f"\nday-level residual d = y_day - base_day: sd all {D.std():.3f} | normal {D[~hi].std():.3f} | high {D[hi].std():.3f} (mean high {D[hi].mean():+.3f})")
for k, v in P.items():
    e = D - v
    print(f"  {k:7s}: resid RMSE all {rm(e):.4f} | normal {rm(e[~hi]):.4f} | high {rm(e[hi]):.4f} | pass-2 {rm(e[p2]):.4f} | corr(pred, d) {np.corrcoef(v, D)[0, 1] if v.std() > 1e-9 else float('nan'):.3f}")
