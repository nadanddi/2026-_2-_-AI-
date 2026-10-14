# -*- coding: utf-8 -*-
"""HR1 (2026-10-14 집 클로드; 사용자: '고EC 날 24시 데이터만 RNN 으로 학습하면? 적은 자료에도 쓸 모델 없나').  서술(비인과 상한, 판정 아님).  EC.
질문: 고EC 날(정답 하루 평균 >= 1.0, 잠금 제외)만으로 학습한 작은 모델이 '그날 고EC 크기'를 입력에서 읽을 수 있나.
  HX9(6.797, 사람이 만든 변수 300 → 상한 0)·MX2(6.471, 고EC 전용 나무 FAIL) 와 다른 점: 원본 24시간 순서를 보는 GRU, 소표본 전용 TabPFN, 릿지, 가우시안 과정.
설계: 에피소드 하나 빼기(el1e_episodes_v1.csv 에피소드; 에피소드 없는 기록은 자기 혼자 묶음) — 같은 에피소드 기록은 함께 숨김.
  목표 = 하루 평균 EC(그리고 시각별 EC RMSE).  입력 = 자기 기록 0~23시 입력 14종(전 시각 = 비인과 상한; 이게 안 되면 인과판은 볼 필요 없음).
  기준: (a) 학습 고EC 평균 상수, (b) 기본 모델 OOF(cp2 base_oof, 모든 날로 학습한 지금 기본).  모델: 릿지(하루 요약 42특징), GP(RBF+잡음), TabPFN(요약), GRU(1층 숨김 8·16, 시드 3 평균).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import torch
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
USE = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
A = pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz")); A = A[A.is_train & ~A.lock]
dm = A.groupby("rec").y.mean(); HI = sorted(dm[dm >= 1.0].index); print("high-EC records (lock excluded):", len(HI))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int); X = X[X.rec.isin(HI)]
CUBE = np.stack([X.pivot(index="rec", columns="h", values=v).reindex(index=HI, columns=range(24)).values for v in USE], 2).astype(np.float32)
Y = np.stack([A[A.rec == r].sort_values("hour").y.values for r in HI]); YD = np.nanmean(Y, 1)
BD = np.array([A[A.rec == r].base_oof.mean() for r in HI])
E = pd.read_csv(os.path.join(HERE, "el1e_episodes_v1.csv")); ep = dict(zip(E.rec, E.ep)); G = np.array([ep.get(r, r) for r in HI])
print("episodes/groups:", len(set(G)))
def summ(C):
    with np.errstate(all="ignore"):
        return np.concatenate([np.nanmean(C, 1), np.nanmean(C[:, 0:8], 1), np.nanmean(C[:, 9:17], 1)], 1)
S = summ(CUBE); S = np.where(np.isfinite(S), S, np.nanmean(S, 0))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
class Net(torch.nn.Module):
    def __init__(s, d, hdim):
        super().__init__(); s.g = torch.nn.GRU(d, hdim, batch_first=True); s.o = torch.nn.Linear(hdim, 1)
    def forward(s, x): return s.o(s.g(x)[0][:, -1]).squeeze(-1)
def gru(tr, te, hdim):
    mu = np.nanmean(CUBE[tr].reshape(-1, len(USE)), 0); sd = np.nanstd(CUBE[tr].reshape(-1, len(USE)), 0) + 1e-6
    f = lambda I: torch.tensor(np.nan_to_num((CUBE[I] - mu) / sd), device=DEV)
    xt, xe = f(tr), f(te); ym = YD[tr].mean(); yt = torch.tensor(YD[tr] - ym, dtype=torch.float32, device=DEV); out = []
    for s in (1, 2, 3):
        torch.manual_seed(s); net = Net(len(USE), hdim).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
        for _ in range(150): loss = ((net(xt) - yt) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(xe).cpu().numpy() + ym)
    return np.mean(out, 0)
def pfn(tr, te):
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(device=DEV, random_state=0); m.fit(S[tr], YD[tr]); return m.predict(S[te])
P = {k: np.zeros(len(HI)) for k in ("const", "ridge", "gp", "tabpfn", "gru8", "gru16")}
for g in sorted(set(G)):
    te = np.where(G == g)[0]; tr = np.where(G != g)[0]
    P["const"][te] = YD[tr].mean()
    P["ridge"][te] = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit((S[tr] - S[tr].mean(0)) / (S[tr].std(0) + 1e-9), YD[tr]).predict((S[te] - S[tr].mean(0)) / (S[tr].std(0) + 1e-9))
    k = ConstantKernel() * RBF(length_scale=np.ones(S.shape[1]) * 5) + WhiteKernel()
    Z = (S - S[tr].mean(0)) / (S[tr].std(0) + 1e-9); P["gp"][te] = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0).fit(Z[tr], YD[tr]).predict(Z[te])
    P["tabpfn"][te] = pfn(tr, te); P["gru8"][te] = gru(tr, te, 8); P["gru16"][te] = gru(tr, te, 16)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
print(f"\nday-mean EC of high records: mean {YD.mean():.3f} sd {YD.std():.3f}")
print(f"  current base model (all-day trained OOF): RMSE {rm(BD - YD):.4f}  corr {np.corrcoef(BD, YD)[0, 1]:.3f}")
for k, v in P.items(): print(f"  {k:7s} (high-only, leave-episode-out): RMSE {rm(v - YD):.4f}  corr {np.corrcoef(v, YD)[0, 1] if v.std() > 1e-9 else float('nan'):.3f}")
for k in ("tabpfn", "gru16", "ridge"):
    b = .5 * BD + .5 * P[k]; print(f"  50/50 base + {k}: RMSE {rm(b - YD):.4f}")
