# -*- coding: utf-8 -*-
"""HR4 (2026-10-14 집 클로드; 사용자: 'TabPFN 말고 다른 것들도 파생 변수 넣어서 해봤어?').  서술(비인과 상한, 판정 아님).  EC.
HR2/HR3 와 같은 자료·목표(하루 수준 잔차)·폴드.  모델 x 특징 묶음 채우기:
  릿지·GP(RBF+잡음, ARD 없음)·ET(300, min_leaf 5)·LightGBM(200, lr .03, leaves 7, min_child 10) x {S0 요약 45, S1 = S0 + 파생 59}.
  GRU: 24시간 순서 + (a) 없음 [HR2 재현용 gru16]  (b) S1 파생 59 를 고정 입력으로 마지막 층에 붙임.
TabPFN 결과는 HR3 저장 예측(hr3_preds_v1.csv)으로 같은 표에 같이 보고.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "hr3_dayresid_derived_feats_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "hr3_dayresid_derived_feats_v1.py")
exec(src[:src.index("def pfn(")])
__file__ = os.path.join(HERE, "hr4_dayresid_models_x_feats_v1.py")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
DER = S1[:, S.shape[1]:]
class Net2(torch.nn.Module):
    def __init__(s, d, hd, ds):
        super().__init__(); s.g = torch.nn.GRU(d, hd, batch_first=True); s.o = torch.nn.Linear(hd + ds, 1)
    def forward(s, x, z): return s.o(torch.cat([s.g(x)[0][:, -1], z], 1)).squeeze(-1)
def gru(tr, te, static):
    mu = np.nanmean(CUBE[tr].reshape(-1, CUBE.shape[2]), 0); sd = np.nanstd(CUBE[tr].reshape(-1, CUBE.shape[2]), 0) + 1e-6
    f = lambda I: torch.tensor(np.nan_to_num((CUBE[I] - mu) / sd), device=DEV)
    zs = DER if static else np.zeros((len(RECS), 0)); zm, zsd = zs[tr].mean(0), zs[tr].std(0) + 1e-9
    g_ = lambda I: torch.tensor(((zs[I] - zm) / zsd).astype(np.float32), device=DEV)
    xt, xe, zt, ze = f(tr), f(te), g_(tr), g_(te); yt = torch.tensor(D[tr], dtype=torch.float32, device=DEV); out = []
    for s in (1, 2, 3):
        torch.manual_seed(s); net = Net2(CUBE.shape[2], 16, zs.shape[1]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-3); gg = torch.Generator().manual_seed(s)
        for _ in range(60):
            perm = torch.randperm(len(xt), generator=gg).to(DEV)
            for b in range(0, len(xt), 32):
                j = perm[b:b + 32]; loss = ((net(xt[j], zt[j]) - yt[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): out.append(net(xe, ze).cpu().numpy())
    return np.mean(out, 0)
FS = {"S0": S, "S1": S1}; P = {}
for b in sorted(set(blk)):
    te = np.where(blk == b)[0]; f_ = farm[te[0]]; lo, hi_ = num[te].min() - 1, num[te].max() + 1
    tr = np.where(~((farm == f_) & (num >= lo) & (num <= hi_)))[0]
    for fs, SS in FS.items():
        Z = (SS - SS[tr].mean(0)) / (SS[tr].std(0) + 1e-9)
        P.setdefault(f"ridge_{fs}", np.zeros(len(RECS)))[te] = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(Z[tr], D[tr]).predict(Z[te])
        k = ConstantKernel() * RBF(length_scale=10.0) + WhiteKernel()
        P.setdefault(f"gp_{fs}", np.zeros(len(RECS)))[te] = GaussianProcessRegressor(k, normalize_y=True).fit(Z[tr], D[tr]).predict(Z[te])
        P.setdefault(f"et_{fs}", np.zeros(len(RECS)))[te] = ExtraTreesRegressor(300, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(SS[tr], D[tr]).predict(SS[te])
        P.setdefault(f"lgb_{fs}", np.zeros(len(RECS)))[te] = lgb.LGBMRegressor(n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=10, subsample=.8, subsample_freq=1, colsample_bytree=.8, random_state=0, verbose=-1).fit(SS[tr], D[tr]).predict(SS[te])
    P.setdefault("gru_seq", np.zeros(len(RECS)))[te] = gru(tr, te, False); P.setdefault("gru_seq+S1der", np.zeros(len(RECS)))[te] = gru(tr, te, True)
H3 = pd.read_csv(os.path.join(HERE, "hr3_preds_v1.csv")).set_index("rec").reindex(RECS); P["tabpfn_S0"] = H3.s0.values; P["tabpfn_S1"] = H3.s1.values
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
print(f"zero correction: all {rm(D):.4f} | normal {rm(D[~hi]):.4f} | high {rm(D[hi]):.4f} | pass-2 {rm(D[p2]):.4f}")
for k in sorted(P):
    e = D - P[k]; print(f"  {k:14s}: all {rm(e):.4f} | normal {rm(e[~hi]):.4f} | high {rm(e[hi]):.4f} | pass-2 {rm(e[p2]):.4f} | corr {np.corrcoef(P[k], D)[0, 1]:.3f}")
