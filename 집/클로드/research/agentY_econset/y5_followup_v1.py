# -*- coding: utf-8 -*-
"""agentY_econset y5: y4 후속 점검.
(1) dEC 최상위 단서 do_in_temp_m06 / do_out_temp_m06 단일 특징의 사슬 묶음 CV R2, 농장·전후반 부호 안정성, 모델 잔차와의 편상관
(2) RAMP 분류에서 전날 입력이 pred06(모델 0~6시 예측)에 더해 주는 것이 있는가
(3) 고EC 날 잔차 크기와 EC 수준 관계 (onset 날 잔차가 작은 이유)"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from y_common import *
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
D = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "y4_D.pkl"))
def gkf_splits(groups, k=5, seed=0):
    ug = np.unique(groups); rs = np.random.default_rng(seed); rs.shuffle(ug)
    fold = {g: n % k for n, g in enumerate(ug)}; f = np.array([fold[g] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]
S = D[D.p1.notna() & D.dEC.notna()].copy()
for c in ["do_in_temp_m06", "do_out_temp_m06", "p1_act_thermal_nchg", "p1_act_shade_nchg"]:
    s = S[S[c].notna()]; x = s[[c, "ec_p1"]].values; y = s.dEC.values; r2 = []; r2b = []
    for rep in range(10):
        pr = np.zeros(len(y)); pb = np.zeros(len(y))
        for a_, b_ in gkf_splits(s.grp.values, 5, rep):
            pr[b_] = LinearRegression().fit(x[a_], y[a_]).predict(x[b_])
            pb[b_] = LinearRegression().fit(x[a_][:, 1:], y[a_]).predict(x[b_][:, 1:])
        r2.append(1 - np.sum((y - pr) ** 2) / np.sum((y - pb) ** 2))
    parts = []
    for lab, m in [("F13", s.farm == "F13"), ("F47", s.farm == "F47"), ("num<179", s.num < 179), ("num>=179", s.num >= 179),
                   ("ec_p1>=.6", s.ec_p1 >= .6)]:
        if m.sum() > 8: parts.append(f"{lab} rho {spearmanr(s[m][c], s[m].dEC)[0]:+.2f}(n{m.sum()})")
    sr = s[s.res.notna() & s.pred.notna()]
    rr = spearmanr(sr[c], sr.res - LinearRegression().fit(sr[["pred"]], sr.res).predict(sr[["pred"]]))[0]
    sl = np.polyfit(s[c], s.dEC, 1)[0]
    print(f"{c:22s} CV R2 gain over ec_p1-only {np.mean(r2):+.4f} [{np.min(r2):+.4f},{np.max(r2):+.4f}] slope {sl:+.4f}/unit | {'; '.join(parts)} | rho(res|pred) {rr:+.3f} n{len(sr)}")
# (2) RAMP with pred06 vs pred06 + prev features (top in-fold 5)
L = D[D.p1.notna() & D.pred06.notna()].copy(); lab = ((L.dEC >= 0.08) & (L.ec >= 0.7)).values.astype(int)
PRV = [c for c in L.columns if c.startswith(("p1_", "dp_", "do_")) and L[c].notna().mean() > .3 and L[c].nunique() > 2]
def prep(Xtr, Xte):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9; return (Xtr - mu) / sd, (Xte - mu) / sd
for nm_, extra in [("pred06", []), ("pred06+prev top5", PRV)]:
    aucs = []
    for rep in range(5):
        pr = np.zeros(len(lab))
        for a_, b_ in gkf_splits(L.grp.values, 5, rep):
            cols = ["pred06"]
            if extra:
                Xt0, _ = prep(L[extra].values[a_], L[extra].values[b_])
                sc = [abs(roc_auc_score(lab[a_], Xt0[:, j]) - .5) for j in range(Xt0.shape[1])]
                cols += [extra[j] for j in np.argsort(sc)[::-1][:5]]
            Xt, Xv = prep(L[cols].values[a_], L[cols].values[b_])
            pr[b_] = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(Xt, lab[a_]).predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(lab, pr))
    print(f"RAMP {nm_:18s} AUC {np.mean(aucs):.3f} [{np.min(aucs):.3f},{np.max(aucs):.3f}] (pos {lab.sum()}/{len(lab)})")
# (3) residual by EC level among OOF days
dd = D[D.pred.notna()]
for lo, hi in [(0, .6), (.6, .8), (.8, 1.0), (1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 9)]:
    m = (dd.ec >= lo) & (dd.ec < hi)
    print(f"truth day EC [{lo},{hi}): n {m.sum():3d} mean res {dd[m].res.mean():+.3f}  linked-prev {dd[m].p1.notna().mean():.2f}")
