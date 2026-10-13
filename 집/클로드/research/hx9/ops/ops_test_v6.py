# -*- coding: utf-8 -*-
"""HX9-OPS v6 (2026-10-13 집 클로드): v5 의 사전 기준 결함 정정 + '이웃 회수(에피소드 정체성)' vs '옮겨지는 관리→EC 법칙' 구분.
사후 점검(v5 결과 본 뒤)임을 명시.
v5 결함: 기준선으로 쓴 'sg 단독 GBM' 자체가 무보정보다 크게 나쁨(DIAG10 R2 −.23, 달력 블록 −.71) → 'gain_vs_sgonly' 가 부풀려짐.
  올바른 기준선 = 무보정(현재 모델 sg 그대로; R2_vs_nocorr = 0). 이 기준선으로 보면 v5 의 usable41+sg GBM 은
  DIAG10 +.164, 달력 블록 −.182, 1차→2차 +.325 로 묶음 방식에 따라 부호가 갈림.
========================  계산 전에 적어 둔 기준 (사전 고정, v6)  ========================
기준선 = 무보정. 묶음 4종:
  DIAG10  (현재 모델의 폴드; 같은 에피소드 이웃이 학습에 들어감)
  CAL13   (DM11 13일 달력 블록; 같은 시기 이웃 제외 — DM11 은 묶음 설계에만 쓰는 서술용)
  CHAIN10 (clean 링크 연결 성분(동의 연속 기록 사슬)을 10묶음에 무작위 배정, 배정 시드 3개; 같은 사슬 이웃은 빠지지만 같은 날짜 다른 동은 학습에 남음)
  P1toP2  (1차 314 학습 → 2차 46 시험; 평가 상황과 비슷하게 같은 사슬 이웃이 학습에 있을 수 있음) + 가족 행 섞기 위약 50회
판정:
  '옮겨지는 신호' = 네 묶음 모두(CHAIN10 은 세 배정 모두) R2_vs_nocorr > 0 AND P1toP2 위약 p < .05
  '이웃 회수'     = DIAG10·P1toP2 는 > 0 이지만 CHAIN10 또는 CAL13 에서 ≤ 0  → 관리 기제가 아니라 같은 사슬·시기 기록을 알아보는 것(에피소드 지속성 가족 몫)
  'none'          = DIAG10 에서도 ≤ 0
모델: GBM(usable41+sg), GBM(noTE37+sg), GBM(작은 3개: chg_inp_heath, heat_hours, sg), 선형(chg_inp_heath + sg). 시드 0,1,2 평균.
"""
import os, sys, json
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
rng = np.random.default_rng(606)

R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec").loc[R.index]
M = pd.read_csv(os.path.join(HERE, "vars_v1_meta.csv")).set_index("variable")
VARS = [v for v in M.index if v != "op_cluster"]
USE = [v for v in VARS if M.at[v, "usable"] == "usable"]
TE = ["te_clu_hirate", "te_clu_mag", "proto_sim_hi", "knn3_hi_mag"]
USE_NOTE = [v for v in USE if v not in TE]
R = R.join(F[VARS])

# ---- clean-link connected components over the 400 training records ----
par = {r: r for r in R.index}
def find(a):
    while par[a] != a:
        par[a] = par[par[a]]; a = par[a]
    return a
for r, p in R.prev_rec.items():
    if isinstance(p, str) and p in par:
        par[find(r)] = find(p)
R["chain"] = [find(r) for r in R.index]
cs = R.chain.value_counts()
print(f"clean-link chains: {len(cs)} | size dist: {cs.value_counts().sort_index().to_dict()}")

SA = R[R.has_sg == 1].copy()
yA = SA.resid_mean.values; sgA = SA.sg_mean.values[:, None]; hm = SA.hi10.values == 1; p2m = SA.pass2.values == 1
gD = SA.fold.values.astype(int)
blk = np.floor((SA.dm11_day.values - 1) / 13); gC = np.where(np.isfinite(blk), blk, -1).astype(int)
uch = SA.chain.unique()
gCH = []
for s in range(3):
    lab = dict(zip(uch, np.random.default_rng(100 + s).permutation(len(uch)) % 10))
    gCH.append(SA.chain.map(lab).values.astype(int))

def gbm(s):
    return GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=3, random_state=s)
def impute(X, tr):
    med = np.nanmedian(X[tr], 0); med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(X), X, med)
def predict(X, y, splits, kind, seeds=(0, 1, 2)):
    p = np.full(len(y), np.nan)
    for tr, te in splits:
        Xi = impute(X, tr)
        if kind == "lin":
            A = np.column_stack([np.ones(len(y)), Xi]); p[te] = A[te] @ np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        else:
            p[te] = np.mean([gbm(s).fit(Xi[tr], y[tr]).predict(Xi[te]) for s in seeds], 0)
    return p
def grp_splits(g):
    return [(g != u, g == u) for u in np.unique(g)]
SPL = {"DIAG10": grp_splits(gD), "CAL13": grp_splits(gC), "CHAIN10_s0": grp_splits(gCH[0]), "CHAIN10_s1": grp_splits(gCH[1]),
       "CHAIN10_s2": grp_splits(gCH[2]), "P1toP2": [(~p2m, p2m)]}
MODELS = {"gbm_usable41+sg": (USE, "gbm"), "gbm_noTE37+sg": (USE_NOTE, "gbm"),
          "gbm_small3": (["chg_inp_heath", "heat_hours"], "gbm"), "lin_heath+sg": (["chg_inp_heath"], "lin")}
def r2(y, p, m):
    k = m & np.isfinite(p)
    return 1 - np.sum((y[k] - p[k]) ** 2) / np.sum(y[k] ** 2)
rows = []; PRED = {}
for sn, spl in SPL.items():
    test_mask = np.zeros(len(yA), bool)
    for _, te in spl: test_mask |= te
    for mn, (vs, kind) in MODELS.items():
        X = np.column_stack([SA[vs].values.astype(float), sgA])
        p = predict(X, yA, spl, kind)
        PRED[(sn, mn)] = p
        tm = test_mask
        rows.append(dict(split=sn, model=mn, n=int(tm.sum()), R2_nocorr_all=r2(yA, p, tm), R2_nocorr_hi=r2(yA, p, tm & hm),
                         R2_nocorr_normal=r2(yA, p, tm & ~hm), R2_nocorr_pass2=r2(yA, p, tm & p2m),
                         n_improved=int(np.sum((np.abs(yA - p) < np.abs(yA))[tm])),
                         hi_improved=int(np.sum((np.abs(yA - p) < np.abs(yA))[tm & hm])), n_hi=int((tm & hm).sum())))
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))
T.to_csv(os.path.join(HERE, "test_v6_splits.csv"), index=False, encoding="utf-8")

# ---- P1toP2 permutation null (shuffle family rows across all 360 records, keep sg & y) ----
PERM = {}
for mn in ["gbm_usable41+sg", "gbm_noTE37+sg", "gbm_small3"]:
    vs, kind = MODELS[mn]; Xf = SA[vs].values.astype(float)
    obs = r2(yA, PRED[("P1toP2", mn)], p2m)
    null = []
    for k in range(50):
        Xp = Xf[rng.permutation(len(yA))]
        null.append(r2(yA, predict(np.column_stack([Xp, sgA]), yA, SPL["P1toP2"], kind, seeds=(0,)), p2m))
    PERM[mn] = dict(obs=obs, null_mean=float(np.mean(null)), null_q95=float(np.percentile(null, 95)), p=(np.sum(np.array(null) >= obs) + 1) / 51)
    print("P1toP2 perm", mn, {a: round(float(b), 3) for a, b in PERM[mn].items()})

# ---- verdicts ----
V = {}
for mn in MODELS:
    t = T[T.model == mn].set_index("split").R2_nocorr_all
    allpos = all(t[s] > 0 for s in SPL)
    permok = PERM.get(mn, {}).get("p", np.nan) < .05 if mn in PERM else None
    if t["DIAG10"] <= 0: v = "none"
    elif allpos and (permok is None or permok): v = "transferable"
    elif (t["CAL13"] <= 0) or any(t[s] <= 0 for s in SPL if s.startswith("CHAIN")): v = "neighbor_retrieval"
    else: v = "unclear"
    V[mn] = dict(verdict=v, **{s: round(float(t[s]), 3) for s in SPL}, perm_p=PERM.get(mn, {}).get("p"))
print("\nverdicts:", json.dumps(V, indent=1, default=float))

# ---- P1toP2 per-record view for the main model ----
p = PRED[("P1toP2", "gbm_usable41+sg")]
D = SA.loc[p2m, ["hi10", "ec_mean", "sg_mean", "resid_mean", "chain", "ep_id"]].copy(); D["corr"] = p[p2m]
D["abs_before"] = D.resid_mean.abs(); D["abs_after"] = (D.resid_mean - D["corr"]).abs()
D["chain_n_pass1"] = [int(((SA.chain == c) & ~p2m).sum()) for c in D.chain]
print("\nP1toP2 pass2 records (main model):")
print(D.sort_values("abs_before", ascending=False).round(3).to_string())
# share of SSE reduction from high records
red = (D.abs_before ** 2 - D.abs_after ** 2)
print(f"SSE reduction total {red.sum():.3f}; from hi10 {red[D.hi10 == 1].sum():.3f}; from normal {red[D.hi10 == 0].sum():.3f}")
print(f"pass2 records whose chain has pass1 members: {(D.chain_n_pass1 > 0).sum()}/{len(D)}")
json.dump(dict(verdicts=V, perm=PERM, sse_red_hi=float(red[D.hi10 == 1].sum()), sse_red_normal=float(red[D.hi10 == 0].sum())),
          open(os.path.join(HERE, "test_v6_verdicts.json"), "w", encoding="utf-8"), indent=1, default=float)
