# -*- coding: utf-8 -*-
"""재분석 18 (2026-10-01, 집 클로드): 배지-공기 "분리" 날을 시간 모양으로 알아챌 수 있는가 — 탐색(비인과 상한).

대상: G_C2 DIAG10 하루 오프셋 e_day (400일). 날 단위 5일 묶음 10겹(DIAG10과 같은 묶음) 교차적합.
특징 집합:
  A 하루 평균 14개 (E2 재현, 기준)
  B 24시간 모양: in_temp·in_hum·in_co2·heat·therm·vent·circ·shade 각 24값 (192)
  C B + 같은 날짜 형제 기록(같은 온실 우선, 없으면 다른 온실)과의 시간별 차이 (192) + 형제 유무
  D C의 인과판: 각 날의 0~h시만 쓰는 대신, 여기서는 상한 확인이 목적이라 생략(상한이 의미 있을 때만 다음 단계)
모델: HistGB(깊이 3, 150회, lr .05, 잎 최소 15) 3시드 + Ridge(표준화, alpha 10). 지표: 교차적합 R², |e|>0.7 판별 AUC.
결과 local/re18_decouple_shape.txt
"""
import env  # noqa
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from harness import load
from anal_q1_errors import diag_folds

out = open(env.LOCAL + "/re18_decouple_shape.txt", "w", encoding="utf-8")
def p(x): print(x); out.write(x + "\n")

_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
t = lab.in_temp.values; g = np.where(np.isnan(t), 1, np.clip((t - 8) / 2, 0, 1)); s = "DIAG10"
base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], 0)
cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], 0)
pfn = np.load(env.LOCAL + f"/web_tabpfn_v2_temp_{s}.npy").mean(0)
lab = lab.assign(e=(0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn - lab.sub_temp)
lab = lab.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
V = {"in_temp": "tin", "in_hum": "hum", "in_co2": "co2", "act_heating": "heat", "act_thermal": "therm",
     "act_vent": "vent", "act_circfan": "circ", "act_shade": "shade"}
keys = lab[["farm", "day"]].drop_duplicates().values
piv = {v: lab.pivot_table(index=["farm", "day"], columns="hour", values=c) for c, v in V.items()}
idx = pd.MultiIndex.from_arrays([keys[:, 0], keys[:, 1].astype(int)], names=["farm", "day"])
B = pd.concat([piv[v].reindex(idx).add_prefix(v + "_") for v in V.values()], axis=1)
e_day = lab.groupby(["farm", "day"]).e.mean().reindex(idx)
meanA = lab.groupby(["farm", "day"]).agg(**{f"{c}_m": (c, "mean") for c in list(V) + ["out_temp", "out_hum", "out_rad", "out_wspd", "act_co2", "act_fog"]}).reindex(idx)

# 형제: 외기 4변수 24시간 완전 일치
W = lab.groupby(["farm", "day"]).apply(lambda q: tuple(np.round(q[["out_temp", "out_hum", "out_rad", "out_wspd"]].values.ravel(), 1)) if len(q) == 24 else None).dropna()
by = {}
for k, sg in W.items():
    by.setdefault(sg, []).append(k)
sibling = {}
for k in idx:
    sg = W.get(k)
    cand = [x for x in by.get(sg, []) if x != k] if sg is not None else []
    same = [x for x in cand if x[0] == k[0]]
    sibling[k] = (same or cand or [None])[0]
Cdiff = pd.DataFrame(np.nan, index=idx, columns=["d_" + c for c in B.columns])
for k in idx:
    sk = sibling[k]
    if sk is not None:
        Cdiff.loc[k] = B.loc[k].values - B.loc[sk].values
C = pd.concat([B, Cdiff], axis=1).assign(has_sib=[float(sibling[k] is not None) for k in idx])
F47 = pd.Series((idx.get_level_values(0) == "F47").astype(float), index=idx, name="f47")
sets = {"A 하루평균": meanA.assign(f47=F47), "B 24시간 모양": B.assign(f47=F47), "C 모양+형제차": C.assign(f47=F47)}
p(f"날 {len(idx)}, 형제 있는 날 {sum(v is not None for v in sibling.values())} (같은 온실 {sum(v is not None and v[0]==k[0] for k, v in sibling.items())})")

fold_of = {}
for i, fd in enumerate(diag_folds(lab)):
    for f, ds in fd.items():
        for d in ds:
            fold_of[(f, d)] = i
fk = np.array([fold_of[k] for k in idx])
y = e_day.values
big = (np.abs(y) > 0.7).astype(int)

def cv(X, model_fn):
    pr = np.zeros(len(y))
    for k in range(10):
        tr, te = fk != k, fk == k
        m = model_fn()
        Xtr, Xte = X[tr], X[te]
        m.fit(Xtr, y[tr])
        pr[te] = m.predict(Xte)
    return pr

for name, X in sets.items():
    Xv = X.values.astype(float)
    r2s, aucs = [], []
    for sd in (0, 1, 2):
        pr = cv(Xv, lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=15, random_state=sd))
        r2s.append(1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        aucs.append(roc_auc_score(big, np.abs(pr)))
    Xr = np.nan_to_num((Xv - np.nanmean(Xv, 0)) / (np.nanstd(Xv, 0) + 1e-9))
    prr = cv(Xr, lambda: Ridge(alpha=10.0))
    r2r = 1 - ((y - prr) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    p(f"{name:14s} 특징 {Xv.shape[1]:3d} | GBM R² {np.round(r2s,3)} | |e|>0.7 AUC {np.round(aucs,3)} | Ridge R² {r2r:+.3f}")

# 서술: 분리 날 vs 정상 날 시간별 평균 모양 차 (heat, therm, vent, circ, tin)
p("\n시간별 평균 (과대 e>0.7 / 과소 e<-0.7 / 정상 |e|<0.2) — 서술용")
grp = np.where(y > 0.7, "과대", np.where(y < -0.7, "과소", np.where(np.abs(y) < 0.2, "정상", "기타")))
for v in ("tin", "heat", "therm", "vent", "circ"):
    p(f"  [{v}]")
    for gname in ("과대", "과소", "정상"):
        m = B.loc[grp == gname, [f"{v}_{h}" for h in range(24)]].mean().values
        p(f"    {gname}(n={int((grp==gname).sum())}): " + " ".join(f"{x:5.1f}" for x in m[::2]))
