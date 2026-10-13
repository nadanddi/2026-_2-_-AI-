# -*- coding: utf-8 -*-
"""HX9 근권 온도 v6 (2026-10-13 집 클로드, 사후 점검 — 판정 아님, 진단).
질문: 고EC 사슬 안 연속일에서 'ln EC 변화 ~ Δ예측 배지온(pst, 입력 대리)' 관계(v2: r .53, v4: Δ잔차 r .55)가
      전날 정답 앵커(진단: 링크가 정답 연속성으로 고른 것)에 얹었을 때 앵커만 쓰는 것보다 실제로 나은가?
비교(하루 평균 EC, hi10 기록 중 clean hi10 전임자가 있는 것; 모델 OOF 있으면 sg 와도 비교):
  M0 현재 모델 sg_mean / M1 전날 평균 그대로(persistence) / M2 전날 × exp(.02·Δpst) (센서 효과만, 고정)
  M3 전날 × exp(β·Δpst), β 는 사슬 하나 빼기로 적합 / M4 M3 와 같지만 Δst(정답 배지온, 진단 상한)
결과 보기 전 예측: M3 가 M1 보다 낫다면 그 이득은 다른 id 링크(n 10)에 몰리고 같은 id 링크에서는 ≈0 (v3 결과에서 유추).
 M2(센서만)는 M1 과 거의 같다(Δpst SD ~1.4℃ → ±3%).
 이 비교의 '앵커'는 정답 연속성으로 고른 링크라 평가 행에서 그대로 쓸 수 없음(입력만으로 링크를 찾아야 사용 가능).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import env  # noqa: F401
import numpy as np, pandas as pd
pd.set_option("display.width", 250)
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
RNG = np.random.default_rng(6)
V = pd.read_csv(os.path.join(HERE, "vars_v1.csv"), index_col="rec")
R = pd.read_csv(os.path.join(HX9, "data", "records.csv"), index_col="rec")

def run(L, lab, tvar):
    y = L.ec_mean.values; ya = R.loc[L.prev_rec, "ec_mean"].values
    dx = V.loc[L.index, tvar].values - V.loc[L.prev_rec, tvar].values
    g = L.ep_id.values if "ep_id" in L and L.ep_id.notna().all() else pd.factorize(V.loc[L.index, "linkcomp"])[0]
    dl = np.log(y) - np.log(ya)
    pred3 = np.zeros(len(L)); betas = []
    for u in np.unique(g):
        te = g == u; tr = ~te
        b = np.sum(dx[tr] * dl[tr]) / np.sum(dx[tr] ** 2)   # through origin: no drift term (persistence + temperature)
        betas.append(b); pred3[te] = ya[te] * np.exp(b * dx[te])
    P = {"M1_persist": ya, "M2_sensor02": ya * np.exp(0.02 * dx), "M3_beta_loco": pred3}
    if L.has_sg.all(): P["M0_model"] = L.sg_mean.values
    sf = L.prev_same_farm.values == 1
    out = []
    for k, p in P.items():
        e = (p - y) ** 2
        out.append(dict(set=lab, tvar=tvar, model=k, n=len(L), rmse=np.sqrt(e.mean()), rmse_samefarm=np.sqrt(e[sf].mean()),
                        rmse_crossfarm=np.sqrt(e[~sf].mean()) if (~sf).any() else np.nan))
    e1 = (P["M1_persist"] - y) ** 2; e3 = (P["M3_beta_loco"] - y) ** 2
    # chain bootstrap of RMSE difference M3-M1
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; bs = []
    for _ in range(4000):
        s = RNG.choice(ug, len(ug), replace=True); ii = np.concatenate([idx[u] for u in s])
        bs.append(np.sqrt(e3[ii].mean()) - np.sqrt(e1[ii].mean()))
    print(f"{lab:28s} {tvar:9s} n {len(L):3d}: beta LOCO range [{min(betas):+.3f},{max(betas):+.3f}]; RMSE M3-M1 {np.sqrt(e3.mean())-np.sqrt(e1.mean()):+.4f} "
          f"CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]; M3 better on {(e3<e1).sum()}/{len(L)} "
          f"(same-farm {(e3<e1)[sf].sum()}/{sf.sum()}, cross {(e3<e1)[~sf].sum()}/{(~sf).sum()})")
    return out

rows = []
Lhi = R[(R.hi10 == 1) & (R.prev_hi10 == 1)].copy()
Lhi_sg = Lhi[(Lhi.has_sg == 1) & np.array([R.at[p, "has_sg"] == 1 for p in Lhi.prev_rec])]
Lno = R[(R.hi10 == 0) & (R.prev_hi10 == 0) & R.prev_rec.notna()].copy()
Lno["ep_id"] = np.nan
for tv in ["pst_mean", "pst_min", "st_mean"]:
    rows += run(Lhi, "hi10->hi10 (all)", tv)
    rows += run(Lhi_sg.assign(), "hi10->hi10 (OOF both)", tv)
    rows += run(Lno, "normal->normal", tv)
T = pd.DataFrame(rows).drop_duplicates(["set", "tvar", "model"])
print(T.round(4).to_string(index=False))
T.to_csv(os.path.join(HERE, "anchor_v6.csv"), index=False, encoding="utf-8-sig")
