# -*- coding: utf-8 -*-
"""HR3 (2026-10-14 집 클로드; 사용자: '입력 재료에 파생 변수도 더 넣을 만하지 않아?').  서술(비인과 상한, 판정 아님).  EC.
HR2(6.887)와 같은 자료·목표(하루 수준 잔차 d = 실제 하루 평균 − 기본 OOF 하루 평균)·폴드(농장 x 번호 10, 같은 농장 ±1 숨김).
특징: S0 = HR2 요약 45(15재료 x 하루/밤 0~7/낮 9~16 평균);  S1 = S0 + 하루 안 파생(물리 의미, 결과 보고 고르지 않음):
  변수별 하루 max·min·std(14 x 3), VPD 낮·밤 평균, 일사 합, 일사 x VPD 낮 평균, 실내−외기 온도차 낮·밤, 구동기 가동 시간(난방·환기·차광·보온·CO2·포그·유동팬 > 0 인 시각 수),
  보온커튼 닫힘 시각(오후 첫 > 50)·열림 시각(오전 마지막 > 50), CO2 아침 최고 − 오후 최저, 습도 낮 최저.
모델: TabPFN v2 (random_state 0·1·2 각각 보고 + 평균), 릿지(S1).  HX9(6.797: 고EC 크기 300변수 상한 0)와 달리 목표에 보통 날 포함.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import torch
from sklearn.linear_model import RidgeCV
src = open(os.path.join(HERE, "hr2_dayresid_gru_all_days_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "hr2_dayresid_gru_all_days_v1.py")
exec(src[:src.index("DEV = ")])
__file__ = os.path.join(HERE, "hr3_dayresid_derived_feats_v1.py")
C = {v: CUBE[:, :, i] for i, v in enumerate(USE)}
def es(T, RH): return 0.6108 * np.exp(17.27 * T / (T + 237.3)) * (1 - np.clip(RH, 0, 100) / 100)
with np.errstate(all="ignore"):
    vpd = es(C["in_temp"], C["in_hum"]); day = slice(9, 17); night = slice(0, 8)
    F = [np.nanmax(CUBE[:, :, :14], 1), np.nanmin(CUBE[:, :, :14], 1), np.nanstd(CUBE[:, :, :14], 1),
         np.nanmean(vpd[:, day], 1)[:, None], np.nanmean(vpd[:, night], 1)[:, None], np.nansum(C["out_rad"], 1)[:, None],
         np.nanmean((C["out_rad"] * vpd)[:, day], 1)[:, None],
         np.nanmean((C["in_temp"] - C["out_temp"])[:, day], 1)[:, None], np.nanmean((C["in_temp"] - C["out_temp"])[:, night], 1)[:, None]]
    for v in ("act_heating", "act_vent", "act_shade", "act_thermal", "act_co2", "act_fog", "act_circfan"): F.append(np.nansum(C[v] > 0, 1)[:, None].astype(float))
    th = C["act_thermal"]; close = np.array([next((h for h in range(12, 24) if th[i, h] > 50), np.nan) for i in range(len(th))]); opn = np.array([max([h for h in range(0, 12) if th[i, h] > 50], default=np.nan) for i in range(len(th))])
    F += [close[:, None], opn[:, None], (np.nanmax(C["in_co2"][:, 5:11], 1) - np.nanmin(C["in_co2"][:, 12:18], 1))[:, None], np.nanmin(C["in_hum"][:, day], 1)[:, None]]
S1 = np.concatenate([S] + F, 1); S1 = np.where(np.isfinite(S1), S1, np.nanmean(S1, 0)); S1 = np.where(np.isfinite(S1), S1, 0)
print("features S0", S.shape[1], "S1", S1.shape[1])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
def pfn(SS, tr, te, rs):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEV, random_state=rs, ignore_pretraining_limits=True); m.fit(SS[tr], D[tr]); return m.predict(SS[te])
P = {k: np.zeros(len(RECS)) for k in ["S0_rs0", "S0_rs1", "S0_rs2", "S1_rs0", "S1_rs1", "S1_rs2", "ridge_S1"]}
for b in sorted(set(blk)):
    te = np.where(blk == b)[0]; f_ = farm[te[0]]; lo, hi_ = num[te].min() - 1, num[te].max() + 1
    tr = np.where(~((farm == f_) & (num >= lo) & (num <= hi_)))[0]
    for rs in (0, 1, 2):
        P[f"S0_rs{rs}"][te] = pfn(S, tr, te, rs); P[f"S1_rs{rs}"][te] = pfn(S1, tr, te, rs)
    Z = (S1 - S1[tr].mean(0)) / (S1[tr].std(0) + 1e-9); P["ridge_S1"][te] = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(Z[tr], D[tr]).predict(Z[te])
P["S0_mean"] = np.mean([P[f"S0_rs{r}"] for r in range(3)], 0); P["S1_mean"] = np.mean([P[f"S1_rs{r}"] for r in range(3)], 0)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
print(f"\nzero correction: all {rm(D):.4f} | normal {rm(D[~hi]):.4f} | high {rm(D[hi]):.4f} | pass-2 {rm(D[p2]):.4f}")
for k, v in P.items():
    e = D - v; print(f"  {k:9s}: all {rm(e):.4f} | normal {rm(e[~hi]):.4f} | high {rm(e[hi]):.4f} | pass-2 {rm(e[p2]):.4f} | corr {np.corrcoef(v, D)[0, 1]:.3f}")
pd.DataFrame(dict(rec=RECS, d=D, s0=P["S0_mean"], s1=P["S1_mean"])).to_csv(os.path.join(HERE, "hr3_preds_v1.csv"), index=False)
