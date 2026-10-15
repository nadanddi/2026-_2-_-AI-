# -*- coding: utf-8 -*-
"""LOGO1 (2026-10-16 집 클로드; 사용자 '진행해' — 다른 온실 추운 날 빌리기를 5점형(사람이 고르지 않는 자동 절차 + 51개 온실 일반성 검증)으로).
사전 고정(커밋 후 실행).  온도.  시험대 성격: 제출 모델 판정이 아니라 '자동 닮은 온실 빌리기 절차가 온실 일반적으로 통하는가'.

대상 온실 T 하나씩:
  가림 = T 의 '하루 최저 실내온도 <= 8℃' 인 날 전부(추운 날).  T 의 학습 = 나머지 날(추운 행 경험 없음 = 추운 외삽 흉내).
  평가 = 가린 날의 실내 <= 8℃ 행 RMSE(주) + 가린 날 전체 행 RMSE(보조).
  대상 조건: 가린 날 >= 3 & 그 안 <= 8℃ 행 >= 20 & 학습 날 >= 30.  F32 는 추운 행 없어 자동 제외.
  F13·F47 이 대상일 때 서로(같은 장소·같은 날짜 쌍둥이)는 기증에서 뺌 — 같은 날짜 추운 날 정답 누설 방지.
입력(모든 온실 공통, 6b.29 와 같은 계열): in_temp·in_hum·in_co2, 시각, 하루 안 지연 1·2·3·6h(시각 < 지연이면 결측), 하루 0..h 평균·최저·최고 실내온도, 1h 변화.
  하루 안만 쓰는 이유: F13·F47 기록은 이어붙임이라 자정 넘는 지연이 틀림 → 모든 온실에 같은 규칙.
정제(6b.29 와 같음): in_temp < 0 또는 > 45 → 결측, sub_temp >= 40 행 제외, in_hum·in_co2 <= 0 → 결측.
모델: LightGBM(400 나무, lr .05, 잎 31, min_child 50, 행 .8, 열 .8, 시드 7), 목표 sub_temp, 온실 범주 특징(풀링 안).
  SELF : T 학습 날만.
  POOL : T 학습 날 + 기증 온실 전부, 기증 총 가중 = T 학습 행 수(온실마다 균등).
  SIM  : POOL 과 같되 기증 온실 가중 = exp(−(d/τ)^2), τ = 그 T 에 대한 기증 d 의 중앙값, 총 가중 = T 학습 행 수.
         d(닮음 거리) = T 학습 날에서만 계산(가린 날 정답·입력 안 봄): 실내온도 구간 [8,10),[10,12),[12,15),[15,20),[20,25),[25,∞) 별 평균(배지−실내)
         차의 RMS(양쪽 행 >= 30 인 구간만, 공통 구간 < 2 면 d = 최대) 와 하루 진폭비(배지 하루 표준편차/실내 하루 표준편차, 중앙값) 차 x T 실내 하루 표준편차 중앙값(℃ 환산)의
         제곱합 제곱근.  기증 온실 행: 하루 최저 <= 12℃ 인 날 전부 + 나머지 날 25%(시드 7) — 크기 줄이기.
판정(사전 고정, 단일 주 비교 SIM vs SELF, α .025):
  대상 온실의 2/3 이상에서 SIM 추운 행 RMSE < SELF  &  Wilcoxon 부호순위(한쪽, 로그 RMSE 비) p < .025  → '자동 빌리기 일반성 PASS'.
서술: SIM vs POOL, POOL vs SELF, 온실별 표, F13·F47 줄, 6℃ 이하 행, 기증 상위 3 온실(가중) — 사람이 아는 '닮은 곳'과 겹치는지.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u logo1_cold_transfer_testbed_v1.py
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import wilcoxon
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp", "in_hum", "in_co2"]); Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
D = X.merge(Y, on="row_id"); D["farm"] = D.row_id.str[:3]; D["day"] = D.row_id.str[4:7].astype(int); D["hour"] = D.row_id.str[8:10].astype(int)
D.loc[(D.in_temp < 0) | (D.in_temp > 45), "in_temp"] = np.nan; D.loc[D.in_hum <= 0, "in_hum"] = np.nan; D.loc[D.in_co2 <= 0, "in_co2"] = np.nan
D = D[~(D.sub_temp >= 40) & D.sub_temp.notna()].sort_values(["farm", "day", "hour"]).reset_index(drop=True)
g = D.groupby(["farm", "day"])
for c in ("in_temp", "in_hum", "in_co2"):
    for L in (1, 2, 3, 6): D[f"{c}_l{L}"] = g[c].shift(L).where(D.hour >= L)   # within-day only (rows are contiguous hours)
D["cm_t"] = g.in_temp.transform(lambda s: s.expanding().mean()); D["cmin_t"] = g.in_temp.cummin(); D["cmax_t"] = g.in_temp.cummax(); D["d1_t"] = D.in_temp - D.in_temp_l1
D["dmin"] = g.in_temp.transform("min"); D["fcat"] = D.farm.astype("category")
FEAT = ["hour", "in_temp", "in_hum", "in_co2"] + [f"{c}_l{L}" for c in ("in_temp", "in_hum", "in_co2") for L in (1, 2, 3, 6)] + ["cm_t", "cmin_t", "cmax_t", "d1_t"]
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
BANDS = [8, 10, 12, 15, 20, 25, 99]
def profile(R):
    gap = R.sub_temp - R.in_temp; b = pd.cut(R.in_temp, BANDS, right=False); pr = gap.groupby(b, observed=False).agg(["mean", "size"])
    ds = R.groupby("day").agg(ss=("sub_temp", "std"), si=("in_temp", "std")); amp = float((ds.ss / ds.si).replace([np.inf, -np.inf], np.nan).median()); sig = float(ds.si.median())
    return pr["mean"].values, pr["size"].values, amp, sig
FARMS = sorted(D.farm.unique()); rng = np.random.default_rng(7)
DON = {}
for f in FARMS:
    R = D[D.farm == f]; keep = (R.dmin <= 12).values | (rng.random(len(R)) < .25); DON[f] = R[keep]
PROF_ALL = {f: profile(D[D.farm == f]) for f in FARMS}
PRM = dict(n_estimators=400, learning_rate=.05, num_leaves=31, min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8, random_state=7, verbose=-1, n_jobs=4)
def run_target(T):
    R = D[D.farm == T]; cold_days = set(R.day[R.dmin <= 8]); te = R[R.day.isin(cold_days)]; tr = R[~R.day.isin(cold_days)]
    if len(cold_days) < 3 or (te.in_temp <= 8).sum() < 20 or tr.day.nunique() < 30: return None
    pm, pn, pa, ps = profile(tr); donors = [f for f in FARMS if f != T and not ({T, f} == {"F13", "F47"})]; dist = {}
    for f in donors:
        qm, qn, qa, _ = PROF_ALL[f]; ok = (pn >= 30) & (qn >= 30) & np.isfinite(pm) & np.isfinite(qm)
        dg = float(np.sqrt(np.mean((pm[ok] - qm[ok]) ** 2))) if ok.sum() >= 2 else np.nan; da = (pa - qa) * ps if np.isfinite(pa) and np.isfinite(qa) else np.nan
        dist[f] = float(np.sqrt(dg ** 2 + da ** 2)) if np.isfinite(dg) and np.isfinite(da) else np.nan
    dv = pd.Series(dist); dv = dv.fillna(dv.max() * 2 if dv.notna().any() else 1.0); tau = float(dv.median()); ws = np.exp(-(dv / tau) ** 2)
    out = dict(farm=T, cold_days=len(cold_days), cold_rows=int((te.in_temp <= 8).sum()), le6_rows=int((te.in_temp <= 6).sum()), top3=",".join(ws.sort_values(ascending=False).index[:3]))
    m8 = (te.in_temp <= 8).values; m6 = (te.in_temp <= 6).values; P = {}
    m = lgb.LGBMRegressor(**PRM).fit(tr[FEAT], tr.sub_temp); P["SELF"] = m.predict(te[FEAT])
    for nm, wv in (("POOL", pd.Series(1.0, index=dv.index)), ("SIM", ws)):
        parts, wts = [tr], [np.ones(len(tr))]
        tot = float(sum(wv[f] for f in donors))
        for f in donors:
            Rd = DON[f]; parts.append(Rd); wts.append(np.full(len(Rd), len(tr) * wv[f] / tot / len(Rd)))
        A = pd.concat(parts); Aw = np.concatenate(wts); feats = FEAT + ["fcat"]
        m = lgb.LGBMRegressor(**PRM).fit(A[feats], A.sub_temp, sample_weight=Aw); P[nm] = m.predict(te[feats])
    y = te.sub_temp.values
    for nm, p in P.items():
        out[f"{nm}_le8"] = rm((p - y)[m8]); out[f"{nm}_all"] = rm(p - y); out[f"{nm}_le6"] = rm((p - y)[m6]) if m6.sum() >= 10 else np.nan; out[f"{nm}_bias8"] = float((p - y)[m8].mean())
    print(T, "done", {k: round(v, 3) for k, v in out.items() if k.endswith("_le8")}, "| top3", out["top3"], flush=True); return out
with ThreadPoolExecutor(4) as ex: res = [r for r in ex.map(run_target, FARMS) if r is not None]
S = pd.DataFrame(res); pd.set_option("display.width", 250); print(S.round(3).to_string(index=False))
n = len(S); win = int((S.SIM_le8 < S.SELF_le8).sum()); lr_ = np.log(S.SIM_le8 / S.SELF_le8); p = float(wilcoxon(lr_, alternative="less").pvalue)
for a, b in (("SIM", "POOL"), ("POOL", "SELF")):
    l2 = np.log(S[f"{a}_le8"] / S[f"{b}_le8"]); print(f"  [desc] {a} vs {b}: better {int((l2 < 0).sum())}/{n}, median ratio {np.exp(l2.median()):.3f}, Wilcoxon p {wilcoxon(l2, alternative='less').pvalue:.4f}")
print(f"  [desc] SIM vs SELF all rows of cold days: better {int((S.SIM_all < S.SELF_all).sum())}/{n} | <=6 rows: better {int((S.SIM_le6 < S.SELF_le6).sum())}/{int(S.SELF_le6.notna().sum())}")
print(S[S.farm.isin(["F13", "F47"])].round(3).to_string(index=False))
print(f"LOGO1 SIM vs SELF (<=8 rows of held-out cold days): better {win}/{n} (need >= {int(np.ceil(2 * n / 3))}) | median ratio {np.exp(lr_.median()):.3f} | Wilcoxon p {p:.4f} -> {'PASS' if win >= 2 * n / 3 and p < .025 else 'FAIL'}")
S.to_csv(os.path.join(HERE, "logo1_v1_table.csv"), index=False)
