# -*- coding: utf-8 -*-
"""재분석 13 — 사전 고정 가설 H-P1, H-P2 (2026-10-01, 집 클로드). 자료 선택: 51개 온실 온도 정답 활용.

배경: 리더보드 1위 온도 0.3885(제출 4회)로 역탐색·복사본·이웃 라벨 가설 기각(re10~re12).
  설명회 11·41쪽이 "어떤 온실 자료를 얼마나 쓸지"를 미션으로 명시. 다른 49개 온실 sub_temp 295,365행 미사용.
  E6: 온실마다 배지−공기 차가 달라 그대로 섞으면 안 됨 → 온실별 수준은 따로, 반응 동역학만 공유하는 설계.
공통 입력: 모든 온실에 있는 실내 3종(in_temp, in_hum, in_co2)과 그 인과 이력, 시각. (외기·구동기는 49곳 대부분 결측)
특징은 온실별 연속 시간축(일차*24+시)에서 과거·현재만 사용. F13·F47 학습행은 MASK(test_X 입력 NaN과 동일: test 행 미포함).

멤버 (모두 LightGBM, 특징·하이퍼파라미터 실행 전 고정, 튜닝 없음):
  M0 (비교용, 판정 대상 아님): F13·F47 학습 폴드만, 공통 특징 + 온실 범주.
  H-P1 공동 학습: 51개 온실 전부(F13·F47은 학습 폴드만), 공통 특징 + 온실 범주(51).
  H-P2 사전학습+미세조정: 49개 온실로 사전학습(온실 범주 없음) → F13·F47 학습 폴드에서 init_score로 이어 학습.
대상은 sub_temp.
판정 (실행 전 고정): 각 가설 멤버 M에 대해 G_C2′ = 0.8·G_C2 + 0.2·M (가중 0.2 고정)
  1. DIAG10, EXT10(전체 행), EXT12 × 시드 (7, 101) 여섯 칸 모두 G_C2′ < G_C2
  2. DIAG10 P(worse) < 0.025/5 = 0.005 (캠페인 누적 가설 5개, 본페로니) — 두 시드 모두
  3. 둘 다 만족 → "후보". 둘 다 후보면 DIAG10 평균 개선이 큰 쪽.
진단(판정 아님): 멤버 단독 RMSE, G_C2와 오차 상관, in_temp<8 편향, M0 대비.
정제 (Cleaning Log #4 + 실행 전 추가 고정): 다른 온실 in_temp<0 또는 >45 → NaN, sub_temp≥40 행 제외,
  in_hum<=0 또는 in_co2<=0 → NaN (물리 불가능값).
결과: local/re13_pooled51_v1.txt, local/re13_pooled51_v1_oof.npz
실행: cd 집/클로드/research && PYTHONPATH="" python -u re13_pooled51_v1.py
"""
import env  # noqa: F401
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from harness import load
from common import split_mask, rmse
from screen_v6 import boot
from anal_q1_errors import diag_folds

SEEDS = (7, 101)
P_MAX = 0.025 / 5
W_NEW = 0.2
TARGETS = ("F13", "F47")
OUT = os.path.join(env.LOCAL, "re13_pooled51_v1.txt")
_f = open(OUT, "w", encoding="utf-8")

PARAMS = dict(objective="l2", n_estimators=800, learning_rate=0.05, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
FT_PARAMS = dict(objective="l2", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                 deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _f.write(s + "\n")
    _f.flush()


def build_panel():
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp", "in_hum", "in_co2"])
    Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
    k = X.row_id.str.extract(r"^(F\d+)_(\d+)_(\d+)$")
    X["farm"], X["day"], X["hour"] = k[0], k[1].astype(int), k[2].astype(int)
    oth = ~X.farm.isin(TARGETS)
    X.loc[oth & ((X.in_temp < 0) | (X.in_temp > 45)), "in_temp"] = np.nan
    X.loc[oth & (X.in_hum <= 0), "in_hum"] = np.nan
    X.loc[oth & (X.in_co2 <= 0), "in_co2"] = np.nan
    X["t"] = X.day * 24 + X.hour
    parts = []
    for fm, g in X.groupby("farm"):
        g = g.set_index("t").sort_index().assign(present=1.0)
        full = g.reindex(np.arange(g.index.min(), g.index.max() + 1))   # 빠진 시각·test 시각은 NaN (MASK)
        full["farm"] = fm
        full["day"] = full.index // 24
        full["hour"] = full.index % 24
        s = full.in_temp
        F = pd.DataFrame(index=full.index)
        F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * full.hour / 24), np.cos(2 * np.pi * full.hour / 24)
        for c in ("in_temp", "in_hum", "in_co2"):
            F[c] = full[c]
        for L in (1, 2, 3, 6, 12, 24):
            F[f"t_lag{L}"] = s.shift(L)
        F["t_d1"], F["t_d3"] = s - s.shift(1), s - s.shift(3)
        for sp in (2, 3, 6, 12, 24, 48):
            F[f"t_ewm{sp}"] = s.ewm(span=sp, min_periods=1).mean()
        F["t_min24"], F["t_max24"] = s.rolling(24, min_periods=6).min(), s.rolling(24, min_periods=6).max()
        F["t_mean24"] = s.rolling(24, min_periods=6).mean()
        for c in ("in_hum", "in_co2"):
            F[f"{c}_ewm6"] = full[c].ewm(span=6, min_periods=1).mean()
            F[f"{c}_ewm24"] = full[c].ewm(span=24, min_periods=1).mean()
        es = 0.6108 * np.exp(17.27 * s / (s + 237.3))
        F["vpd"] = es * (1 - full.in_hum / 100)
        F["dew_gap"] = s - (237.3 * np.log(np.maximum(full.in_hum, 1) / 100 * es / 0.6108)
                            / (17.27 - np.log(np.maximum(full.in_hum, 1) / 100 * es / 0.6108)))
        F["farm"], F["day"], F["hour"] = fm, full.day.values, full.hour.values
        F["row_id"] = [f"{fm}_{d:03d}_{h:02d}" for d, h in zip(full.day, full.hour)]
        parts.append(F[full.present.notna().values])
    P = pd.concat(parts, ignore_index=True)
    P = P.merge(Y, on="row_id", how="inner")
    P = P[~((~P.farm.isin(TARGETS)) & (P.sub_temp >= 40))].reset_index(drop=True)
    P["farm_cat"] = P.farm.astype("category")
    return P


FEATS = None


def fit(params, X, y, seed, init=None, cat=None):
    m = lgb.LGBMRegressor(random_state=seed, **params)
    m.fit(X, y, init_score=init, categorical_feature=cat if cat else "auto")
    return m


def main():
    global FEATS
    t0 = time.time()
    P = build_panel()
    FEATS = [c for c in P.columns if c not in ("farm", "day", "hour", "row_id", "sub_temp", "farm_cat")]
    p(f"패널 {len(P)}행, 온실 {P.farm.nunique()}, 특징 {len(FEATS)}개, 구성 {time.time()-t0:.0f}s")

    _, labH, _ = load()
    z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
    assert (labH.row_id.values == z["row_id"]).all()
    pos = pd.Series(np.arange(len(labH)), index=labH.row_id)
    tgt = P[P.farm.isin(TARGETS)].copy()
    assert len(tgt) == 9600 and set(tgt.row_id) == set(labH.row_id)
    oth = P[~P.farm.isin(TARGETS)]
    p(f"다른 온실 학습행 {len(oth)}")
    sets = {"DIAG10": diag_folds(labH)}
    for s in ("EXT10", "EXT12"):
        m = ~np.isnan(z[f"{s}__MASK__7"])
        dd = labH.loc[m, ["farm", "day"]].drop_duplicates()
        sets[s] = [{f: set(dd.loc[dd.farm == f, "day"].astype(int)) for f in TARGETS}]

    Xcat = FEATS + ["farm_cat"]
    res = {}
    for sd in SEEDS:
        pre = fit(PARAMS, oth[FEATS], oth.sub_temp.values, sd)          # H-P2 사전학습 (폴드 무관)
        p(f"  시드 {sd} 사전학습 완료 {time.time()-t0:.0f}s")
        for s, fds in sets.items():
            o = {k: np.full(len(labH), np.nan) for k in ("M0", "HP1", "HP2")}
            for fd in fds:
                trm, vam = split_mask(tgt, fd)
                tr, va = tgt[trm], tgt[vam]
                ip = pos[va.row_id].values
                # M0
                m0 = fit(PARAMS, tr[Xcat], tr.sub_temp.values, sd, cat=["farm_cat"])
                o["M0"][ip] = m0.predict(va[Xcat])
                # H-P1
                pool = pd.concat([oth, tr])
                m1 = fit(PARAMS, pool[Xcat], pool.sub_temp.values, sd, cat=["farm_cat"])
                o["HP1"][ip] = m1.predict(va[Xcat])
                # H-P2
                itr, iva = pre.predict(tr[FEATS]), pre.predict(va[FEATS])
                m2 = fit(FT_PARAMS, tr[Xcat], tr.sub_temp.values, sd, init=itr, cat=["farm_cat"])
                o["HP2"][ip] = iva + m2.predict(va[Xcat])
            for k, v in o.items():
                res[(s, sd, k)] = v
            p(f"  시드 {sd} {s} 완료 {time.time()-t0:.0f}s")
    np.savez(os.path.join(env.LOCAL, "re13_pooled51_v1_oof.npz"), row_id=labH.row_id.values,
             **{f"{s}__{k}__{sd}": v for (s, sd, k), v in res.items()})

    y = labH.sub_temp.values
    t = labH.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    verdict = {}
    for k in ("HP1", "HP2"):
        passed, gains = True, []
        p(f"\n== 판정 {k}: G_C2 → 0.8·G_C2 + 0.2·{k} ==")
        for s in sets:
            base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], axis=0)
            cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], axis=0)
            pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{s}.npy")).mean(0)
            gc2 = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
            for sd in SEEDS:
                mem = res[(s, sd, k)]
                b = (1 - W_NEW) * gc2 + W_NEW * mem
                ok = ~np.isnan(gc2) & ~np.isnan(b)
                ra, rb = rmse(gc2[ok], y[ok]), rmse(b[ok], y[ok])
                _, lo, hi, pw = boot(labH[ok].reset_index(drop=True), "sub_temp", gc2[ok], b[ok])
                win = rb < ra
                passed &= win
                if s == "DIAG10":
                    passed &= pw < P_MAX
                    gains.append(rb / ra - 1)
                m0 = res[(s, sd, "M0")]
                cold = t[ok] < 8
                p(f"  {s:6s} 시드 {sd:3d}: G_C2 {ra:.5f} → {rb:.5f} ({100*(rb/ra-1):+.2f}%) CI [{lo:+.4f},{hi:+.4f}] "
                  f"P(worse) {pw:.4f} {'개선' if win else '악화'}")
                p(f"         단독 {k} {rmse(mem[ok], y[ok]):.4f} (M0 {rmse(m0[ok], y[ok]):.4f}) | G_C2 오차상관 "
                  f"{np.corrcoef(mem[ok]-y[ok], gc2[ok]-y[ok])[0,1]:.3f} | in<8 편향 {(mem[ok]-y[ok])[cold].mean():+.3f} "
                  f"(G_C2 {(gc2[ok]-y[ok])[cold].mean():+.3f}, n {cold.sum()})")
        verdict[k] = (passed, np.mean(gains))
        p(f"  {k}: {'후보' if passed else '기각'}")
    p(f"\n최종: {verdict}  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
    _f.close()
