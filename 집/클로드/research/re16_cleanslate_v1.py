# -*- coding: utf-8 -*-
"""재분석 16 — 백지 상태 단순 모델 점검 (2026-10-01, 집 클로드). 진단 전용, 채택 후보 아님.

질문: 기존 특징·가중·정제·혼합을 전부 버리고 원자료 입력 14개의 인과 이력만으로 만든 LightGBM 하나가
  G_C2(여러 멤버·게이트 혼합)와 비교해 어느 정도인가. 우리 파이프라인의 복잡함이 이득을 내고 있는가.
설계 (실행 전 고정, 튜닝 없음):
  자료: F13·F47 학습 9,600행만. 가중치·오염 플래그 없음.
  특징: 사용 가능 입력 14개(out 4, in 3, act 7) 각각 현재·lag1·lag3·ewm3·ewm12·ewm24,
        in_temp는 추가로 lag2·6·12·24, d1·d3, ewm2·6·48, 24시간 min·max·mean, 자정 이후 누적평균.
        시각 sin·cos, 온실(F47=1). 연속 시간축에서 과거·현재만, test 행 제외(MASK와 동일).
  모델: LGBM(l2, 1000트리, lr 0.03, 잎 31, min_child 30, subsample 0.8, colsample 0.8), 시드 7·101.
해석 규칙 (실행 전 고정):
  S 단독 DIAG10 RMSE ≤ G_C2 × 1.02 이면 "복잡함이 이득을 거의 못 냄", ≥ G_C2 × 1.10 이면 "기존 특징이 제 역할".
  참고: 0.8·G_C2 + 0.2·S 변화(판정 아님).
결과: local/re16_cleanslate_v1.txt, local/re16_cleanslate_v1_oof.npz
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
TARGETS = ("F13", "F47")
VARS = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2",
        "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
PARAMS = dict(objective="l2", n_estimators=1000, learning_rate=0.03, num_leaves=31, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
OUT = os.path.join(env.LOCAL, "re16_cleanslate_v1.txt")
_f = open(OUT, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _f.write(s + "\n")
    _f.flush()


def build():
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
    Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
    X = X[X.row_id.str[:3].isin(TARGETS)].copy()
    k = X.row_id.str.extract(r"^(F\d+)_(\d+)_(\d+)$")
    X["farm"], X["day"], X["hour"] = k[0], k[1].astype(int), k[2].astype(int)
    X["t"] = X.day * 24 + X.hour
    parts = []
    for fm, g in X.groupby("farm"):
        g = g.set_index("t").sort_index().assign(present=1.0)
        full = g.reindex(np.arange(g.index.min(), g.index.max() + 1))
        day, hour = full.index // 24, full.index % 24
        F = pd.DataFrame(index=full.index)
        F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        F["f47"] = float(fm == "F47")
        for v in VARS:
            s = full[v]
            F[v] = s
            F[f"{v}_l1"], F[f"{v}_l3"] = s.shift(1), s.shift(3)
            for sp in (3, 12, 24):
                F[f"{v}_e{sp}"] = s.ewm(span=sp, min_periods=1).mean()
        s = full.in_temp
        for L in (2, 6, 12, 24):
            F[f"in_temp_l{L}"] = s.shift(L)
        F["in_temp_d1"], F["in_temp_d3"] = s - s.shift(1), s - s.shift(3)
        for sp in (2, 6, 48):
            F[f"in_temp_e{sp}"] = s.ewm(span=sp, min_periods=1).mean()
        F["in_temp_min24"] = s.rolling(24, min_periods=6).min()
        F["in_temp_max24"] = s.rolling(24, min_periods=6).max()
        F["in_temp_mean24"] = s.rolling(24, min_periods=6).mean()
        F["in_temp_daycum"] = s.groupby(day).transform(lambda q: q.expanding().mean())
        F["farm"], F["day"] = fm, day
        F["row_id"] = [f"{fm}_{d:03d}_{h:02d}" for d, h in zip(day, hour)]
        parts.append(F[full.present.notna().values])
    P = pd.concat(parts, ignore_index=True).merge(Y, on="row_id")
    return P


def main():
    t0 = time.time()
    P = build()
    feats = [c for c in P.columns if c not in ("farm", "day", "row_id", "sub_temp")]
    _, labH, _ = load()
    z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
    assert (labH.row_id.values == z["row_id"]).all()
    P = P.set_index("row_id").loc[labH.row_id].reset_index()
    assert len(P) == 9600
    p(f"특징 {len(feats)}개, 구성 {time.time()-t0:.0f}s")
    sets = {"DIAG10": diag_folds(labH)}
    for s in ("EXT10", "EXT12"):
        m = ~np.isnan(z[f"{s}__MASK__7"])
        dd = labH.loc[m, ["farm", "day"]].drop_duplicates()
        sets[s] = [{f: set(dd.loc[dd.farm == f, "day"].astype(int)) for f in TARGETS}]
    y = labH.sub_temp.values
    t = labH.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    res = {}
    for sd in SEEDS:
        for s, fds in sets.items():
            o = np.full(len(P), np.nan)
            for fd in fds:
                trm, vam = split_mask(P, fd)
                m = lgb.LGBMRegressor(random_state=sd, **PARAMS).fit(P.loc[trm, feats], P.loc[trm, "sub_temp"])
                o[vam] = m.predict(P.loc[vam, feats])
            res[(s, sd)] = o
        p(f"  시드 {sd} 완료 {time.time()-t0:.0f}s")
    np.savez(os.path.join(env.LOCAL, "re16_cleanslate_v1_oof.npz"), row_id=labH.row_id.values,
             **{f"{s}__S__{sd}": v for (s, sd), v in res.items()})
    for s in sets:
        base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], axis=0)
        cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], axis=0)
        pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{s}.npy")).mean(0)
        gc2 = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
        for sd in SEEDS:
            S = res[(s, sd)]
            ok = ~np.isnan(gc2) & ~np.isnan(S)
            rg, rs = rmse(gc2[ok], y[ok]), rmse(S[ok], y[ok])
            b = 0.8 * gc2 + 0.2 * S
            rb = rmse(b[ok], y[ok])
            _, lo, hi, pw = boot(labH[ok].reset_index(drop=True), "sub_temp", gc2[ok], b[ok])
            e = S[ok] - y[ok]
            dsh = (pd.Series(e).groupby(labH.loc[ok, "farm"].values + labH.loc[ok, "day"].astype(str).values).transform("mean") ** 2).sum() / (e ** 2).sum()
            p(f"{s:6s} 시드 {sd:3d}: G_C2 {rg:.4f} | 단순 S {rs:.4f} (비 {rs/rg:.3f}) | 멤버 MASK {rmse(base[ok], y[ok]):.4f} "
              f"CODEX {rmse(cx[ok], y[ok]):.4f} TabPFN {rmse(pfn[ok], y[ok]):.4f} | 오차상관 {np.corrcoef(e, gc2[ok]-y[ok])[0,1]:.3f} "
              f"| S 하루오프셋 비중 {100*dsh:.1f}% | 0.8G+0.2S {100*(rb/rg-1):+.2f}% P(worse) {pw:.3f}")
    p(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
    _f.close()
