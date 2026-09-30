# -*- coding: utf-8 -*-
"""재분석 14 — 사전 고정 가설 H-P3 (2026-10-01, 집 클로드). ※ re13 진단에서 도출.

근거 (re13 진단, 판정 아님): 49개 온실 실내3종 사전학습 모델은 공통 특징만으로는 약하지만(DIAG10 단독 0.94)
  추운 날 외삽을 크게 개선(EXT12 단독 M0 1.89~1.93 → H-P2 1.15~1.19).
가설 H-P3: 49개 온실로 사전학습한 모델(re13의 PARAMS·FEATS, 온실 범주 없음)의 예측 pre49를
  Codex 멤버(물리 Ridge + LightGBM 잔차)의 LightGBM 잔차 특징에 한 열로 추가 (그 외 전부 동일).
  G_C2에서 CODEX만 이것으로 교체. 사전학습은 다른 온실만 쓰므로 폴드와 무관, F13·F47 정답 미사용.
시드: 사전학습 시드 (7, 101) — Codex LightGBM은 무작위 요소 없음(re07 발견), 시드 변동은 사전학습에서만.
판정 (실행 전 고정):
  1. DIAG10, EXT10(전체 행), EXT12 × 사전학습 시드 (7, 101) 여섯 칸 모두 G_C2′ < G_C2
  2. DIAG10 P(worse) < 0.025/6 (캠페인 누적 가설 6개, 본페로니) — 두 시드 모두
  3. 둘 다 만족 → "후보"
진단: 멤버 단독 RMSE, in_temp<8 편향, G_C2 오차 상관.
결과: local/re14_pretrain_feature_v1.txt, local/re14_pretrain_feature_v1_oof.npz
"""
import env  # noqa: F401
import os
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
import harness
from common import split_mask, rmse, TARGET_FARMS
from screen_v6 import boot
from anal_q1_errors import diag_folds
import train_flags_v6 as TF
import temp_mask_v1 as TM
from resid_reset_features import build_features, FEATURE_COLUMNS, PHYSICS_COLUMNS
import re13_pooled51_v1 as R13

SEEDS = (7, 101)
P_MAX = 0.025 / 6
OUT = os.path.join(env.LOCAL, "re14_pretrain_feature_v1.txt")
_f = open(OUT, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _f.write(s + "\n")
    _f.flush()


def codex_fit_predict(tr, va, w, cols):
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    lin.fit(tr[PHYSICS_COLUMNS], tr.sub_temp.values, ridge__sample_weight=w)
    m = LGBMRegressor(n_estimators=220, learning_rate=0.035, num_leaves=12, max_depth=-1, min_child_samples=100,
                      reg_lambda=15, verbosity=-1, n_jobs=4, random_state=726)
    m.fit(tr[cols], tr.sub_temp.values - lin.predict(tr[PHYSICS_COLUMNS]), sample_weight=w)
    return lin.predict(va[PHYSICS_COLUMNS]) + m.predict(va[cols])


def main():
    t0 = time.time()
    labF, _, _ = TM.build_world()
    common.load_raw = TM.masked_loader
    try:
        labM, _, _ = TM.build_world()
    finally:
        common.load_raw = TM.ORIG
        harness._CACHE.clear()
    tX, ty, sX = TM.ORIG()
    CF = build_features(tX, sX).drop(columns=["farm", "day", "hour", "t"]).set_index("row_id")
    for c in FEATURE_COLUMNS:
        if c not in labM.columns:
            labM[c] = CF.loc[labM.row_id, c].values
    w = TF.row_weights(labM, 0.2, w_noisy=0.2)
    y = labM.sub_temp.values
    t = labM.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
    assert (z["row_id"] == labM.row_id.values).all()
    sets = [("DIAG10", diag_folds(labM))]
    for s in ("EXT10", "EXT12"):
        m = ~np.isnan(z[f"{s}__MASK__7"])
        dd = labM.loc[m, ["farm", "day"]].drop_duplicates()
        sets.append((s, [{f: set(dd.loc[dd.farm == f, "day"].astype(int)) for f in TARGET_FARMS}]))

    P = R13.build_panel()
    feats = [c for c in P.columns if c not in ("farm", "day", "hour", "row_id", "sub_temp", "farm_cat")]
    oth = P[~P.farm.isin(R13.TARGETS)]
    tgt = P[P.farm.isin(R13.TARGETS)].set_index("row_id").loc[labM.row_id]
    p(f"준비 {time.time()-t0:.0f}s, 다른 온실 {len(oth)}행")

    # 재현 검사: pre49 없이 원 Codex 설정이 저장 OOF와 같은가 (EXT10)
    fd = sets[1][1][0]
    trm, vam = split_mask(labM, fd)
    o = codex_fit_predict(labM[trm], labM[vam], w[trm], FEATURE_COLUMNS)
    dmax = np.max(np.abs(o - z["EXT10__CODEX__726"][vam]))
    p(f"재현 검사 EXT10 최대 절대차 {dmax:.2e}")
    if dmax > 1e-6:
        p("재현 실패 — 중단")
        return

    res = {}
    cols = FEATURE_COLUMNS + ["pre49"]
    for sd in SEEDS:
        pre = R13.fit(R13.PARAMS, oth[feats], oth.sub_temp.values, sd)
        labM["pre49"] = pre.predict(tgt[feats])
        for s, fds in sets:
            o = np.full(len(labM), np.nan)
            for fd in fds:
                trm, vam = split_mask(labM, fd)
                o[vam] = codex_fit_predict(labM[trm], labM[vam], w[trm], cols)
            res[(s, sd)] = o
        p(f"  시드 {sd} 완료 {time.time()-t0:.0f}s")
    np.savez(os.path.join(env.LOCAL, "re14_pretrain_feature_v1_oof.npz"), row_id=labM.row_id.values,
             **{f"{s}__CODEXPRE__{sd}": v for (s, sd), v in res.items()})

    passed = True
    p("\n== 판정 H-P3: G_C2 → G_C2′ (CODEX → CODEX+pre49) ==")
    for s, _ in sets:
        base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], axis=0)
        cx = z[f"{s}__CODEX__726"]
        pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{s}.npy")).mean(0)
        blend = lambda c: (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * c + 0.2 * g * pfn
        a = blend(cx)
        for sd in SEEDS:
            b = blend(res[(s, sd)])
            ok = ~np.isnan(a) & ~np.isnan(b)
            ra, rb = rmse(a[ok], y[ok]), rmse(b[ok], y[ok])
            _, lo, hi, pw = boot(labM[ok].reset_index(drop=True), "sub_temp", a[ok], b[ok])
            win = rb < ra
            passed &= win
            if s == "DIAG10":
                passed &= pw < P_MAX
            cold = t[ok] < 8
            p(f"  {s:6s} 시드 {sd:3d}: G_C2 {ra:.5f} → {rb:.5f} ({100*(rb/ra-1):+.2f}%) CI [{lo:+.4f},{hi:+.4f}] "
              f"P(worse) {pw:.4f} {'개선' if win else '악화'}")
            p(f"         멤버 단독 {rmse(cx[ok], y[ok]):.4f} → {rmse(res[(s, sd)][ok], y[ok]):.4f} | in<8 편향 "
              f"{(cx[ok]-y[ok])[cold].mean():+.3f} → {(res[(s, sd)][ok]-y[ok])[cold].mean():+.3f} (n {cold.sum()})")
    p(f"\n판정: {'후보' if passed else '기각'}  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
    _f.close()
