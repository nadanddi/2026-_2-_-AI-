# -*- coding: utf-8 -*-
"""재분석 7 — 사전 고정 가설 H-LT1 (2026-10-01, 집 클로드).

근거 (re06_member_bias): EXT10·EXT12에서 트리 비중이 큰 멤버일수록 추운 행(in_temp<8)을 과대예측
  (MASK +0.56/+0.94, TabPFN +1.02/+1.90, CODEX +0.07/−0.23). 트리는 학습 범위 밖에서 평평하게 외삽한다.
가설 H-LT1: Codex 멤버(물리 Ridge + LightGBM 잔차)의 잔차 LightGBM을 linear_tree=True로 바꾸면
  (그 외 전부 동일) 외삽 비용이 줄고, G_C2에서 CODEX를 이것으로 바꾼 G_C2′가 개선된다.

한 가지만 바꿈: linear_tree=True, linear_lambda=10.0 (실행 전 고정, 튜닝 안 함).
G_C2 식과 가중치는 그대로, CODEX 멤버만 교체.

판정 규칙 (실행 전 고정; 사용자 채택 기준):
  1. 검증기 DIAG10, EXT10(전체 행), EXT12 × Codex 시드 (726, 727) 여섯 칸 모두 G_C2′ RMSE < G_C2 RMSE
     (G_C2의 MASK·TabPFN 멤버는 저장본 그대로, CODEX만 같은 시드끼리 교체)
  2. DIAG10 날 블록 부트스트랩 P(worse) < 0.025 (가설 1개, 본페로니 k=1) — 두 시드 모두
  3. 둘 다 만족하면 "후보", 아니면 기각. 크기 무관.
진단(판정 아님): 멤버 단독 RMSE, in_temp<8 편향, EXT10 오프셋~heat 스피어만.
재현 검사: 원래 Codex 설정으로 다시 적합해 저장 OOF와 같은지 먼저 확인 (최대 절대차 < 1e-6 아니면 중단).

결과: local/re07_linear_tree_v1.txt, local/re07_linear_tree_v1_oof.npz
실행: cd 집/클로드/research && PYTHONPATH="" python -u re07_linear_tree_v1.py
"""
import env  # noqa: F401
import os
import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
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

SEEDS_C = (726, 727)
LINEAR_LAMBDA = 10.0
OUT = os.path.join(env.LOCAL, "re07_linear_tree_v1.txt")
_f = open(OUT, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _f.write(s + "\n")
    _f.flush()


def codex_fit_predict(tr, va, w, seed, linear_tree):
    lin = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=100.0))
    lin.fit(tr[PHYSICS_COLUMNS], tr.sub_temp.values, ridge__sample_weight=w)
    kw = dict(linear_tree=True, linear_lambda=LINEAR_LAMBDA) if linear_tree else {}
    m = LGBMRegressor(n_estimators=220, learning_rate=0.035, num_leaves=12, max_depth=-1, min_child_samples=100,
                      reg_lambda=15, verbosity=-1, n_jobs=4, random_state=seed, **kw)
    m.fit(tr[FEATURE_COLUMNS], tr.sub_temp.values - lin.predict(tr[PHYSICS_COLUMNS]), sample_weight=w)
    return lin.predict(va[PHYSICS_COLUMNS]) + m.predict(va[FEATURE_COLUMNS])


def main():
    # temp_mask_v1과 같은 세계 구성
    labF, ct, phc = TM.build_world()
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
    dmin = labF.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(labM))]
    for th in (10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))

    z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
    assert (z["row_id"] == labM.row_id.values).all()
    t = labM.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))

    # 재현 검사 (원래 설정, 시드 726, DIAG10 첫 폴드 + EXT10)
    for s, fds in sets[:2]:
        o = np.full(len(labM), np.nan)
        for fd in (fds[:1] if s == "DIAG10" else fds):
            trm, vam = split_mask(labM, fd)
            o[vam] = codex_fit_predict(labM[trm], labM[vam], w[trm], 726, False)
        ok = ~np.isnan(o)
        dmax = np.nanmax(np.abs(o[ok] - z[f"{s}__CODEX__726"][ok]))
        p(f"재현 검사 {s}: 행 {ok.sum()} 최대 절대차 {dmax:.2e}")
        if dmax > 1e-6:
            p("재현 실패 — 중단")
            return

    res = {}
    for s, fds in sets:
        for sc in SEEDS_C:
            o = np.full(len(labM), np.nan)
            for fd in fds:
                trm, vam = split_mask(labM, fd)
                o[vam] = codex_fit_predict(labM[trm], labM[vam], w[trm], sc, True)
            res[(s, sc)] = o
        p(f"  {s} 적합 완료")
    np.savez(os.path.join(env.LOCAL, "re07_linear_tree_v1_oof.npz"), row_id=labM.row_id.values,
             **{f"{s}__CODEXLT__{sc}": v for (s, sc), v in res.items()})

    p("\n== 판정: G_C2 → G_C2′ (CODEX만 linear_tree 판으로, 같은 시드) ==")
    passed = True
    for s, _ in sets:
        base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], axis=0)
        pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{s}.npy")).mean(0)
        for sc in SEEDS_C:
            cx_old, cx_new = z[f"{s}__CODEX__{sc}"], res[(s, sc)]
            a = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx_old + 0.2 * g * pfn
            b = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx_new + 0.2 * g * pfn
            ok = ~np.isnan(a) & ~np.isnan(b)
            ra, rb = rmse(a[ok], y[ok]), rmse(b[ok], y[ok])
            pr, lo, hi, pw = boot(labM[ok].reset_index(drop=True), "sub_temp", a[ok], b[ok])
            win = rb < ra
            passed &= win
            if s == "DIAG10":
                passed &= pw < 0.025
            p(f"  {s:6s} 시드 {sc}: G_C2 {ra:.5f} → G_C2′ {rb:.5f} ({100*(rb/ra-1):+.2f}%) "
              f"CI [{lo:+.4f},{hi:+.4f}] P(worse) {pw:.4f} {'개선' if win else '악화'}")
            # 진단: 멤버 단독
            e_old, e_new = cx_old[ok] - y[ok], cx_new[ok] - y[ok]
            cold = (t[ok] < 8)
            p(f"         멤버 단독 RMSE {rmse(cx_old[ok], y[ok]):.4f} → {rmse(cx_new[ok], y[ok]):.4f}, "
              f"in<8 편향 {e_old[cold].mean():+.3f} → {e_new[cold].mean():+.3f} (n {cold.sum()})")
    p(f"\n판정: {'후보 (모든 칸 개선 + DIAG10 P(worse)<0.025)' if passed else '기각'}")


if __name__ == "__main__":
    main()
    _f.close()
