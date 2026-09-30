# -*- coding: utf-8 -*-
"""재분석 8 — 사전 고정 가설 H-DT1 (2026-10-01, 집 클로드).

근거: re06 — MASK 멤버(0.65 res + 0.25 ridge + 0.10 nys)가 추운 행을 과대예측(EXT10 +0.56, EXT12 +0.94).
  re05 C6 — 추운 구간 배지−공기 차는 F13 1.00·F47 0.72로 작고 안정적.
  H-LT1(선형 잎) 기각으로 "기울기 연장"은 해법이 아님이 확인됨.
가설 H-DT1: 평평하게 외삽하는 두 멤버(트리 res, 커널 nys)가 배우는 대상을
  sub_temp(또는 sub_temp − 물리선형기준) → sub_temp − 앵커, 앵커 = ph_in_temp_3(3시간 평활 실내온도)
  로 바꾸고 예측 때 앵커를 다시 더한다. 공기온도를 따라 움직이는 부분을 트리 밖으로 빼서
  범위 밖에서도 "공기 + 작은 차"로 외삽하게 한다.
  앵커 결측 행은 물리 선형기준 예측으로 채움. ridge는 그대로. 가중치·특징·하이퍼파라미터 그대로.

판정 규칙 (실행 전 고정):
  1. DIAG10, EXT10(전체 행), EXT12 × MASK 시드 (7, 101) 여섯 칸 모두
     G_C2′(MASK→MASK_DT, 같은 시드) RMSE < G_C2(같은 시드 MASK) RMSE. CODEX·TabPFN은 저장본 그대로.
  2. DIAG10 P(worse) < 0.0125 (이번 캠페인 두 번째 가설, 본페로니 0.025/2) — 두 시드 모두.
  3. 둘 다 만족하면 "후보", 아니면 기각. 크기 무관.
진단(판정 아님): 구성 멤버별 RMSE, in_temp<8 편향, G_C2와 오차 상관.
재현 검사: 원 설정 MASK(시드 7)를 DIAG10 첫 폴드·EXT10에서 다시 만들어 저장 OOF와 비교 (최대 절대차 < 1e-6).

결과: local/re08_delta_target_v1.txt, local/re08_delta_target_v1_oof.npz
실행: cd 집/클로드/research && PYTHONPATH="" python -u re08_delta_target_v1.py
"""
import env  # noqa: F401
import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

import common
import harness
import cold_v5
from cold_v5 import lgbh, ridge, nys
from common import split_mask, rmse, TARGET_FARMS
from screen_v6 import boot, fit_w, collect
from anal_q1_errors import diag_folds
import train_flags_v6 as TF
import temp_mask_v1 as TM

SEEDS_T = (7, 101)
P_MAX = 0.0125
ANCHOR = "ph_in_temp_3"
OUT = os.path.join(env.LOCAL, "re08_delta_target_v1.txt")
_f = open(OUT, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _f.write(s + "\n")
    _f.flush()


def members(tr, va, ct, phc, w):
    """원 멤버(res, ridge, nys)와 H-DT1 멤버(res_dt, nys_dt)를 같은 폴드에서 함께 만든다."""
    imp = SimpleImputer(strategy="median").fit(tr[phc])
    b = fit_w(LinearRegression(), imp.transform(tr[phc]), tr.sub_temp.values, w, None)
    btr, bva = b.predict(imp.transform(tr[phc])), b.predict(imp.transform(va[phc]))
    y = tr.sub_temp.values
    out = {
        "res": bva + fit_w(lgbh(), tr[ct], y - btr, w, None).predict(va[ct]),
        "ridge": fit_w(ridge(), tr[ct], y, w, "ridge").predict(va[ct]),
        "nys": fit_w(nys(), tr[ct], y, w, "ridge").predict(va[ct]),
    }
    atr = np.where(np.isnan(tr[ANCHOR].values), btr, tr[ANCHOR].values)
    ava = np.where(np.isnan(va[ANCHOR].values), bva, va[ANCHOR].values)
    out["res_dt"] = ava + fit_w(lgbh(), tr[ct], y - atr, w, None).predict(va[ct])
    out["nys_dt"] = ava + fit_w(nys(), tr[ct], y - atr, w, "ridge").predict(va[ct])
    return out


def main():
    labF, ct, phc = TM.build_world()
    common.load_raw = TM.masked_loader
    try:
        labM, _, _ = TM.build_world()
    finally:
        common.load_raw = TM.ORIG
        harness._CACHE.clear()
    w = TF.row_weights(labM, 0.2, w_noisy=0.2)
    y = labM.sub_temp.values
    t = labM.in_temp.values
    g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
    dmin = labF.groupby(["farm", "day"]).ph_in_temp_3.min()
    sets = [("DIAG10", diag_folds(labM))]
    for th in (10.0, 12.0):
        cd = dmin[dmin < th]
        sets.append(("EXT%d" % th, [{f: set(int(d) for (ff, d) in cd.index if ff == f) for f in TARGET_FARMS}]))
    z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
    assert (z["row_id"] == labM.row_id.values).all()
    p(f"앵커 {ANCHOR} 결측 행: {int(np.isnan(labM[ANCHOR].values).sum())}")

    res = {}
    for s, fds in sets:
        for sd in SEEDS_T:
            cold_v5.SEED = sd
            M = collect(labM, fds, lambda tr, va, m: members(tr, va, ct, phc, w[m]))
            res[(s, sd, "MASK")] = 0.65 * M["res"] + 0.25 * M["ridge"] + 0.10 * M["nys"]
            res[(s, sd, "MASK_DT")] = 0.65 * M["res_dt"] + 0.25 * M["ridge"] + 0.10 * M["nys_dt"]
            for k in ("res", "ridge", "nys", "res_dt", "nys_dt"):
                res[(s, sd, k)] = M[k]
            if sd == 7 and s in ("DIAG10", "EXT10"):
                ok = ~np.isnan(res[(s, sd, "MASK")])
                dmax = np.nanmax(np.abs(res[(s, sd, "MASK")][ok] - z[f"{s}__MASK__7"][ok]))
                p(f"재현 검사 {s} 시드 7: 행 {ok.sum()} 최대 절대차 {dmax:.2e}")
                if dmax > 1e-6:
                    p("재현 실패 — 중단")
                    return
        cold_v5.SEED = 7
        p(f"  {s} 적합 완료")
    np.savez(os.path.join(env.LOCAL, "re08_delta_target_v1_oof.npz"), row_id=labM.row_id.values,
             **{f"{s}__{k}__{sd}": v for (s, sd, k), v in res.items()})

    p("\n== 판정: G_C2 → G_C2′ (MASK → MASK_DT, 같은 시드) ==")
    passed = True
    for s, _ in sets:
        cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], axis=0)
        pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{s}.npy")).mean(0)
        for sd in SEEDS_T:
            def blend(base):
                return (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
            a, b = blend(res[(s, sd, "MASK")]), blend(res[(s, sd, "MASK_DT")])
            ok = ~np.isnan(a) & ~np.isnan(b)
            ra, rb = rmse(a[ok], y[ok]), rmse(b[ok], y[ok])
            pr, lo, hi, pw = boot(labM[ok].reset_index(drop=True), "sub_temp", a[ok], b[ok])
            win = rb < ra
            passed &= win
            if s == "DIAG10":
                passed &= pw < P_MAX
            p(f"  {s:6s} 시드 {sd:3d}: G_C2 {ra:.5f} → G_C2′ {rb:.5f} ({100*(rb/ra-1):+.2f}%) "
              f"CI [{lo:+.4f},{hi:+.4f}] P(worse) {pw:.4f} {'개선' if win else '악화'}")
            cold = t[ok] < 8
            for k in ("MASK", "MASK_DT", "res", "res_dt", "nys", "nys_dt"):
                v = res[(s, sd, k)][ok]
                p(f"         {k:8s} RMSE {rmse(v, y[ok]):.4f}  in<8 편향 {(v - y[ok])[cold].mean():+.3f} (n {cold.sum()})")
            p(f"         G_C2 오차 상관 MASK {np.corrcoef(res[(s, sd, 'MASK')][ok]-y[ok], a[ok]-y[ok])[0,1]:.3f} "
              f"MASK_DT {np.corrcoef(res[(s, sd, 'MASK_DT')][ok]-y[ok], a[ok]-y[ok])[0,1]:.3f}")
    p(f"\n판정: {'후보' if passed else '기각'}")


if __name__ == "__main__":
    main()
    _f.close()
