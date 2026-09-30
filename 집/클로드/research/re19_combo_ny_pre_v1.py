# -*- coding: utf-8 -*-
"""재분석 19 — 사전 고정 가설 H-C1 (2026-10-01, 집 클로드). ※ 사후 선택 조합: H-NY1(re09)·H-P3(re14) 결과를 본 뒤 묶음.
H-C1: G_C2에서 (1) MASK 안 nys → nys_dt (re08 OOF: 0.65 res + 0.25 ridge + 0.10 nys_dt)
                 (2) CODEX → CODEX+pre49 (re14 OOF) 를 동시에 적용. 가중·게이트 그대로, 재적합 없음.
시드 짝: (MASK 7, 사전학습 7), (MASK 101, 사전학습 101).
판정 (실행 전 고정): DIAG10·EXT10·EXT12 × 두 짝 여섯 칸 모두 개선 + DIAG10 P(worse) < 0.025/8 두 짝 모두.
주의: 두 구성요소 모두 같은 검증 자료에서 방향을 본 뒤 고른 것 — 통과해도 "작은 개선 후보"이며 과대평가 가능.
결과 local/re19_combo_ny_pre_v1.txt
"""
import env  # noqa
import numpy as np
from harness import load
from common import rmse
from screen_v6 import boot
P_MAX = 0.025 / 8
_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
m8 = np.load(env.LOCAL + "/re08_delta_target_v1_oof.npz", allow_pickle=True)
m14 = np.load(env.LOCAL + "/re14_pretrain_feature_v1_oof.npz", allow_pickle=True)
for q in (z, m8, m14):
    assert (q["row_id"] == lab.row_id.values).all()
y = lab.sub_temp.values; t = lab.in_temp.values
g = np.where(np.isnan(t), 1.0, np.clip((t - 8) / 2, 0, 1))
out = open(env.LOCAL + "/re19_combo_ny_pre_v1.txt", "w", encoding="utf-8")
def p(s): print(s); out.write(s + "\n")
passed = True
for s in ("DIAG10", "EXT10", "EXT12"):
    pfn = np.load(env.LOCAL + f"/web_tabpfn_v2_temp_{s}.npy").mean(0)
    blend = lambda b, c: (0.6 - 0.2 * (1 - g)) * b + (0.2 + 0.4 * (1 - g)) * c + 0.2 * g * pfn
    for sd in (7, 101):
        a = blend(m8[f"{s}__MASK__{sd}"], z[f"{s}__CODEX__726"])
        mask_ny = 0.65 * m8[f"{s}__res__{sd}"] + 0.25 * m8[f"{s}__ridge__{sd}"] + 0.10 * m8[f"{s}__nys_dt__{sd}"]
        b = blend(mask_ny, m14[f"{s}__CODEXPRE__{sd}"])
        ok = ~np.isnan(a) & ~np.isnan(b)
        ra, rb = rmse(a[ok], y[ok]), rmse(b[ok], y[ok])
        _, lo, hi, pw = boot(lab[ok].reset_index(drop=True), "sub_temp", a[ok], b[ok])
        win = rb < ra; passed &= win
        if s == "DIAG10": passed &= pw < P_MAX
        cold = t[ok] < 8
        p(f"{s:6s} 짝 {sd:3d}: G_C2 {ra:.5f} → H-C1 {rb:.5f} ({100*(rb/ra-1):+.2f}%) CI [{lo:+.4f},{hi:+.4f}] P(worse) {pw:.4f} "
          f"{'개선' if win else '악화'} | in<8 RMSE {rmse(a[ok][cold], y[ok][cold]):.3f} → {rmse(b[ok][cold], y[ok][cold]):.3f}")
p(f"\n판정 H-C1: {'후보(작은 개선)' if passed else '기각'}")
