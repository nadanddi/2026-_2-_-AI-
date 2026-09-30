# -*- coding: utf-8 -*-
"""재분석 15 — 사전 고정 가설 H-P4 (2026-10-01, 집 클로드). 사용자 제안: 추운 정도에 따라 49개 온실 사전학습 모델 가중.
H-P4: G_C2′ = (1−w)·G_C2 + w·HP2,  w = 0.2·(1−g),  g = clip((in_temp−8)/2, 0, 1)  (G_C2와 같은 게이트, 튜닝 없음)
  HP2 = re13의 49개 온실 사전학습 → F13·F47 미세조정 OOF (local/re13_pooled51_v1_oof.npz, 재적합 없음).
판정 (실행 전 고정): DIAG10·EXT10·EXT12 × 시드(7,101) 여섯 칸 모두 개선 + DIAG10 P(worse) < 0.025/7 두 시드.
진단(판정 아님): 가중 상한 0.1~1.0에 대한 곡선 — 서술용, 선택에 쓰지 않음.
결과 local/re15_cold_gate_pre49_v1.txt
"""
import env  # noqa
import os, numpy as np
from harness import load
from common import rmse
from screen_v6 import boot
P_MAX = 0.025 / 7
_, lab, _ = load()
z = np.load(env.LOCAL + "/temp_mask_v1_oof.npz", allow_pickle=True)
r = np.load(env.LOCAL + "/re13_pooled51_v1_oof.npz", allow_pickle=True)
assert (z["row_id"] == lab.row_id.values).all() and (r["row_id"] == lab.row_id.values).all()
y = lab.sub_temp.values; t = lab.in_temp.values
g = np.where(np.isnan(t), 1.0, np.clip((t - 8) / 2, 0, 1))
out = open(env.LOCAL + "/re15_cold_gate_pre49_v1.txt", "w", encoding="utf-8")
def p(s): print(s); out.write(s + "\n")
passed = True
curves = {}
for s in ("DIAG10", "EXT10", "EXT12"):
    base = np.nanmean([z[f"{s}__MASK__7"], z[f"{s}__MASK__101"]], 0)
    cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], 0)
    pfn = np.load(env.LOCAL + f"/web_tabpfn_v2_temp_{s}.npy").mean(0)
    gc2 = (0.6 - 0.2 * (1 - g)) * base + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
    for sd in (7, 101):
        hp2 = r[f"{s}__HP2__{sd}"]
        w = 0.2 * (1 - g)
        b = (1 - w) * gc2 + w * hp2
        ok = ~np.isnan(gc2) & ~np.isnan(b)
        ra, rb = rmse(gc2[ok], y[ok]), rmse(b[ok], y[ok])
        _, lo, hi, pw = boot(lab[ok].reset_index(drop=True), "sub_temp", gc2[ok], b[ok])
        win = rb < ra; passed &= win
        if s == "DIAG10": passed &= pw < P_MAX
        cold = t[ok] < 10
        p(f"{s:6s} 시드 {sd:3d}: G_C2 {ra:.5f} → {rb:.5f} ({100*(rb/ra-1):+.2f}%) CI [{lo:+.4f},{hi:+.4f}] P(worse) {pw:.4f} "
          f"{'개선' if win else '악화'} | in<10 행 RMSE {rmse(gc2[ok][cold], y[ok][cold]):.4f} → {rmse(b[ok][cold], y[ok][cold]):.4f} (n {cold.sum()})")
        curves[(s, sd)] = [100 * (rmse(((1 - k*(1-g)) * gc2 + k*(1-g) * hp2)[ok], y[ok]) / ra - 1) for k in (0.1, 0.2, 0.4, 0.6, 1.0)]
p(f"\n판정 H-P4: {'후보' if passed else '기각'}")
p("\n[진단·서술용] 게이트 가중 상한 k (w=k·(1−g)) 별 변화% — 선택에 쓰지 않음")
p("  k:            0.1     0.2     0.4     0.6     1.0")
for (s, sd), v in curves.items():
    p(f"  {s:6s} {sd:3d}  " + "  ".join(f"{x:+6.2f}" for x in v))
