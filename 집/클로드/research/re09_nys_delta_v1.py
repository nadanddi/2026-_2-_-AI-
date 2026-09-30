# -*- coding: utf-8 -*-
"""재분석 9 — 사전 고정 가설 H-NY1 (2026-10-01, 집 클로드). ※ re08 진단에서 사후 도출한 가설.

근거 (re08 진단, 판정 아님): MASK 안 커널 멤버 nys(10%)의 추운 행 편향이 DIAG10 +0.77, EXT10 +2.9,
  EXT12 +9.2℃로 폭주. 대상을 sub_temp − ph_in_temp_3로 바꾼 nys_dt는 +0.23~+0.34.
  트리 res는 대상 변경 시 DIAG10 악화(0.718→0.743) → H-DT1 기각 원인.
가설 H-NY1: MASK = 0.65 res + 0.25 ridge + 0.10 nys 에서 nys만 nys_dt로 교체 (그 외 전부 동일).
  re08이 저장한 멤버 OOF(local/re08_delta_target_v1_oof.npz)를 그대로 조합, 재적합 없음.
  (이 조합의 점수는 이 스크립트 실행 전 계산·열람하지 않았음.)

판정 규칙 (실행 전 고정):
  1. DIAG10, EXT10(전체 행), EXT12 × MASK 시드 (7, 101) 여섯 칸 모두 G_C2′ < G_C2 (같은 시드, CODEX·TabPFN 저장본).
  2. DIAG10 P(worse) < 0.0083 (캠페인 세 번째 가설, 본페로니 0.025/3) — 두 시드 모두.
  3. 둘 다 만족하면 "후보", 아니면 기각.
진단: 온도구간·온실별 RMSE 변화(DIAG10·EXT10), G_C2 오차 상관.
결과: local/re09_nys_delta_v1.txt
"""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from harness import load
from common import rmse
from screen_v6 import boot

P_MAX = 0.025 / 3
OUT = os.path.join(env.LOCAL, "re09_nys_delta_v1.txt")
_f = open(OUT, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _f.write(s + "\n")


_, lab, _ = load()
m = np.load(os.path.join(env.LOCAL, "re08_delta_target_v1_oof.npz"), allow_pickle=True)
z = np.load(os.path.join(env.LOCAL, "temp_mask_v1_oof.npz"), allow_pickle=True)
assert (m["row_id"] == lab.row_id.values).all() and (z["row_id"] == lab.row_id.values).all()
y = lab.sub_temp.values
t = lab.in_temp.values
g = np.where(np.isnan(t), 1.0, np.clip((t - 8.0) / 2.0, 0, 1))
passed = True
for s in ("DIAG10", "EXT10", "EXT12"):
    cx = np.nanmean([z[f"{s}__CODEX__726"], z[f"{s}__CODEX__727"]], axis=0)
    pfn = np.load(os.path.join(env.LOCAL, f"web_tabpfn_v2_temp_{s}.npy")).mean(0)
    for sd in (7, 101):
        old = m[f"{s}__MASK__{sd}"]
        new = 0.65 * m[f"{s}__res__{sd}"] + 0.25 * m[f"{s}__ridge__{sd}"] + 0.10 * m[f"{s}__nys_dt__{sd}"]
        blend = lambda b: (0.6 - 0.2 * (1 - g)) * b + (0.2 + 0.4 * (1 - g)) * cx + 0.2 * g * pfn
        a, b = blend(old), blend(new)
        ok = ~np.isnan(a) & ~np.isnan(b)
        ra, rb = rmse(a[ok], y[ok]), rmse(b[ok], y[ok])
        pr, lo, hi, pw = boot(lab[ok].reset_index(drop=True), "sub_temp", a[ok], b[ok])
        win = rb < ra
        passed &= win
        if s == "DIAG10":
            passed &= pw < P_MAX
        p(f"{s:6s} 시드 {sd:3d}: G_C2 {ra:.5f} → G_C2′ {rb:.5f} ({100*(rb/ra-1):+.2f}%) "
          f"CI [{lo:+.4f},{hi:+.4f}] P(worse) {pw:.4f} {'개선' if win else '악화'}")
        if sd == 7 and s in ("DIAG10", "EXT10"):
            d = lab.loc[ok, ["farm", "in_temp"]].assign(ea=(a - y)[ok], eb=(b - y)[ok])
            band = pd.cut(d.in_temp, [-99, 6, 8, 10, 12, 15, 99])
            tb = d.groupby(band, observed=True).apply(
                lambda q: pd.Series({"n": len(q), "G_C2": rmse(q.ea, 0), "G_C2′": rmse(q.eb, 0)}))
            tb["변화%"] = 100 * (tb["G_C2′"] / tb["G_C2"] - 1)
            p(tb.round(4).to_string())
            fb = d.groupby("farm").apply(lambda q: pd.Series({"G_C2": rmse(q.ea, 0), "G_C2′": rmse(q.eb, 0)}))
            p(fb.round(4).to_string())
p(f"\n판정: {'후보' if passed else '기각'}")
_f.close()
