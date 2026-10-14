# -*- coding: utf-8 -*-
"""TY1 (2026-10-14 집 클로드; 사용자: '온도쪽에서 해보고 싶은 가설 해봐').  서술(판정 아님).  온도.
가설: 지금 앵커는 전날 23시 배지 한 점(β 0시 .76 → 낮 .1~.2).  같은 동의 '전날 같은 시각 잔차'(배지가 그 동에서 모델보다 얼마나 높/낮았나)가
  오늘 같은 시각 잔차를 설명하면, 낮 시각에 새 정보(동별 하루 모양 — 멀칭·급액·배지 위치 등 동 고유 차이).
자료: EV 하네스 저장 최종 예측(ev_all.pkl, v22_a 4칸 평균 = 13회차형) 잔차 r = y − p.  쌍 = truth.pkl cert 1일 연결(둘 다 정답), 30 거친 날 제외 안 함(서술).
보는 것: 시각 h 별 corr(r_prev[h], r_today[h]), 회귀 기울기, 그리고 '전날 23시 잔차' 통제 후 부분 상관(이미 앵커가 쓰는 정보 빼고 남는 것).
  1차/2차, F13/F47 분해.  상한 효과: 잔차를 β_h · r_prev[h] 로 뺄 때(β 는 쌍 하나 빼기로 적합) RMSE 변화.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import FARM, NUM, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
DD = pd.read_pickle(os.path.join(HERE, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); DD["rec"] = DD.row_id.str[:7]
DD["p"] = DD[[f"v22_a_{c}" for c in ("7A", "7B", "101A", "101B")]].mean(1); DD["r"] = DD.sub_temp - DD.p
R = DD.pivot_table(index="rec", columns="hour", values="r")
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); P = [(rid(int(a)), rid(int(b))) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c]
P = [(a, b) for a, b in P if a in R.index and b in R.index]; print("cert 1-day pairs with both residual curves:", len(P))
A = R.loc[[a for a, _ in P]].values; B = R.loc[[b for _, b in P]].values; p2 = np.array([int(b[4:]) >= 179 for _, b in P]); f47 = np.array([b[:3] == "F47" for _, b in P])
def pcorr(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 10: return np.nan
    rx = x[m] - np.polyval(np.polyfit(z[m], x[m], 1), z[m]); ry = y[m] - np.polyval(np.polyfit(z[m], y[m], 1), z[m]); return float(np.corrcoef(rx, ry)[0, 1])
rows = []
for h in range(24):
    x, y = A[:, h], B[:, h]; m = np.isfinite(x) & np.isfinite(y)
    rows.append(dict(h=h, n=int(m.sum()), corr=float(np.corrcoef(x[m], y[m])[0, 1]), slope=float(np.polyfit(x[m], y[m], 1)[0]),
                     pcorr_given_prev23=pcorr(x, y, A[:, 23]), corr_p2=float(np.corrcoef(x[m & p2], y[m & p2])[0, 1]) if (m & p2).sum() > 5 else np.nan,
                     corr_F47=float(np.corrcoef(x[m & f47], y[m & f47])[0, 1]), corr_F13=float(np.corrcoef(x[m & ~f47], y[m & ~f47])[0, 1]), sd_today=float(np.nanstd(y))))
S = pd.DataFrame(rows); pd.set_option("display.width", 200); print(S.round(3).to_string(index=False))
# day-mean version
am, bm = np.nanmean(A, 1), np.nanmean(B, 1); print(f"\nday-mean residual corr {np.corrcoef(am, bm)[0, 1]:.3f} | partial given prev 23h residual {pcorr(am, bm, A[:, 23]):.3f}")
# leave-one-pair-out upper bound: today r_h -> r_h - beta_h * prev r_h
new = B.copy()
for k in range(len(P)):
    for h in range(24):
        m = np.isfinite(A[:, h]) & np.isfinite(B[:, h]); m[k] = False
        if np.isfinite(A[k, h]) and np.isfinite(B[k, h]):
            beta = np.sum(A[m, h] * B[m, h]) / np.sum(A[m, h] ** 2); new[k, h] = B[k, h] - beta * A[k, h]
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
for nm, msk in (("all", np.ones(len(P), bool)), ("pass-2", p2), ("F47", f47), ("F13", ~f47)):
    print(f"{nm:7s} n {msk.sum():3d}: RMSE {rm(B[msk]):.4f} -> {rm(new[msk]):.4f} ({100 * (rm(new[msk]) / rm(B[msk]) - 1):+.1f}%) | day hours 9-17 {rm(B[msk][:, 9:18]):.4f} -> {rm(new[msk][:, 9:18]):.4f}")
