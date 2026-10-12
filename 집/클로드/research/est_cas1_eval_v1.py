# -*- coding: utf-8 -*-
"""EST-CAS1 (2026-10-13 집 클로드; 사용자: 'CAS 평가에 적용하면 예상 점수') 서술: 13회차 EC 위에 CAS 계단을 평가 60 에 적용한 변화와 점수 범위.
v7 앵커 기록 = 13회차 그대로. 나머지: v15(학습 전체) p0 ≥ .8 → 전 시각 EA2(A); 아니면 앞뒤 합의(v15 p0 ≥ .5 의 A, NXL1(학습 전체) p0 ≥ .8 의 B_h, |A−B_h| ≤ .15 → (A+B_h)/2) 해당 행.
기준 예측 sg = 13회차 EC(앵커 없는 기록에선 12회차와 같음). 점수: 13회차 EC .1279 (LB). MSE_new = S² + mean(d²) + 2·mean(d·err_old) → 상관 범위·본전 상관,
  그리고 '앵커가 맞으면 그 기록의 새 오차 ≈ 검증에서 맞은 앵커 기록 RMSE(.03)' 가정의 낙관 추정."""
import os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "nxl1_next_day_null_linker_v1.py"), encoding="utf-8").read()
exec(src[:src.index("L7 = pd.read_csv")])
P = pd.read_csv(glob.glob(os.path.join(env.ROOT, "제출", "13회차_2026-10-12(나)", "*.csv"))[0]); P["rec"] = P.row_id.str[:7]; P["h"] = P.row_id.str[8:].astype(int)
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = {r for r, q, l, d in zip(E7.row, E7.p_h0, E7.prev_is_labelled.astype(bool), E7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
V15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_eval.csv")).set_index("rec")
W = weights("ALL", np.zeros(N, bool))
new = P.sub_ec.values.copy(); log = []
for r, g in P.groupby("rec"):
    if r[:3] not in ("F13", "F47") or r in v7e: continue
    ix = g.sort_values("h").index.values; sg = P.sub_ec.values[ix]; cm = np.cumsum(sg) / np.arange(1, 25)
    A_ = None; p0 = 0
    if r in V15.index and bool(V15.lab0[r]): A_ = EC[idx[V15.prev0[r]], 23]; p0 = V15.p0[r]
    if A_ is not None and p0 >= .8:
        new[ix] = np.clip(sg + (A_ - cm), lo_, hi_); log.append((r, "v15>=.8", V15.prev0[r], round(p0, 2), round(A_, 3), 24)); continue
    if A_ is None or p0 < .5: continue
    Bh, pk = next_anchor(r, TRAIN.copy(), W); m = np.isfinite(Bh) & (np.abs(Bh - A_) <= .15)
    if m.any():
        new[ix[m]] = np.clip(sg[m] + ((A_ + Bh[m]) / 2 - cm[m]), lo_, hi_); log.append((r, "agree", V15.prev0[r] + "/" + ",".join(sorted({p for p in pk if p})), round(p0, 2), round(float((A_ + np.nanmean(Bh[m])) / 2), 3), int(m.sum())))
d = new - P.sub_ec.values; F = P.row_id.str[:3].isin(["F13", "F47"]).values; n = int(F.sum()); d = d[F]
S = .1279; rms = float(np.sqrt(np.mean(d ** 2)))
print("changed records:"); [print("  ", x) for x in log]
print(f"rows {n} | changed rows {int((np.abs(d) > 1e-9).sum())} | RMS change over all rows {rms:.4f} | breakeven corr {rms / (2 * S):.3f}")
for rho in (-.5, -.3, -.1, 0, .1, .3):
    mse = S ** 2 + np.mean(d ** 2) - 2 * rho * rms * S
    print(f"   corr(d, -err) {rho:+.1f}: EC {np.sqrt(max(mse, 0)):.4f}")
ch = np.abs(d) > 1e-9; dd = d[ch]
for e_new in (.03, .05):
    sse_old = np.sum(dd ** 2) + 0.0  # if anchor correct, old error on those rows ≈ −d (anchor ≈ truth)
    mse = S ** 2 + (len(dd) * e_new ** 2 - sse_old) / n
    print(f"   optimistic (anchors correct, new row error {e_new}): EC {np.sqrt(mse):.4f}")
