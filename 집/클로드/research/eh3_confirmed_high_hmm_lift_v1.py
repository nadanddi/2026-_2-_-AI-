# -*- coding: utf-8 -*-
"""EH3 (2026-10-14 집 클로드; 사용자: '1번(HMM 에 확인 관문)에 써보자').  사전 고정(커밋 후 실행).  EC.  과거 정보만.
관문 = EH2 와 같음(h >= 9; 학습 달력 d−1·d−2 의 보이는 정답 고EC 후보 중 0..h 지문 최근접이 확신(2등/1등 >= 1.5) & 거리 <= KEEP_h).
보정 = HMM 고EC 상태 회귀(DS 계열의 상태별 직선 회귀를 P(H) = 1 로): 학습 고EC 기록(하루 평균 >= 1.0)에서 '기본 예측 하루 평균 c → 실제 하루 평균 y' 직선 y = a + b c 를
  대상 기록과 그 같은 농장 ±1 번호를 뺀 자료로 맞추고(기록 하나씩 빼기), 관문 통과 시각에 P_h = base_h + max(0, (a + b cm_h) − cm_h)  (cm_h = base 0..h 누적평균; 내리지는 않음).
  비평(6.865): 상수 +.22 는 이미 높게 맞힌 기록에서 과함 → 예측 수준에 따라 줄어드는 보정.
판정 칸(비평 제안 — 평가형 가시성): 학습 정답 기록 전체(1·2차, 잠금 제외) 각각을 대상으로, 보이는 정답 = 학습 정답 − 대상 − 잠금(다른 농장 가까운 날도 보임, 평가처럼).
  기본 = DIAG10 CUR OOF(cp2 base_oof, 시드 평균).
  통과(미리 정함): 가짜 발동(관문 통과한 실제 보통 기록) <= 1  AND  발동 기록 dSSE 합 < 0 이고 기록 재표집 부트스트랩 P(악화) < .025  AND  손해 기록 수 <= 발동 기록의 1/4.
서술: 같은 규칙의 EV(폴드 숨김) 결과, 1차/2차 분해, 상수 +.22(EH2)와 비교, 평가 발동 기록.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "eh1_continuing_high_lift_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "eh1_continuing_high_lift_v1.py")
exec(src[:src.index("res, recd, blocks, fl = [], [], {}, []")])
__file__ = os.path.join(HERE, "eh3_confirmed_high_hmm_lift_v1.py"); H0 = 9
A = pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz")); A = A[A.is_train & ~A.lock].sort_values(["rec", "hour"])
Y = {r: g.y.values for r, g in A.groupby("rec")}; BASE = {r: g.base_oof.values for r, g in A.groupby("rec")}
LOCK = set(pd.read_csv(os.path.join(HERE, "cp2_tracks_v1.csv")).query("lock").rec)
HI = [r for r in Y if np.nanmean(Y[r]) >= 1.0 and np.isfinite(BASE[r]).all()]
def reg_without(r):
    f, n = r[:3], int(r[4:]); use = [q for q in HI if not (q[:3] == f and abs(int(q[4:]) - n) <= 1)]
    c = np.array([np.nanmean(BASE[q]) for q in use]); yy = np.array([np.nanmean(Y[q]) for q in use]); return np.polyfit(c, yy, 1)
rows = []
VIS0 = TRAIN & np.isfinite(DM) & ~np.array([rid(i) in LOCK for i in range(N)])
for r in Y:
    b = idx[r]; base = BASE[r]
    if not np.isfinite(base).all(): continue
    vis = VIS0.copy(); vis[b] = False
    f, pick = fires(b, vis); f = [x and h >= H0 for h, x in enumerate(f)]
    if not any(f): continue
    bb, aa = reg_without(r); cm = pd.Series(base).expanding().mean().values; y = Y[r]
    p_hmm = np.array([base[h] + max(0.0, (aa + bb * cm[h]) - cm[h]) if f[h] else base[h] for h in range(24)])
    p_c = np.array([base[h] + .22 if f[h] else base[h] for h in range(24)])
    rows.append(dict(rec=r, p2=int(r[4:]) >= 179, high=bool(np.nanmean(y) >= 1.0), hours=sum(f), pick=rid(pick), d_hmm=float(np.nansum((p_hmm - y) ** 2) - np.nansum((base - y) ** 2)),
                     d_const=float(np.nansum((p_c - y) ** 2) - np.nansum((base - y) ** 2)), base_mean=float(np.nanmean(base)), y_mean=float(np.nanmean(y)), lift_mean=float(np.mean((p_hmm - base)[np.array(f)]))))
R = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(R.round(3).sort_values("d_hmm").to_string(index=False))
false = int((~R.high).sum()); tot = R.d_hmm.sum(); v = R.d_hmm.values; ix = np.random.default_rng(20261213).integers(0, len(v), (20000, len(v))); P = float((v[ix].sum(1) >= 0).mean()); harmed = int((R.d_hmm > 1e-6).sum())
ok = false <= 1 and tot < 0 and P < .025 and harmed <= len(R) / 4
print(f"\nfired {len(R)} (pass-2 {int(R.p2.sum())}) | false fires {false} | dSSE HMM-lift {tot:+.3f} (const .22: {R.d_const.sum():+.3f}) | P(worse) {P:.4f} | harmed {harmed}/{len(R)}")
print("by pass: HMM", R.groupby("p2").d_hmm.sum().round(3).to_dict(), "| const", R.groupby("p2").d_const.sum().round(3).to_dict())
print("EH3:", "ADOPT" if ok else "FAIL")
visE = TRAIN & np.isfinite(DM) & ~np.array([rid(i) in LOCK for i in range(N)])
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; P14 = P14.sort_values(["rec", "row_id"])
bb, aa = np.polyfit([np.nanmean(BASE[q]) for q in HI], [np.nanmean(Y[q]) for q in HI], 1); print(f"\nfull high-state regression y = {aa:.3f} + {bb:.3f} c (n {len(HI)})")
for i in np.where(~TRAIN)[0]:
    f, pick = fires(i, visE); f = [x and h >= H0 for h, x in enumerate(f)]
    if not any(f): continue
    p = P14[P14.rec == rid(i)].sub_ec.values; cm = pd.Series(p).expanding().mean().values
    lift = [max(0.0, (aa + bb * cm[h]) - cm[h]) if f[h] else 0.0 for h in range(24)]
    print(f"   eval {rid(i)}: hours {sum(f)} from {min(h for h in range(24) if f[h])}h, ancestor {rid(pick)} (EC {DM[pick]:.2f}), 14th mean {p.mean():.3f}, mean lift on fired hours {np.mean([l for l, x in zip(lift, f) if x]):+.3f}")
