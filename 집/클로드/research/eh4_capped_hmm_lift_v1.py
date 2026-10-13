# -*- coding: utf-8 -*-
"""EH4 (2026-10-14 집 클로드; 사용자: '1번으로 손해 줄이기').  사전 고정(커밋 후 실행).  EC.  과거 정보만.
EH3(6.866) 와 같은 관문·HMM 고EC 회귀·판정 칸·판정 기준.  바꾼 것 하나: 올림 상한 = 조상(관문 짝)의 23시 EC − cm_h —
  P_h = base_h + max(0, min((a + b cm_h) − cm_h, A23_anc − cm_h))  ('어제 수준을 넘겨 올리지 않기').
[사후 조정 명시] EH3 의 손해 기록(이미 조상 수준 근처로 예측한 F47_138·139·151 등)을 보고 정함; 같은 판정 칸으로 판정(그 사건들을 담은 새 배치 없음) → 통과해도 같은 자료로 고친 결과라는 한계.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "eh1_continuing_high_lift_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "eh1_continuing_high_lift_v1.py")
exec(src[:src.index("res, recd, blocks, fl = [], [], {}, []")])
__file__ = os.path.join(HERE, "eh4_capped_hmm_lift_v1.py"); H0 = 9
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
    A23 = float(EC[pick, 23]) if pick is not None else np.nan
    p_hmm = np.array([base[h] + max(0.0, min((aa + bb * cm[h]) - cm[h], A23 - cm[h])) if f[h] else base[h] for h in range(24)])
    p_c = np.array([base[h] + .22 if f[h] else base[h] for h in range(24)])
    rows.append(dict(rec=r, p2=int(r[4:]) >= 179, high=bool(np.nanmean(y) >= 1.0), hours=sum(f), pick=rid(pick), d_hmm=float(np.nansum((p_hmm - y) ** 2) - np.nansum((base - y) ** 2)),
                     d_const=float(np.nansum((p_c - y) ** 2) - np.nansum((base - y) ** 2)), base_mean=float(np.nanmean(base)), y_mean=float(np.nanmean(y)), lift_mean=float(np.mean((p_hmm - base)[np.array(f)]))))
R = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(R.round(3).sort_values("d_hmm").to_string(index=False))
false = int((~R.high).sum()); tot = R.d_hmm.sum(); v = R.d_hmm.values; ix = np.random.default_rng(20261213).integers(0, len(v), (20000, len(v))); P = float((v[ix].sum(1) >= 0).mean()); harmed = int((R.d_hmm > 1e-6).sum())
ok = false <= 1 and tot < 0 and P < .025 and harmed <= len(R) / 4
print(f"\nfired {len(R)} (pass-2 {int(R.p2.sum())}) | false fires {false} | dSSE HMM-lift {tot:+.3f} (const .22: {R.d_const.sum():+.3f}) | P(worse) {P:.4f} | harmed {harmed}/{len(R)}")
print("by pass: HMM", R.groupby("p2").d_hmm.sum().round(3).to_dict(), "| const", R.groupby("p2").d_const.sum().round(3).to_dict())
print("EH4:", "ADOPT" if ok else "FAIL")
visE = TRAIN & np.isfinite(DM) & ~np.array([rid(i) in LOCK for i in range(N)])
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; P14 = P14.sort_values(["rec", "row_id"])
bb, aa = np.polyfit([np.nanmean(BASE[q]) for q in HI], [np.nanmean(Y[q]) for q in HI], 1); print(f"\nfull high-state regression y = {aa:.3f} + {bb:.3f} c (n {len(HI)})")
for i in np.where(~TRAIN)[0]:
    f, pick = fires(i, visE); f = [x and h >= H0 for h, x in enumerate(f)]
    if not any(f): continue
    p = P14[P14.rec == rid(i)].sub_ec.values; cm = pd.Series(p).expanding().mean().values
    lift = [max(0.0, min((aa + bb * cm[h]) - cm[h], float(EC[pick, 23]) - cm[h])) if f[h] else 0.0 for h in range(24)]
    print(f"   eval {rid(i)}: hours {sum(f)} from {min(h for h in range(24) if f[h])}h, ancestor {rid(pick)} (EC {DM[pick]:.2f}), 14th mean {p.mean():.3f}, mean lift on fired hours {np.mean([l for l, x in zip(lift, f) if x]):+.3f}")
