# -*- coding: utf-8 -*-
"""EH5 (2026-10-14 집 클로드; 사용자: 'EH3 더 발전시켜 볼 만하지 않아?').  사전 고정(커밋 후 실행).  EC.  과거 정보만.
EH3(6.866)·EH4(6.867): 손해와 이득이 '오늘이 어제보다 높은가' 한 축에 있음 — 상한(EH4)은 맞바꿈만.  구조 변경: 올림 크기를 어제 수준까지 보는 2변수 회귀로.
관문·판정 칸·판정 기준은 EH3 와 같음.  보정: 학습 '확인 쌍'(관문 조건을 만족하는 학습 기록 = 1~2일 전 같은 운영의 정답 고EC 짝이 있는 고EC 기록 — 관문을 하루 전체로 적용해
  모은 것, 대상·같은 농장 ±1 제외)에서 y_day = a + b c + d A23 (c = 기본 예측 하루 평균, A23 = 짝 23시 EC) 를 맞추고,
  관문 통과 시각에 P_h = base_h + max(0, (a + b cm_h + d A23) − cm_h).  확인 쌍이 8 개 미만이면 EH3 의 1변수 회귀로.
[한계] 판정 칸은 EH3·EH4 와 같음(이 사건들을 담은 새 배치 없음).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "eh1_continuing_high_lift_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "eh1_continuing_high_lift_v1.py")
exec(src[:src.index("res, recd, blocks, fl = [], [], {}, []")])
__file__ = os.path.join(HERE, "eh5_two_var_lift_v1.py"); H0 = 9
A = pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz")); A = A[A.is_train & ~A.lock].sort_values(["rec", "hour"])
Y = {r: g.y.values for r, g in A.groupby("rec")}; BASE = {r: g.base_oof.values for r, g in A.groupby("rec")}
LOCK = set(pd.read_csv(os.path.join(HERE, "cp2_tracks_v1.csv")).query("lock").rec)
HI = [r for r in Y if np.nanmean(Y[r]) >= 1.0 and np.isfinite(BASE[r]).all()]
VIS0_ALL = TRAIN & np.isfinite(DM) & ~np.array([rid(i) in LOCK for i in range(N)])
def reg_without(r):
    f, n = r[:3], int(r[4:]); use = [q for q in HI if not (q[:3] == f and abs(int(q[4:]) - n) <= 1)]
    c = np.array([np.nanmean(BASE[q]) for q in use]); yy = np.array([np.nanmean(Y[q]) for q in use]); return np.polyfit(c, yy, 1)
CONF = {}
for q in HI:
    vq = VIS0_ALL.copy(); vq[idx[q]] = False; t = tdate(idx[q], vq)
    if t is None: continue
    cand = [a for a in np.where(vq)[0] if SEGT[a] == t[1] and DAYT[a] in (t[0] - 1, t[0] - 2) and DM[a] >= 1.0]
    if not cand: continue
    d = sorted((dist(a, idx[q], 23), a) for a in cand)
    if (len(d) == 1 or d[1][0] >= 1.5 * max(d[0][0], 1e-9)) and d[0][0] <= KEEP[23]: CONF[q] = float(EC[d[0][1], 23])
def reg2_without(r):
    f, n = r[:3], int(r[4:]); use = [q for q in CONF if not (q[:3] == f and abs(int(q[4:]) - n) <= 1)]
    if len(use) < 8: return None
    X = np.c_[np.ones(len(use)), [np.nanmean(BASE[q]) for q in use], [CONF[q] for q in use]]; yy = np.array([np.nanmean(Y[q]) for q in use])
    return np.linalg.lstsq(X, yy, rcond=None)[0]
rows = []
VIS0 = TRAIN & np.isfinite(DM) & ~np.array([rid(i) in LOCK for i in range(N)])
for r in Y:
    b = idx[r]; base = BASE[r]
    if not np.isfinite(base).all(): continue
    vis = VIS0.copy(); vis[b] = False
    f, pick = fires(b, vis); f = [x and h >= H0 for h, x in enumerate(f)]
    if not any(f): continue
    bb, aa = reg_without(r); cm = pd.Series(base).expanding().mean().values; y = Y[r]; A23 = float(EC[pick, 23]); co = reg2_without(r)
    if co is None: p_hmm = np.array([base[h] + max(0.0, (aa + bb * cm[h]) - cm[h]) if f[h] else base[h] for h in range(24)])
    else: p_hmm = np.array([base[h] + max(0.0, (co[0] + co[1] * cm[h] + co[2] * A23) - cm[h]) if f[h] else base[h] for h in range(24)])
    p_c = np.array([base[h] + .22 if f[h] else base[h] for h in range(24)])
    rows.append(dict(rec=r, p2=int(r[4:]) >= 179, high=bool(np.nanmean(y) >= 1.0), hours=sum(f), pick=rid(pick), d_hmm=float(np.nansum((p_hmm - y) ** 2) - np.nansum((base - y) ** 2)),
                     d_const=float(np.nansum((p_c - y) ** 2) - np.nansum((base - y) ** 2)), base_mean=float(np.nanmean(base)), y_mean=float(np.nanmean(y)), lift_mean=float(np.mean((p_hmm - base)[np.array(f)]))))
R = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(R.round(3).sort_values("d_hmm").to_string(index=False))
false = int((~R.high).sum()); tot = R.d_hmm.sum(); v = R.d_hmm.values; ix = np.random.default_rng(20261213).integers(0, len(v), (20000, len(v))); P = float((v[ix].sum(1) >= 0).mean()); harmed = int((R.d_hmm > 1e-6).sum())
ok = false <= 1 and tot < 0 and P < .025 and harmed <= len(R) / 4
print(f"\nfired {len(R)} (pass-2 {int(R.p2.sum())}) | false fires {false} | dSSE HMM-lift {tot:+.3f} (const .22: {R.d_const.sum():+.3f}) | P(worse) {P:.4f} | harmed {harmed}/{len(R)}")
print("by pass: HMM", R.groupby("p2").d_hmm.sum().round(3).to_dict(), "| const", R.groupby("p2").d_const.sum().round(3).to_dict())
print("EH5:", "ADOPT" if ok else "FAIL")
X = np.c_[np.ones(len(CONF)), [np.nanmean(BASE[q]) for q in CONF], list(CONF.values())]; co_full = np.linalg.lstsq(X, np.array([np.nanmean(Y[q]) for q in CONF]), rcond=None)[0]; print(f"confirmed pairs {len(CONF)} | full 2-var fit y = {co_full[0]:.3f} + {co_full[1]:.3f} c + {co_full[2]:.3f} A23")
visE = TRAIN & np.isfinite(DM) & ~np.array([rid(i) in LOCK for i in range(N)])
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; P14 = P14.sort_values(["rec", "row_id"])
bb, aa = np.polyfit([np.nanmean(BASE[q]) for q in HI], [np.nanmean(Y[q]) for q in HI], 1); print(f"\nfull high-state regression y = {aa:.3f} + {bb:.3f} c (n {len(HI)})")
for i in np.where(~TRAIN)[0]:
    f, pick = fires(i, visE); f = [x and h >= H0 for h, x in enumerate(f)]
    if not any(f): continue
    p = P14[P14.rec == rid(i)].sub_ec.values; cm = pd.Series(p).expanding().mean().values
    A23 = float(EC[pick, 23]); lift = [max(0.0, (co_full[0] + co_full[1] * cm[h] + co_full[2] * A23) - cm[h]) if f[h] else 0.0 for h in range(24)]
    print(f"   eval {rid(i)}: hours {sum(f)} from {min(h for h in range(24) if f[h])}h, ancestor {rid(pick)} (EC {DM[pick]:.2f}), 14th mean {p.mean():.3f}, mean lift on fired hours {np.mean([l for l, x in zip(lift, f) if x]):+.3f}")
