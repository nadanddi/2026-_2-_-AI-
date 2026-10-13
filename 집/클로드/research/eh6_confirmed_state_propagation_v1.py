# -*- coding: utf-8 -*-
"""EH6 (2026-10-14 집 클로드; 사용자: 'EH6 진행해봐').  사전 고정(커밋 후 실행).  EC.  과거 정보만.
EH3(6.866, 유망 후보): 관문(9시 이후, 1~2일 전 같은 운영의 정답 고EC 짝) + HMM 고EC 회귀 올림.  평가에선 F13_221 하나만 걸림.
EH6 = 확인된 고EC 상태를 정답 없는 기록 한 번을 거쳐 넘김(HMM 상태 전달):
  '확인 기록' m = 정답이 없지만(평가에선 같은 ID 앞 번호 평가 기록) 그 자신이 하루 전체 기준 EH 관문을 통과(1~2일 전 같은 운영(23시 거리 <= KEEP_23, 확신)의 정답 고EC 짝 있음).
  대상 b 의 시각 h(>= 9): 후보 = 날짜 d−1·d−2(같은 구간)의 보이는 정답 고EC 기록 ∪ 확인 기록; 0..h 지문 최근접이 확신(2등/1등 >= 1.5) & 거리 <= KEEP_h 이고
    그게 확인 기록 m 이면 → 올림 = .6 x EH3 올림( max(0, a + b cm_h − cm_h), 고EC 회귀는 대상·이웃 빼고 적합 ).
    (.6 = DG1 6.856 고EC 지속 1일 .88 → 2일 .56 의 비; 이 결과로 고르지 않음.)  최근접이 정답 고EC 기록이면 EH3 그대로(직접 발동).  확인 기록 두 번 거치기는 안 함.
학습 시험(평가 덩어리 안쪽 흉내, 평가형 가시성): 학습 정답 기록 b 마다 b 와 잠금을 숨기고, 날짜 d−1 후보 중 b 와 하루 지문(23시)이 가장 가까운 기록 m* 의 정답도 가림
  (m* = '정답 없는 어제'); m* 가 확인 기록이면 위 규칙 적용.  기본 = DIAG10 CUR OOF.
판정(미리 정함; 넘겨받기로 걸린 기록만, 직접 발동 기록 제외): 가짜 발동(실제 보통) <= max(1, 10%) AND dSSE 합 < 0 AND 기록 재표집 P(악화) < .025 AND 손해 <= 1/4.
서술: 직접+넘겨받기 합친 결과, 평가 적용(같은 ID 앞 번호 평가 기록을 확인 기록 후보로).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "eh3_confirmed_high_hmm_lift_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "eh3_confirmed_high_hmm_lift_v1.py")
exec(src[:src.index("rows = []\nVIS0")])   # EH1 machinery (dist, tdate, KEEP, fires), A, Y, BASE, LOCK, HI, reg_without
__file__ = os.path.join(HERE, "eh6_confirmed_state_propagation_v1.py"); H0 = 9; MULT = .6
VIS0 = TRAIN & np.isfinite(DM) & ~np.array([rid(i) in LOCK for i in range(N)])
def date_of(i, vis):
    if TRAIN[i] and np.isfinite(DAYT[i]): return DAYT[i], SEGT[i]
    t = tdate(i, vis); return t if t is not None else (np.nan, np.nan)
def confirmed(m, vis, mdate):
    d, s = mdate
    if not np.isfinite(d): return False
    cand = [a for a in np.where(vis)[0] if a != m and SEGT[a] == s and DAYT[a] in (d - 1, d - 2) and DM[a] >= 1.0]
    if not cand: return False
    dd = sorted((dist(a, m, 23), a) for a in cand)
    return (len(dd) == 1 or dd[1][0] >= 1.5 * max(dd[0][0], 1e-9)) and dd[0][0] <= KEEP[23]
def gate(b, vis, unlab):
    """unlab: dict m -> date for allowed unlabelled records.  returns per-hour list of (kind, partner) kind in {None, 'direct', 'prop'}"""
    t = tdate(b, vis)
    if t is None: return [(None, None)] * 24
    lab = [a for a in np.where(vis)[0] if a != b and SEGT[a] == t[1] and DAYT[a] in (t[0] - 1, t[0] - 2) and DM[a] >= 1.0]
    un = [m for m, md in unlab.items() if md[1] == t[1] and md[0] in (t[0] - 1, t[0] - 2) and confirmed(m, vis, md)]
    cand = [(a, "direct") for a in lab] + [(m, "prop") for m in un]; out = []
    for h in range(24):
        if h < H0 or not cand: out.append((None, None)); continue
        dd = sorted((dist(a, b, h), a, k) for a, k in cand)
        ok = (len(dd) == 1 or dd[1][0] >= 1.5 * max(dd[0][0], 1e-9)) and dd[0][0] <= KEEP[h]
        out.append((dd[0][2], dd[0][1]) if ok else (None, None))
    return out
rows = []
for r in Y:
    b = idx[r]; base = BASE[r]
    if not np.isfinite(base).all(): continue
    vis = VIS0.copy(); vis[b] = False; t = tdate(b, vis)
    if t is None: continue
    y1 = [a for a in np.where(vis)[0] if SEGT[a] == t[1] and DAYT[a] == t[0] - 1]
    if not y1: continue
    m_star = min(y1, key=lambda a: dist(a, b, 23)); vis[m_star] = False          # hide 'yesterday' label
    g = gate(b, vis, {m_star: (DAYT[m_star], SEGT[m_star])})
    kinds = [k for k, _ in g]
    if not any(k is not None for k in kinds): continue
    bb, aa = reg_without(r); cm = pd.Series(base).expanding().mean().values; y = Y[r]
    lift = np.array([(MULT if k == "prop" else 1.0) * max(0.0, (aa + bb * cm[h]) - cm[h]) if k is not None else 0.0 for h, k in enumerate(kinds)])
    p = base + lift
    rows.append(dict(rec=r, kind="prop" if "prop" in kinds else "direct", hours=sum(k is not None for k in kinds), m_star=rid(m_star), m_star_high=bool(DM[m_star] >= 1.0),
                     high=bool(np.nanmean(y) >= 1.0), p2=int(r[4:]) >= 179, dsse=float(np.nansum((p - y) ** 2) - np.nansum((base - y) ** 2)), lift_mean=float(lift[lift > 0].mean()) if (lift > 0).any() else 0.0,
                     base_mean=float(np.nanmean(base)), y_mean=float(np.nanmean(y))))
R = pd.DataFrame(rows); pd.set_option("display.width", 220); print(R.round(3).sort_values(["kind", "dsse"]).to_string(index=False))
Pr = R[R.kind == "prop"]
if len(Pr):
    false = int((~Pr.high).sum()); tot = Pr.dsse.sum(); v = Pr.dsse.values; ix = np.random.default_rng(20261214).integers(0, len(v), (20000, len(v))); P = float((v[ix].sum(1) >= 0).mean()); harmed = int((Pr.dsse > 1e-6).sum())
    ok = false <= max(1, .1 * len(Pr)) and tot < 0 and P < .025 and harmed <= len(Pr) / 4
    print(f"\nPROPAGATED fires {len(Pr)} | false {false} | dSSE {tot:+.3f} | P(worse) {P:.4f} | harmed {harmed} -> {'ADOPT' if ok else 'FAIL'}")
else: ok = False; print("\nno propagated fires")
print(f"all fires {len(R)} (direct {int((R.kind == 'direct').sum())}) | dSSE total {R.dsse.sum():+.3f}")
print("EH6:", "ADOPT" if ok else "FAIL")
# eval application
visE = VIS0.copy(); EV_UN = {i: date_of(i, visE) for i in np.where(~TRAIN)[0]}
P14 = pd.read_csv(os.path.join(env.ROOT, "제출", "14회차_2026-10-13(나)", "submission_23.csv")); P14["rec"] = P14.row_id.str[:7]; P14 = P14.sort_values(["rec", "row_id"])
bb, aa = np.polyfit([np.nanmean(BASE[q]) for q in HI], [np.nanmean(Y[q]) for q in HI], 1)
print("\neval application:")
for i in np.where(~TRAIN)[0]:
    unl = {m: d for m, d in EV_UN.items() if m != i and FARM[m] == FARM[i] and NUM[m] < NUM[i]}   # rule: same-ID earlier eval records only
    g = gate(i, visE, unl); kinds = [k for k, _ in g]
    if not any(k is not None for k in kinds): continue
    p = P14[P14.rec == rid(i)].sub_ec.values; cm = pd.Series(p).expanding().mean().values
    lift = [(MULT if k == "prop" else 1.0) * max(0.0, (aa + bb * cm[h]) - cm[h]) if k is not None else 0.0 for h, k in enumerate(kinds)]
    partners = sorted({rid(pp) for k, pp in g if k is not None})
    print(f"   {rid(i)}: kinds {sorted({k for k in kinds if k})}, hours {sum(k is not None for k in kinds)}, partner {partners}, 14th mean {p.mean():.3f}, mean lift on fired hours {np.mean([l for l in lift if l > 0]) if any(l > 0 for l in lift) else 0:+.3f}")
