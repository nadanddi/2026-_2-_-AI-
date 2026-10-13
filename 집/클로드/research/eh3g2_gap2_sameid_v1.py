# -*- coding: utf-8 -*-
"""EH3-G2 (2026-10-14 집 클로드; 비평 E1, 사용자: '비평대로 실험 이어가봐').  사전 고정(커밋 후 실행).  EC.  과거 정보만.
비평(6.870 예정): EH3 하네스 발동 20건 중 11건은 간격 1(d−1 이 보임) — 평가 F13_221 은 d−1(F13_219)이 지워진 '간격 2 · 같은 ID' 경우.
  → 평가 상황을 흉내 낸 하네스: 학습 정답 기록 b 가 같은 동 전날 cert 짝 a(CERT[b])를 가지면, a 를 숨기고(지워진 기록 흉내),
    후보 = 보이는 정답 고EC 기록 중 날짜 d−2(같은 구간) & 같은 ID(농장) 만.  나머지는 EH3 그대로(h >= 9, 확신 1.5, KEEP_h, 고EC 회귀 올림, 내리지 않음).
  기본 = DIAG10 CUR OOF(cp2 base_oof).  보이는 정답 = 학습 정답 − b − a − 잠금.
판정(미리 정함; 작은 표본 길): 가짜 발동(실제 보통) 0 AND 발동 dSSE 합 < 0 AND 손해 기록 <= 1.
  서술: 기록 재표집 P, 조상(pick) 군집 재표집 P, 소진 기록 장부(F13_231·F13_233·F47_216·F47_229·F47_241·F13_217·F13_243·F47_231) 제외 dSSE.
  꼬리표: 같은 31개 고EC 기록 위의 판정(EH3~EH6 와 같은 자료) → '같은 자료 사후 성격'.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "eh3_confirmed_high_hmm_lift_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "eh3_confirmed_high_hmm_lift_v1.py")
exec(src[:src.index("rows = []\nVIS0")])
__file__ = os.path.join(HERE, "eh3g2_gap2_sameid_v1.py"); H0 = 9
LEDGER = {"F13_231", "F13_233", "F47_216", "F47_229", "F47_241", "F13_217", "F13_243", "F47_231"}
VIS0 = TRAIN & np.isfinite(DM) & ~np.array([rid(i) in LOCK for i in range(N)])
def fires_g2(b, vis):
    t = tdate(b, vis)
    if t is None: return [False] * 24, None
    cand = [a for a in np.where(vis)[0] if a != b and FARM[a] == FARM[b] and SEGT[a] == t[1] and DAYT[a] == t[0] - 2 and DM[a] >= 1.0]
    out, pick = [], None
    for h in range(24):
        if h < H0 or not cand: out.append(False); continue
        d = sorted((dist(a, b, h), a) for a in cand); ok = (len(d) == 1 or d[1][0] >= 1.5 * max(d[0][0], 1e-9)) and d[0][0] <= KEEP[h]
        out.append(bool(ok)); pick = d[0][1] if ok else pick
    return out, pick
rows = []; n_tested = 0
for r in Y:
    b = idx[r]; base = BASE[r]
    if not np.isfinite(base).all() or b not in CERT: continue
    a = CERT[b]; vis = VIS0.copy(); vis[b] = False; vis[a] = False; n_tested += 1
    f, pick = fires_g2(b, vis)
    if not any(f): continue
    bb, aa = reg_without(r); cm = pd.Series(base).expanding().mean().values; y = Y[r]
    p = np.array([base[h] + max(0.0, (aa + bb * cm[h]) - cm[h]) if f[h] else base[h] for h in range(24)])
    rows.append(dict(rec=r, hidden_prev=rid(a), prev_high=bool(DM[a] >= 1.0), p2=int(r[4:]) >= 179, high=bool(np.nanmean(y) >= 1.0), hours=sum(f), pick=rid(pick), pick_ec=float(DM[pick]),
                     dsse=float(np.nansum((p - y) ** 2) - np.nansum((base - y) ** 2)), base_mean=float(np.nanmean(base)), y_mean=float(np.nanmean(y)), lift_mean=float(np.mean((p - base)[np.array(f)]))))
R = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(f"tested records with a cert previous day: {n_tested}")
print(R.round(3).sort_values("dsse").to_string(index=False))
false = int((~R.high).sum()); tot = R.dsse.sum(); harmed = int((R.dsse > 1e-6).sum())
v = R.dsse.values; g = np.random.default_rng(20261215); ix = g.integers(0, len(v), (20000, len(v))); P = float((v[ix].sum(1) >= 0).mean())
cl = R.groupby("pick").dsse.sum().values; ixc = g.integers(0, len(cl), (20000, len(cl))); Pc = float((cl[ixc].sum(1) >= 0).mean())
led = R[~R.rec.isin(LEDGER)].dsse.sum()
ok = false == 0 and tot < 0 and harmed <= 1
print(f"\nfired {len(R)} (pass-2 {int(R.p2.sum())}) | false {false} | dSSE {tot:+.3f} | harmed {harmed} | record-boot P {P:.4f} | ancestor-cluster ({len(cl)}) P {Pc:.4f} | ledger-excluded dSSE {led:+.3f}")
print("EH3-G2:", "ADOPT(same-data post-hoc character)" if ok else "FAIL")
