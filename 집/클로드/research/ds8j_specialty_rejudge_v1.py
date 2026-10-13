# -*- coding: utf-8 -*-
"""DS8J (2026-10-13 집 클로드; 사용자: '그 배정표로 규칙 바꾸고 DS8 다시 판정').  **사후 재판정**(DS8 결과를 본 뒤 만든 규칙 — 라벨 명시).  EC.
새 EC 분야 배정 규칙(메모리 skill-not-luck, 2026-10-13):
 (1) 주 = 평가 닮음: P2LOO·EL1 모든 시드 ΔSSE < 0  AND  (P2LOO 시드 평균 블록 부트스트랩 P(악화) < .025  OR  작은 표본 길: P2LOO·EL1 어느 칸에서도 나빠진 기록 없음).
 (2) 링커 견고성 = EV (DS8 은 v15R 사슬에 의존 → 적용): EV 모든 시드 전체 행 ΔSSE ≤ 0.
 (3) 손해 감시 = DIAG10 전체 행: 시드 평균 악화 AND 블록 '후보 나음' 비율 < .025 이면 FAIL.
DS8 을 그대로 다시 실행(같은 코드·시드)하고 기록별 ΔSSE 를 모아 위 규칙으로 판정.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "ds8_ancestor_anchored_hmm_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "ds8_ancestor_anchored_hmm_v1.py")
exec(src)   # res, blocks, fl, S, FL, pw ...
print("\n================ DS8 re-judged under the EC specialty rule (post-hoc) ================")
def boot(v):
    s = pd.concat(blocks[v], axis=1).mean(axis=1).values; ix = np.random.default_rng(20261201).integers(0, len(s), (20000, len(s))); m = s[ix].mean(1)
    return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE < 0).all()); pP2, _ = boot("P2LOO")
harmed = FL[FL.val.isin(["P2LOO", "EL1"]) & (FL.dsse > 1e-6)] if len(FL) else FL
c1b = pP2 < .025 or len(harmed) == 0
print(f"(1) main P2LOO+EL1: all cells better {c1} ({int((main.dSSE < 0).sum())}/{len(main)}) | P2LOO P(worse) {pP2:.4f} | harmed records {len(harmed)} -> {'PASS' if c1 and c1b else 'FAIL'}"
      f"{'  [via small-sample route]' if c1 and pP2 >= .025 and len(harmed) == 0 else ''}")
ev = S[S.val == "EV"]; c2 = bool((ev.dSSE <= 0).all())
print(f"(2) linker robustness EV: seed dSSE {ev.dSSE.round(2).tolist()} -> {'PASS' if c2 else 'FAIL'}")
dg = S[S.val == "DIAG10"]; _, pb = boot("DIAG10"); c3 = not (dg.dSSE.mean() > 0 and pb < .025)
print(f"(3) harm guard DIAG10: mean dSSE {dg.dSSE.mean():+.2f}, share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
print("DS8 (specialty rule, post-hoc):", "PASS" if (c1 and c1b and c2 and c3) else "FAIL")
