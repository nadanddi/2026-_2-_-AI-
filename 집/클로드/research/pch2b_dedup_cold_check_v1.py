# -*- coding: utf-8 -*-
"""PCH2b (2026-10-16 집 클로드; PCH2 최종 비평 후속).  서술(판정 바꾸지 않음), 실행 전 커밋.  PCH2 를 그대로 다시 돌린 뒤:
 (a) 배치별 블록 P, (b) 같은 날(농장 x 5일 블록)을 배치 간 평균해 한 번만 센 P, (c) 사슬 추운 기록(하루 최저 <= 8도)별 DAWN 이득(배치별), 해 입은 기록 수.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u pch2b_dedup_cold_check_v1.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "pch2_dawn_chain_fresh_v1.py"), encoding="utf-8").read()
src = src.replace('__file__ = os.path.join(HERE, "pch2_dawn_chain_fresh_v1.py")', '__file__ = os.path.join(HERE, "pch2b_dedup_cold_check_v1.py")')
src = src.replace("    RES[L] = dict(P=P, y=y, t=t, ch=chm, h=h,", "    RES[L] = dict(key=key, dmin=dmin, P=P, y=y, t=t, ch=chm, h=h,")
exec(compile(src, "pch2", "exec"))
rng = np.random.default_rng(20270153); per = {}
for L in RES:
    R_ = RES[L]; d = np.mean([(p - R_["y"]) ** 2 - (b - R_["y"]) ** 2 for p, b in zip(R_["P"]["DAWN"], R_["P"]["B"])], 0)
    s = pd.Series(d).groupby(R_["bk"]).mean(); per[L] = s.rename(index=lambda k: k.split("_", 1)[1])
    print(f"(a) {L}: block P(worse) {float((s.values[rng.integers(0, len(s), (20000, len(s)))].mean(1) >= 0).mean()):.4f} | blocks {len(s)}")
U = pd.concat(per, axis=1).mean(axis=1).values
print(f"(b) dedup (same farm x 5-day block averaged over layouts): blocks {len(U)} | P(worse) {float((U[rng.integers(0, len(U), (20000, len(U)))].mean(1) >= 0).mean()):.4f}")
rows = []
for L in RES:
    R_ = RES[L]; d = np.mean([(p - R_["y"]) ** 2 - (b - R_["y"]) ** 2 for p, b in zip(R_["P"]["DAWN"], R_["P"]["B"])], 0)
    S = pd.DataFrame(dict(rec=R_["key"], d=d, ch=R_["ch"], dmin=R_["dmin"])); S = S[S.ch & (S.dmin <= 8)]
    for r, g in S.groupby("rec"): rows.append(dict(layout=L, rec=r, dmin=round(g.dmin.iloc[0], 1), dSSE=round(g.d.sum(), 3)))
C = pd.DataFrame(rows); print("(c) chained cold records:"); print(C.to_string(index=False))
print(f"    harmed {int((C.dSSE > 1e-6).sum())} / helped {int((C.dSSE < -1e-6).sum())} / zero {int((C.dSSE.abs() <= 1e-6).sum())}")
