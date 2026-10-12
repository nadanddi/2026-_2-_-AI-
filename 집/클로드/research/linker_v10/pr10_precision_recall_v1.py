# PR10 (2026-10-12 집 클로드; 사용자: 'v10 정확도·포괄률') 서술: 거울 세계 일관 기준(앵커 전날이 그 덩어리에서 '보이는 정답 기록'일 때만 앵커로 인정).
# 이을 수 있는 기록 = 진짜 전날(진단용 연결)이 보이는 정답 기록. v10 은 같은 ID 뒤 번호 학습 기록도 후보이므로 '이을 수 있는' 정의는 v7 과 같음(보이는 정답 기록 전부).
# 앵커 = p ≥ .8 & 정답 기록 & |dT_air| ≤ 1.5.  1·2차 별도.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
idx = {rid(i): i for i in range(N)}
TP = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
vis = {}
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0]; m = s + sh
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
        for b in blk:
            if TRAIN[b]: vis[rid(b)] = TRAIN & ~touch
can = {r for r, v in vis.items() if r in TP and v[idx[TP[r]]]}
for nm, path in (("v7", os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")), ("v10", os.path.join(env.LOCAL, "linker_v10", "cv_all400_choice_v10m.csv"))):
    D = pd.read_csv(path, encoding="utf-8-sig")
    for part, sel in (("all", lambda r: True), ("pass-1", lambda r: int(r[4:7]) < 179), ("pass-2", lambda r: int(r[4:7]) >= 179)):
        A = D[(D.p >= .8) & D.prev_is_labelled.astype(bool) & (D.dT_air.abs() <= 1.5) & D.rec.map(sel)]
        ok = int(sum(TP.get(r) == p for r, p in zip(A.rec, A.prev))); cn = sum(1 for r in can if sel(r))
        top1 = int(sum(TP.get(r) == p for r, p in zip(D.rec, D.prev) if r in can and sel(r)))
        print(f"{nm:4s} {part:6s}: anchors {len(A):3d} | precision {ok/max(1,len(A)):.3f} | recall {ok/cn:.2f} (correct {ok}/{cn} linkable) | top-1 correct among linkable {top1/cn:.2f}")
