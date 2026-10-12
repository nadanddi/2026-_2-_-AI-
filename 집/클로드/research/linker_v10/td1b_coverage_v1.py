# TD1b (2026-10-12 집 클로드; 사용자: 'TD1 포괄량은') 서술: TD1 앵커 집합(BASE v7 54 / CERT 107)의 거울 CV 정확도·포괄률 — 정답 기준 3가지(전체 진단 연결, 확실 연결, 달력 차 1 깨끗한 연결), 1·2차.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day))
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TP = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}; CERT = {rid(int(b)): bool(c) for b, c in zip(TLa.b, TLa.cert)}
idx = {rid(i): i for i in range(N)}
vis = {}
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0]; m = s + sh
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
        for b in blk:
            if TRAIN[b]: vis[rid(b)] = TRAIN & ~touch
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
v7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
P11 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v11_mirror_picks.csv")); P11 = P11[(P11.p >= .6) & (P11.dT.abs() <= 1.5)]
cert_set = dict(v7); cert_set.update({r: p for r, p in zip(P11.rec, P11.prev) if r not in v7})
def linkable(r, kind):
    if r not in TP or r not in vis or not vis[r][idx[TP[r]]]: return False
    if kind == "cert": return CERT[r]
    if kind == "clean": return DAYr.get(r, np.nan) - DAYr.get(TP[r], np.nan) == 1
    return True
for kind in ("all", "cert", "clean"):
    for part, sel in (("all", lambda r: True), ("pass-2", lambda r: int(r[4:7]) >= 179)):
        can = [r for r in vis if linkable(r, kind) and sel(r)]
        line = f"{kind:5s} truth | {part:6s} linkable {len(can):3d} |"
        for nm, S in (("BASE v7", v7), ("CERT v7+v11", cert_set)):
            A = {r: p for r, p in S.items() if sel(r)}
            judged = [r for r in A if r in TP and (kind == "all" or linkable(r, kind) or not (r in TP and vis.get(r) is not None and vis[r][idx[TP[r]]]))]
            ok = sum(1 for r in can if A.get(r) == TP[r])
            prec_pool = [r for r in A if (kind == "all") or (kind == "cert" and CERT.get(r, False)) or (kind == "clean" and DAYr.get(r, np.nan) - DAYr.get(TP.get(r, ""), np.nan) == 1)]
            okp = sum(1 for r in prec_pool if A[r] == TP.get(r))
            line += f" {nm}: anchors {len(A)}, precision {okp / max(1, len(prec_pool)):.3f}, recall {ok / max(1, len(can)):.2f} |"
        print(line)
