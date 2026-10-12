# AG10 (2026-10-12 집 클로드; 사용자: 'AG9 층별 정확도·포괄률은?') 서술.
# AG8 (2026-10-12 집 클로드; 사용자: '97% 면 전날 자료 있는 건 거의 다 이었다는 거냐') 서술: 거울 CV 에서 '진짜 전날이 보이는 정답 기록'인 기록 수(이을 수 있는 몫)와
# 방안별 정확도(precision)·포괄률(recall = 맞게 이은 수 / 이을 수 있는 수).
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
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
print(f"records {len(vis)} | true prev is a visible labelled record (linkable): {len(can)} | not linkable: {len(vis) - len(can)}")
H = r"C:\work\farmai\집\클로드\research"
TP = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
V = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").set_index("rec")
V["lab"] = V.prev_is_labelled.astype(bool)
LM = pd.read_csv(os.path.join(env.LOCAL, "linker_v9", "lm1_choices.csv"))
idx0 = {rid(i): i for i in range(N)}
dT = lambda r, p: float(IN_[idx0[r], 0, 0] - IN_[idx0[p], 23, 0])
v7a = (V.p >= .8) & V.lab & (V.dT_air.abs() <= 1.5)
hv, sv = {}, {}
for r, G in LM.groupby("rec"):
    cnt = G.prev.value_counts(); top = cnt.index[0]
    if cnt.iloc[0] >= 3 and TRAIN[idx0[top]] and abs(dT(r, top)) <= 1.5: hv[r] = top
    s = G.groupby("prev").p.sum() / 4; t2 = s.idxmax()
    if s.max() >= .70 and TRAIN[idx0[t2]] and abs(dT(r, t2)) <= 1.5: sv[r] = t2
V = V.reindex(sorted(set(V.index) | set(hv) | set(sv)))
def arm(prevmap, keep_v7):
    out = {}
    if keep_v7: out.update({r: V.at[r, "prev"] for r in V.index[v7a.reindex(V.index).fillna(False)]})
    for r, p in prevmap.items(): out.setdefault(r, p)
    return out
ARMSP = {"V7": arm({}, True), "HV3": arm(hv, False), "ADD3": arm(hv, True), "SV": arm(sv, False)}
TA = arm(sv, True); TB = {r: p for r, p in hv.items() if r not in TA}
ARMSP["TV"] = {**TA, **TB}; WGT = {**{r: 1.0 for r in TA}, **{r: .5 for r in TB}}
print(f"tier A (v7 U SV) {len(TA)} | tier B (HV3 only, half) {len(TB)}")
ARMS = {k: pd.Series({r: True for r in d}) for k, d in ARMSP.items()}
PREV = lambda k, r: ARMSP[k][r]

for k, d in (("A precise (v7 U SV)", TA), ("B coverage (HV3 only)", TB), ("TV total (A+B)", ARMSP["TV"]), ("v7 (now)", ARMSP["V7"])):
    ok = sum(1 for r, p in d.items() if TP.get(r) == p)
    print(f"  {k:24s} anchors {len(d):3d} | precision {ok/len(d):.3f} | correct {ok:3d} | recall of linkable({len(can)}) {ok/len(can):.2f}")
print("\nmirror-consistent check (anchor counted correct only if its prev is a VISIBLE labelled record in that mirror chunk):")
for k, d in (("A precise (v7 U SV)", TA), ("B coverage (HV3 only)", TB), ("TV total (A+B)", ARMSP["TV"]), ("v7 (now)", ARMSP["V7"])):
    visok = {r: p for r, p in d.items() if r in vis and vis[r][idx[p]]}
    ok = sum(1 for r, p in visok.items() if TP.get(r) == p)
    print(f"  {k:24s} anchors {len(d):3d} (prev visible-labelled in mirror: {len(visok):3d}) | correct among those {ok:3d} | precision {ok/max(1,len(visok)):.3f} | recall {ok/len(can):.2f}")
