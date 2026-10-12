# LR2b (2026-10-12 집 클로드; 사용자: '포괄률 46% 면 전날 있는 데이터 절반을 못 찾는 거냐') 서술: 이을 수 있는 기록이 메타(≥.7)에서 앵커를 못 받은 이유 분해.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
idx = {rid(i): i for i in range(N)}
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TP = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}; CERT = {rid(int(b)): bool(c) for b, c in zip(TLa.b, TLa.cert)}
M = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "lr2_candidates.csv"))
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAY = dict(zip(CAL.rec, CAL.day))
vis = {}
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0]; m = s + sh
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
        for b in blk:
            if TRAIN[b]: vis[rid(b)] = TRAIN & ~touch
can = [r for r in vis if r in TP and vis[r][idx[TP[r]]]]
rows = []
for r in can:
    t = TP[r]; G = M[(M.rec == r) & M.lab]
    gap = (DAY.get(r, np.nan) - DAY.get(t, np.nan))
    if len(G) == 0: cat = "no labelled candidate kept"
    elif t not in set(G.cand): cat = "truth not in candidate pool"
    else:
        top = G.loc[G.pm.idxmax()]
        if top.cand != t: cat = "in pool, meta chose another"
        elif top.pm < .7: cat = "chosen but meta prob < .7"
        elif top.dT > 1.5: cat = "chosen, dT gate"
        else: cat = "ANCHORED correctly"
    rows.append(dict(rec=r, truth=t, cert=CERT[r], p2=int(r[4:7]) >= 179, cat=cat, cal_gap=gap))
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(pd.crosstab(T.cat, T.cert, margins=True).to_string())
print("\ncalendar gap (record day - truth day) for 'truth not in pool': ", T[T.cat == "truth not in candidate pool"].cal_gap.value_counts(dropna=False).head(10).to_dict())
print("calendar gap for ANCHORED:", T[T.cat == "ANCHORED correctly"].cal_gap.value_counts(dropna=False).head(5).to_dict())
