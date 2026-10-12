# ECL1 (2026-10-13 집 클로드; 사용자: 'EC 전용으로 쓴다면 전날 연결 수준은') 서술: 거울 CV, 깨끗한 정답(진단 연결 중 달력 차 1) 기준 v7·v12 의 문턱별 정확도·포괄률(전체/2차).
# 앵커 = 확신 ≥ 문턱 & 전날 보이는 정답 기록 (EC 는 공기 점프와 무관하므로 실내 차 차단 있음/없음 둘 다).
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day))
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TP = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}
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
clean = lambda r: r in TP and r in vis and vis[r][idx[TP[r]]] and DAYr.get(r, np.nan) - DAYr.get(TP[r], np.nan) == 1
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig").rename(columns={"dT_air": "dT"})
L7["labelled"] = L7.prev_is_labelled.astype(bool); L7["date_status"] = "n/a"
P12 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v12_picks_mirror.csv"))
rows = []
for nm, P in (("v7", L7), ("v12 (rule-clean)", P12)):
    for th in (.6, .8, .9, .95):
        for gate in (True, False):
            for part, sel in (("all", lambda r: True), ("pass-2", lambda r: int(r[4:7]) >= 179)):
                can = [r for r in vis if clean(r) and sel(r)]
                A = P[(P.p >= th) & P.labelled & P.rec.map(sel) & ((P.dT.abs() <= 1.5) if gate else True)]
                if "date_status" in A and nm != "v7": A = A[A.date_status != "mismatch"]
                pool = A[A.rec.map(lambda r: clean(r))]; okp = int((pool.prev == pool.rec.map(TP)).sum())
                rows.append(dict(linker=nm, thr=th, gate=gate, part=part, anchors=len(A), precision=okp / max(1, len(pool)), recall=okp / max(1, len(can)), linkable=len(can)))
T = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(T.pivot_table(index=["linker", "thr", "gate"], columns="part", values=["anchors", "precision", "recall"]).round(3).to_string())
