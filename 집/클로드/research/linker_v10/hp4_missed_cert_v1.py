# HP4 (2026-10-12 집 클로드; 사용자: '포괄률 72% 의 나머지는 왜') 서술: 확실 연결 이을 수 있는 65 중 HP2+META(≥.5) 앵커를 못 받은 기록의 이유.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np
T = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "hp3_rows.csv"))
L2 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "lr2_candidates.csv")); B2 = L2[L2.lab].loc[lambda d: d.groupby("rec").pm.idxmax()].set_index("rec")
T["meta"] = T.rec.map(B2.cand); T["metap"] = T.rec.map(B2.pm).fillna(0)
C = T[T.cert & T.linkable].copy()
def why(r):
    if r.nocand: return "no candidate on chosen date"
    if r.ok and r.pick == r.meta and r.metap >= .5 and not r.gate: return "ANCHORED"
    if not r.ok and r.tier == "B": return "date not fixed (no exact outdoor match) -> wrong pick"
    if not r.ok: return "date fixed but HP2 pick wrong"
    if r.gate: return "correct pick but dT gate"
    if r.pick != r.meta: return "correct pick but META chose other"
    return "correct pick, META agrees but prob < .5"
C["why"] = C.apply(why, axis=1)
print(f"cert linkable rows: {len(C)} (of 65; missing = no decision possible)"); print(C.why.value_counts().to_string())
print("\ntier counts among cert linkable:", C.tier.value_counts().to_dict())
print(C[C.why != "ANCHORED"][["rec", "tier", "pick", "meta", "metap", "ok", "gate"]].round(2).to_string(index=False))
