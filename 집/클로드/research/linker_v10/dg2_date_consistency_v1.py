# DG2 (2026-10-12 집 클로드; 사용자: '날짜 불일치는 최대한 비슷한 걸로') 서술: 앵커 짝 중 '대상 날짜 − 1 ≠ 짝 날짜'인 것 개수(평가 = AP11, 검증 = TD2 앵커 집합).
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import pandas as pd, numpy as np
C = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); D = dict(zip(C.rec, C.day)); SPL = dict(zip(C.rec, C.split))
by = C.dropna(subset=["day"]).groupby("day")
labday = {d: list(g.rec[g.split == "train"]) for d, g in by}
E = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "eval60_v11.csv"))
E = E[E.tier != ""].copy()
E["d_t"] = E.rec.map(D); E["d_p"] = E.v11.map(D); E["gap"] = E.d_t - E.d_p
E["n_lab_dm1"] = [len(labday.get(d - 1, [])) if d == d else np.nan for d in E.d_t]
E["status"] = np.select([E.d_t.isna() | E.d_p.isna(), E.gap == 1], ["date unknown", "OK"], "MISMATCH")
print("EVAL anchors (v7 kept + new):"); print(E.groupby(["tier", "status"]).size().to_string())
print(E[E.status != "OK"][["rec", "tier", "v11", "p", "d_t", "d_p", "gap", "n_lab_dm1"]].to_string(index=False))
P = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v11_mirror_picks.csv"))
P["d_t"] = P.rec.map(D); P["d_p"] = P.prev.map(D); P["gap"] = P.d_t - P.d_p
for nm, S in (("v11 picks p>=.6 (validation)", P[P.p >= .6]), ("  of which gate-released (|dT|>1.5)", P[(P.p >= .6) & (P.dT.abs() > 1.5)])):
    st = np.select([S.d_t.isna() | S.d_p.isna(), S.gap == 1], ["date unknown", "OK"], "MISMATCH")
    print(f"{nm}: {pd.Series(st).value_counts().to_dict()}")
