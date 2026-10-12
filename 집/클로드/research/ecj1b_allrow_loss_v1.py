# -*- coding: utf-8 -*-
"""ECJ1b (2026-10-13 집 클로드; ECJ1 후속 서술): EC 변형(ECV) vs V7 전체 행 악화(DIAG10 +5.4%, EV +6.8%)가 어느 기록에서 나오나.
새 앵커 기록별 ΔSSE(시드 평균), 짝이 깨끗한 정답(진단 연결·달력 차 1)과 같은지/틀린지/확인 불가인지, 짝 23시 EC vs 대상 실제 0시 EC."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "ecj1_ec_two_candidates_v1.py"), encoding="utf-8").read()
src = src.replace('S = pd.DataFrame(res); pd.set_option', 'S = pd.DataFrame(res); _stop = 1; pd.set_option')
src = src.replace('for (v, sd), G in O.groupby(["validator", "seed"]):', 'ALLD = {}\nfor (v, sd), G in O.groupby(["validator", "seed"]):')
src = src.replace('            if v == "DIAG10": dsse', '            if v == "DIAG10": ALLD.setdefault((arm, bn), []).append(pd.Series((n - y) ** 2 - (b - y) ** 2).groupby(G.rec.values).sum())\n            if v == "DIAG10": dsse')
src = src.replace('            ev_ds.setdefault', '            ALLD.setdefault(("EV", arm, bn), []).append(pd.Series((n - y) ** 2 - (b - y) ** 2).groupby(G.rec.values).sum())\n            ev_ds.setdefault')
exec(src[:src.index('_stop = 1')])
TPd = {rid(int(b)): rid(int(a)) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day))
def status(r, pv, labf):
    if r not in TPd or not labf(r)[idx[TPd[r]]] or DAYr.get(r, np.nan) - DAYr.get(TPd[r], np.nan) != 1: return "unverifiable"
    return "correct" if TPd[r] == pv else "WRONG"
pd.set_option("display.width", 220)
for name, key, NEWA, labf in (("DIAG10", ("ECV", "V7"), ECVn_m, mirror_lab), ("EV", ("EV", "ECV", "V7"), ECVn_e, lab_ev)):
    D = pd.concat(ALLD[key], axis=1).mean(axis=1); D = D[D.abs() > 1e-9]
    T = pd.DataFrame(dict(dSSE=D)); T["pass"] = [2 if int(r[4:7]) >= 179 else 1 for r in T.index]
    T["prev"] = [NEWA.get(r, "") for r in T.index]; T["link"] = [status(r, NEWA[r], labf) if r in NEWA else "" for r in T.index]
    T["A23"] = [EC[idx[NEWA[r]], 23] if r in NEWA else np.nan for r in T.index]; T["y0"] = [EC[idx[r], 0] for r in T.index]
    print(f"\n== {name}: changed records {len(T)}, sum dSSE {T.dSSE.sum():+.2f} | by link:", T.groupby("link").dSSE.agg(["count", "sum"]).round(2).to_dict("index"))
    print("   by pass:", T.groupby("pass").dSSE.agg(["count", "sum"]).round(2).to_dict("index"))
    print(T.sort_values("dSSE", ascending=False).head(10).round(3).to_string())
