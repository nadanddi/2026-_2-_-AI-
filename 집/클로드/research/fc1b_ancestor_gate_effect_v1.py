# -*- coding: utf-8 -*-
"""FC1B (2026-10-13 집 클로드; FC1 후속, 서술).  DS1 기록별 ΔSSE(DIAG10, ds1y_zone_rows) 를 '가장 가까운 보이는 정답 조상' 상태로 나눔 — 모든 360기록.
+ D1 을 같은 농장으로 좁힌 판."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "fc1_false_chain_breakers_v1.py"), encoding="utf-8").read(); exec(src[:src.index("rows = []")])
Z = pd.read_csv(os.path.join(HERE, "ds1y_zone_rows_v1.csv"))
def anc(r):
    hs = hidden[fold[r]]; cur, k = r, 0
    while k < 6 and cur in CPREV:
        cur = CPREV[cur]; k += 1
        if TRAIN[idx[cur]] and cur not in hs and np.isfinite(DM[idx[cur]]): return ("HIGH" if DM[idx[cur]] >= 1.0 else "normal"), k
    return "none", np.nan
Z[["anc", "dist"]] = [anc(r) for r in Z.rec]
Z["predH"] = Z.pred >= 1.0; Z["type"] = np.select([Z.trueH & Z.predH, Z.trueH & ~Z.predH, ~Z.trueH & Z.predH], ["H_seen", "H_missed", "N_falseH"], "N_ok")
pd.set_option("display.width", 200)
print("DS1 dSSE by nearest labelled ancestor x type (n / sum):"); print(Z.pivot_table(index="type", columns="anc", values="dsse", aggfunc=["size", "sum"]).round(2).to_string())
print("\nDS1 total", round(Z.dsse.sum(), 2), "| applied only where ancestor HIGH:", round(Z[Z.anc == "HIGH"].dsse.sum(), 2), "(n", int((Z.anc == "HIGH").sum()), ")",
      "| pass-2 only:", round(Z[(Z.anc == "HIGH") & Z.p2].dsse.sum(), 2))
print("ancestor-HIGH losses:", Z[(Z.anc == "HIGH") & (Z.dsse > .5)][["rec", "type", "y", "pred", "dist", "dsse"]].round(3).to_dict("records"))
def d1farm(r):
    hs = hidden[fold[r]]
    if r not in DAY: return "no date"
    s = T[(T.key == DAY[r]) & (T.rec != r) & ~T.rec.isin(hs) & (T.farm == r[:3])]
    return "other HIGH" if s.hi.any() else ("normal only" if len(s) else "none visible")
P = Z[Z.predH].copy(); P["D1farm"] = [d1farm(r) for r in P.rec]
print("\nD1 same farm only (predicted-high):"); print(pd.crosstab(P.D1farm, P.trueH).to_string())
Z.to_csv(os.path.join(HERE, "fc1b_rows_v1.csv"), index=False)
