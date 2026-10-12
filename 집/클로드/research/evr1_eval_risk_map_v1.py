# -*- coding: utf-8 -*-
"""EVR1 (2026-10-13 집 클로드; 사용자: 'EC 더 해야') 서술: 평가 60 EC 위험 지도.
기록마다: 13회차 EC 하루 평균 예측, v7 앞 앵커(13회차 사용) 여부, v15(없음 선택지, PI .15) 앞 1순위·p0·정답 여부, 다음날 짝(NXL1 모델, 학습 전체로 학습, 0/9/15/21시 구간 p0 ≥ .8)과 그 0시 EC,
  같은 날짜(0시 외기 쌍둥이) 학습 기록 수·그 EC, 날짜(DMT3e) 확보 여부, CO2 다량 공급(act_co2 하루 평균 ≥ 25), 고EC 예측(≥ 1).
규정: 평가 기록은 자기 입력만, 후보·정답은 학습 데이터(뒤 날짜 포함). 출력 표 + 분류별 개수."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "nxl1_next_day_null_linker_v1.py"), encoding="utf-8").read()
exec(src[:src.index("L7 = pd.read_csv")])  # build, tdate, weights, next_anchor, TG, EC, idx, rid, TRAIN, DAYT, SEGT, X
import glob
P = pd.read_csv(glob.glob(os.path.join(env.ROOT, "제출", "13회차_2026-10-12(나)", "*.csv"))[0]); P["rec"] = P.row_id.str[:7]; PM = P.groupby("rec").sub_ec.agg(["mean", "first"])
E7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig")
v7e = {r: p for r, p, q, l, d in zip(E7.row, E7.prev_h0, E7.p_h0, E7.prev_is_labelled.astype(bool), E7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
V15 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15pi15_eval.csv")).set_index("rec")
W = weights("ALL", np.zeros(N, bool))
Xt = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); Xt["rec"] = Xt.row_id.str[:7]; CO2 = Xt.groupby("rec").act_co2.mean()
rows = []
for r in sorted(PM.index):
    if r[:3] not in ("F13", "F47"): continue
    b = idx[r]; lab = TRAIN.copy(); td = tdate(b, lab)
    twins = [rid(i) for i in np.where(TRAIN & np.all(np.abs(X[:, 0] - X[b, 0]) < 1e-6, axis=1))[0]]
    Bh, pk = next_anchor(r, lab, W); nxt = sorted({p for p in pk if p}); nh = int(np.isfinite(Bh).sum())
    rows.append(dict(rec=r, pred=round(PM.loc[r, "mean"], 3), pred0=round(PM.loc[r, "first"], 3), v7=v7e.get(r, ""), v7_A=round(EC[idx[v7e[r]], 23], 3) if r in v7e else np.nan,
                     v15=V15.prev0.get(r, "") if bool(V15.lab0.get(r, False)) else "", v15_p=round(V15.p0.get(r, np.nan), 2), v15_A=round(EC[idx[V15.prev0[r]], 23], 3) if bool(V15.lab0.get(r, False)) else np.nan,
                     nxt=",".join(nxt), nxt_hours=nh, nxt_B=round(float(np.nanmean(Bh)), 3) if nh else np.nan,
                     date=f"{td[0]:.0f}/{td[1]:.0f}" if td else "?", twins=len(twins), twin_EC=round(float(np.mean([np.nanmean(EC[idx[t]]) for t in twins])), 2) if twins else np.nan,
                     co2=round(CO2.get(r, np.nan), 1)))
T = pd.DataFrame(rows)
T["info"] = np.where(T.v7 != "", "A fwd(v7)", np.where((T.v15 != "") & (T.v15_p >= .8), "B fwd(v15 only)", np.where(T.nxt_hours > 0, "C next-day only", np.where(T.date != "?", "D dated, no link", "E nothing"))))
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 100)
print(T.groupby("info").agg(n=("rec", "size"), high_pred=("pred", lambda s: int((s >= 1).sum())), mean_pred=("pred", "mean")).round(3).to_string())
print(T.sort_values(["info", "rec"]).to_string(index=False))
T.to_csv(os.path.join(env.LOCAL, "evr1_eval_risk_map.csv"), index=False)
