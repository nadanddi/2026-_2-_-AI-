# -*- coding: utf-8 -*-
"""LK1 (2026-10-13 집 클로드; 사용자: '링커 정확도를 높일 수 있는 방법을 찾아봐').  서술·탐색(판정 아님).
EV 폴드별 v15R 짝(v15r15_ev, 폴드마다 링커를 다시 돌린 실제 오류 포함)의 1순위 짝 prev0 이 진짜 전날(진단용 truth a1 & DM11 차 1)과 같은지,
그리고 틀린 짝을 가려낼 단서(모두 규정 안에서 계산 가능한 것):
  p0, 1·2등 차(p0 − p1), p_null, date_src, 짝 정답 여부(lab0), HP2 일치(그 폴드 정답 보임 기준), 학습 달력(DMT3e) 날짜 일치(짝 날짜 = 대상 날짜 − 1),
  공기 온도 연속 |짝 23시 실내온 − 대상 0시 실내온|, 외기 연속, ID 교차, 같은 ID 앞 번호 여부,
  정답 연속(짝이 보이는 정답일 때): |짝 EC 23시 − 대상 EC 예측 0시|, |짝 배지온 23시 − 대상 실내온 0시 + .36|.
대상: 진짜 전날이 정의된 EV 기록.  하위 집합: 예측 고EC(SG2 하루 평균 ≥ 1.0) 기록 및 DS8 가짜 조상 사슬.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CR1SRC = open(os.path.join(HERE, "cr1_chain_gru_temp_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "cr1_chain_gru_temp_v1.py")
exec(CR1SRC[:CR1SRC.index("V = [\"in_temp\", \"in_hum\"")])   # hp2, ST, idx, TRAIN, FARM, NUM, N, rid
__file__ = os.path.join(HERE, "lk1_linker_hop_audit_v1.py")
from sklearn.metrics import roc_auc_score  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X as OUTX, S3  # noqa: E402
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11 = dict(zip(CAL.rec, CAL.day))
CLEAN = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b) if D11.get(rid(int(b)), np.nan) - D11.get(rid(int(a)), np.nan) == 1}
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAYT = dict(zip(C.rec, C.day)); SEGT = dict(zip(C.rec, C.seg))
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); _X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]
_X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
TIN = _X.pivot(index="rec", columns="h", values="in_temp").reindex(columns=range(24)); TOUT = _X.pivot(index="rec", columns="h", values="out_temp").reindex(columns=range(24))
def tdate(r, lab):
    i = idx[r]; vi = np.where(lab)[0]; vi = vi[vi != i]; m = vi[np.all(np.abs(OUTX[vi, 0] - OUTX[i, 0]) < 1e-6, axis=1)]
    ds = [(DAYT.get(rid(j)), SEGT.get(rid(j))) for j in m if rid(j) in DAYT]
    return pd.Series(ds).mode().iloc[0] if ds else None
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv"))
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidEV = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
R = pd.read_csv(os.path.join(HERE, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
SG = R.groupby(["rec", "hour"]).SG2.mean().unstack()
rows = []
for _, q in PE.iterrows():
    r, pv = q.rec, q.prev0
    if r not in CLEAN or not isinstance(pv, str): continue
    k = int(q.fold); hs = hidEV[k]; lab = TRAIN & ~np.array([rid(i) in hs for i in range(N)])
    td = tdate(r, lab); pd_ = (DAYT.get(pv), SEGT.get(pv)) if TRAIN[idx[pv]] and pv not in hs else None
    vis = bool(TRAIN[idx[pv]] and pv not in hs and np.isfinite(DM[idx[pv]]))
    try: h2 = hp2(idx[r], lab)
    except Exception: h2 = None
    rows.append(dict(rec=r, prev=pv, ok=int(pv == CLEAN[r]), p0=q.p0, margin=q.p0 - q.p1, p_null=q.p_null, date_src=q.date_src, vis=vis,
                     hp2_agree=int(h2 == pv), hp2_none=int(h2 is None),
                     date_ok=(np.nan if (td is None or pd_ is None or pd_[0] is None) else float(pd_[1] == td[1] and pd_[0] == td[0] - 1)),
                     dTin=abs(TIN.at[pv, 23] - TIN.at[r, 0]), dTout=abs(TOUT.at[pv, 23] - TOUT.at[r, 0]), cross=int(pv[:3] != r[:3]),
                     same_id_earlier=int(pv[:3] == r[:3] and NUM[idx[pv]] < NUM[idx[r]]),
                     dEC=(abs(EC[idx[pv], 23] - SG.at[r, 0]) if vis else np.nan), dST=(abs(ST[idx[pv], 23] - .36 - TIN.at[r, 0]) if vis else np.nan),
                     predH=SG.loc[r].mean() >= 1.0, trueH=DM[idx[r]] >= 1.0 if np.isfinite(DM[idx[r]]) else False, p2=NUM[idx[r]] >= 179))
D = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(f"EV hops with defined true prev: {len(D)} | accuracy {D.ok.mean():.3f} | predicted-high {int(D.predH.sum())} acc {D[D.predH].ok.mean():.3f} | truly high {int(D.trueH.sum())} acc {D[D.trueH].ok.mean():.3f}")
print("\naccuracy by p0 bin:"); print(D.groupby(pd.cut(D.p0, [0, .3, .5, .6, .7, .8, .9, 1.01])).ok.agg(["size", "mean"]).round(3).to_string())
print("\nsingle-signal AUC for 'hop correct' (higher = signal value larger when correct; for distances expect < .5):")
for c in ["p0", "margin", "p_null", "hp2_agree", "date_ok", "dTin", "dTout", "cross", "same_id_earlier", "dEC", "dST"]:
    m = D[c].notna()
    if D.ok[m].nunique() == 2: print(f"  {c:16s} n {int(m.sum()):3d}  AUC {roc_auc_score(D.ok[m], D[c][m]):.3f}")
print("\nconditional accuracy (all hops / predicted-high hops):")
for nm, m in [("p0>=.8", D.p0 >= .8), ("p0<.8", D.p0 < .8), ("hp2 agree", D.hp2_agree == 1), ("p0<.8 & hp2 agree", (D.p0 < .8) & (D.hp2_agree == 1)),
              ("p0<.8 & hp2 disagree", (D.p0 < .8) & (D.hp2_agree == 0)), ("date_ok==1", D.date_ok == 1), ("date_ok==0", D.date_ok == 0),
              ("p0<.8 & date_ok==1", (D.p0 < .8) & (D.date_ok == 1)), ("cross-ID", D.cross == 1), ("same-ID earlier", D.same_id_earlier == 1),
              ("dTin<=1.0", D.dTin <= 1.0), ("dTin>1.0", D.dTin > 1.0), ("p0<.8 & dTin<=1", (D.p0 < .8) & (D.dTin <= 1.0))]:
    a, b = D[m], D[m & D.predH]
    print(f"  {nm:22s} n {len(a):3d} acc {a.ok.mean():.3f}  |  predH n {len(b):2d} acc {b.ok.mean() if len(b) else np.nan:.3f}")
print("\nDS8 false-ancestor chains (first hop):"); print(D[D.rec.isin(["F47_130", "F13_160", "F47_128", "F47_210", "F47_141", "F47_143", "F47_134"])][["rec", "prev", "ok", "p0", "margin", "hp2_agree", "date_ok", "dTin", "cross", "dEC"]].round(3).to_string(index=False))
D.to_csv(os.path.join(HERE, "lk1_hops_v1.csv"), index=False)
