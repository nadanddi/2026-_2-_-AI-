# -*- coding: utf-8 -*-
"""C1 (2026-10-15 집 클로드; 사용자: 5점 모델 계획 → 'C1 하고 V3-L 진행해봐').  서술 측정(채택 판정 아님), 실행 전 커밋.  온도 + EC.
질문: 현장 조건(같은 동 어제 센서값을 앎)이면 '어제 값' 부품(온도 사슬 새벽 보정 + 23시 앵커, EC 23시 앵커)이 얼마나 좋은가 → 전제가 현장에서 성립할 때의 성능 근거.
진짜 전날 = CLEAN(truth.pkl 의 정답 연속 연결 ∩ 달력 차 1; LK1 등에서 '진단용 진짜 전날' 로 쓴 것).  한계: 정답 연속으로 고른 연결이라 끊긴(플러싱 등) 진짜 전날은 빠짐 → 낙관 쪽.
현장에선 어제가 검증 날이어도 그 측정값을 앎 → TRUE 안은 전날 정답을 폴드 숨김과 상관없이 씀(현장 조건 그대로).
[온도] TV3 와 같은 v23 틀(W40G-S → 사슬 새벽 보정 C2 → 앵커 anch), 판 DIAG10·EL1·F·G·H·I·K·EXT10·EXT12, 셀 전부:
  NONE = W40G-S 만(사슬·앵커 없음) / CUR = v23(사슬표 CT, 앵커 LV = 링커 v7 + TNA8) / TRUE = 사슬표·앵커를 CLEAN 전날로(사슬 ch_in23·ch_eve, anchor_st23 = 전날 23시 정답)
  보고: 판별 RMSE(셀 평균), 전체 / CLEAN 전날 있는 기록만 / 2차.
[EC] ec_rv1 CUR(DIAG10·EL1·P2LOO, 시드 3) 의 sg 위: NONE = sg / CUR = EA 식 앵커(링커 v7 확신 p≥.8·정답 전날·|dT|≤1.5, 6.574) / TRUE = CLEAN 전날
  EA 식: pred_h = p_h + (A − pm_h), A = 전날 23시 EC 정답, pm_h = 그날 0..h 예측 누적 평균.  보고: RMSE 전체 / 앵커 받은 기록만 / 2차 / 고EC.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u c1_true_yesterday_field_condition_v1.py
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "tv3_v3_gru_on_v23_v1.py")
exec(src[:src.index("JUDGE = [")])
__file__ = os.path.join(HERE, "c1_true_yesterday_field_condition_v1.py")
from feat4 import EC as ECA  # noqa: E402
TLc = pd.read_pickle(S3 + r"\truth.pkl"); CALc = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); D11c = dict(zip(CALc.rec, CALc.day))
TRUE = {rid(int(b)): rid(int(a)) for a, b in zip(TLc.a1, TLc.b) if D11c.get(rid(int(b)), np.nan) - D11c.get(rid(int(a)), np.nan) == 1}
TRUE = {r: p for r, p in TRUE.items() if TRAIN[idx[p]] and np.isfinite(ST[idx[p], 23])}
print("CLEAN true-yesterday links with labelled prev:", len(TRUE), flush=True)
CTT = CT.copy(); CTT["ch_in23"] = np.nan; CTT["ch_eve"] = np.nan
for i, r in CTT.iterrows():
    k = f"{r.farm}_{int(r.day):03d}"
    if k in TRUE and TRUE[k] in IN23.index: CTT.at[i, "ch_in23"] = IN23[TRUE[k]]; CTT.at[i, "ch_eve"] = EVE[TRUE[k]]
LT = pd.DataFrame(dict(rec=list(TRUE), prev=list(TRUE.values()), p=1.0, prev_is_labelled=True, anchor_st23=[float(ST[idx[p], 23]) for p in TRUE.values()], dT_air=0.0, steps_to_anchor=1))
D1T = T.layout_frame("DIAG10", CTT)
print("[TEMP]", flush=True); rows = []
for L in ["DIAG10", "EL1", "F", "G", "H", "I", "K", "EXT10", "EXT12"]:
    d, pre, bs, cx = T.LAY[L]; M = T.layout_frame(L, CT); MT = T.layout_frame(L, CTT); y = M.sub_temp.values
    has = M.row_id.str[:7].isin(TRUE.keys()).values; p2 = M.day.values >= 179
    arms = {"NONE": [], "CUR": [], "TRUE": []}
    for bi, b in enumerate(bs):
        for fam in "AB":
            a = B.w40(M, b, cx, fam); arms["NONE"].append(a); arms["CUR"].append(anch(M, C2.corrected(M, a, D1, bi, fam, 7), LV))
            aT = B.w40(MT, b, cx, fam); arms["TRUE"].append(anch(MT, C2.corrected(MT, aT, D1T, bi, fam, 7), LT))
    for k, P in arms.items():
        rows.append(dict(layout=L, arm=k, all=np.mean([rm(p - y) for p in P]), true_prev_recs=np.mean([rm((p - y)[has]) for p in P]) if has.any() else np.nan,
                         no_true_prev=np.mean([rm((p - y)[~has]) for p in P]) if (~has).any() else np.nan, p2=np.mean([rm((p - y)[p2]) for p in P]) if p2.any() else np.nan, n_has=int(has.sum() // 24)))
    print("done", L, flush=True)
R = pd.DataFrame(rows); pd.set_option("display.width", 220); print(R.pivot(index="layout", columns="arm", values=["all", "true_prev_recs", "p2"]).round(4).to_string())
print("n records with true prev per layout:", R.groupby("layout").n_has.first().to_dict())
# ---------- EC ----------
print("[EC]", flush=True)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
CURE = {r: p for r, p, q, l, dT in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if isinstance(p, str) and q >= .8 and l and abs(dT) <= 1.5 and r in v7set}
def ea(g, prev):
    p = g.sg.values
    if prev is None or not np.isfinite(ECA[idx[prev], 23]): return p
    pm = np.cumsum(p) / np.arange(1, len(p) + 1); return p + (ECA[idx[prev], 23] - pm)
rows = []
for (v, sd), G in O[O.validator.isin(["DIAG10", "EL1", "P2LOO"])].groupby(["validator", "seed"]):
    parts = {k: [] for k in ("y", "NONE", "CUR", "TRUE", "has", "hasc", "p2", "hi")}
    for r, g in G.groupby("rec"):
        g = g.sort_values("hour"); y = g.sub_ec.values
        parts["y"].append(y); parts["NONE"].append(g.sg.values); parts["CUR"].append(ea(g, CURE.get(r))); parts["TRUE"].append(ea(g, TRUE.get(r)))
        parts["has"].append(np.full(len(y), r in TRUE)); parts["hasc"].append(np.full(len(y), r in CURE)); parts["p2"].append(g.day.values >= 179); parts["hi"].append(np.full(len(y), np.nanmean(y) >= 1.0))
    P = {k: np.concatenate(v_) for k, v_ in parts.items()}; y = P["y"]
    for k in ("NONE", "CUR", "TRUE"):
        e = P[k] - y; rows.append(dict(val=v, seed=sd, arm=k, all=rm(e), true_prev_recs=rm(e[P["has"]]) if P["has"].any() else np.nan, cur_anchor_recs=rm(e[P["hasc"]]) if P["hasc"].any() else np.nan,
                                       p2=rm(e[P["p2"]]) if P["p2"].any() else np.nan, high=rm(e[P["hi"]]) if P["hi"].any() else np.nan, n_true=int(P["has"].sum() // 24), n_cur=int(P["hasc"].sum() // 24)))
E = pd.DataFrame(rows); print(E.groupby(["val", "arm"])[["all", "true_prev_recs", "cur_anchor_recs", "p2", "high", "n_true", "n_cur"]].mean().round(4).to_string())
