# -*- coding: utf-8 -*-
"""CNT2 (2026-10-13 집 클로드; 사용자: 'CNT1 분류별 개수 + 각 오차 크기 + 오차 큰 것 개수') 서술.
분류(CNT1 과 같음, 거울 기하 G0): 전날 있음(깨끗한 정답 보임) → 잘 찾음 / 못 찾음(앵커 없음) / 틀린 짝 ; 전날 없음(숨김·불확실) → 비움 / 그래도 붙임.
구성 2개: v7(13회차: v7 p≥.8·|dT|≤1.5) , N8(v7 ∪ v15 PI.15 p0≥.8 정답 짝).
EC 오차: rv1 DIAG10 CUR sg(시드 3 평균) 에 EA2 식으로 앵커 적용(짝이 DIAG10 폴드 숨김이면 앵커 버림). 기록별 하루 RMSE.
보고: 분류별 기록 수, 기록 RMSE 평균·중앙값, 앵커 없을 때 RMSE 평균(비교), SSE 몫, 큰 오차 기록 수(RMSE > .2, > .5) — 전체·2차."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
from feat4 import EC
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day))
TLa = pd.read_pickle(os.path.join(S3, "truth.pkl")); TP = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}
idx = {rid(i): i for i in range(N)}
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
def status(r):
    if r not in TP: return "none"
    if DAYr.get(r, np.nan) - DAYr.get(TP[r], np.nan) != 1: return "uncertain"
    return "has" if mirror_lab(r)[idx[TP[r]]] else "hidden"
OUT = os.path.join(env.LOCAL, "linker_v10")
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
D = pd.read_csv(os.path.join(OUT, "v15pi15_mirror.csv")); V15 = {r: p for r, p, q, l in zip(D.rec, D.prev0, D.p0, D.lab0) if q >= .8 and l}
CFG = {"v7 (13th)": V7, "N8 (v7 + v15 .8)": {**V15, **V7}}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
G = O.groupby(["rec", "hour", "fold", "day"], as_index=False).agg(y=("sub_ec", "first"), sg=("sg", "mean")).sort_values(["rec", "hour"]).reset_index(drop=True)
G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()); hidden = G.groupby("fold").rec.apply(set).to_dict()
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN]); recs = sorted(G.rec.unique())
pd.set_option("display.width", 250)
for nm, AA in CFG.items():
    rows = []
    for r, g in G.groupby("rec"):
        st = status(r); hs = hidden[int(g.fold.iloc[0])]
        a = AA.get(r); a = a if (a is not None and a not in hs) else None
        pr = np.clip(g.sg.values + (EC[idx[a], 23] - g.cm.values), lo_, hi_) if a else g.sg.values
        if st == "has": cat = "1 HAS: found" if a == TP[r] else ("3 HAS: wrong anchor" if a else "2 HAS: missed")
        else: cat = "5 NO prev: anchored anyway" if a else "4 NO prev: abstained"
        rows.append(dict(rec=r, cat=cat, pass2=int(r[4:7]) >= 179, sse=float(np.sum((pr - g.y.values) ** 2)), rmse=float(np.sqrt(np.mean((pr - g.y.values) ** 2))),
                         rmse_none=float(np.sqrt(np.mean((g.sg.values - g.y.values) ** 2))), ymean=float(g.y.mean())))
    T = pd.DataFrame(rows)
    for part, S in (("ALL", T), ("PASS-2", T[T.pass2])):
        A = S.groupby("cat").agg(n=("rec", "size"), rmse_mean=("rmse", "mean"), rmse_median=("rmse", "median"), rmse_if_no_anchor=("rmse_none", "mean"),
                                 sse_share=("sse", lambda s: s.sum() / S.sse.sum()), big_gt_0p2=("rmse", lambda s: int((s > .2).sum())), big_gt_0p5=("rmse", lambda s: int((s > .5).sum())),
                                 high_ec_days=("ymean", lambda s: int((s >= 1).sum())))
        print(f"\n=== {nm} | {part}: records {len(S)}, overall RMSE {np.sqrt(S.sse.sum() / (24 * len(S))):.4f}"); print(A.round(3).to_string())
    print("\n   biggest-error records (RMSE > .5):"); print(T[T.rmse > .5].sort_values("rmse", ascending=False)[["rec", "cat", "pass2", "rmse", "rmse_none", "ymean"]].round(3).to_string(index=False))
# 추가(EHF1 후속): 2차 '있는데 못 찾음' 기록별 오차와 못 찾은 이유(v7 구성)
print("\n=== PASS-2 HAS-missed under v7: per-record")
D15 = pd.read_csv(os.path.join(OUT, "v15pi15_mirror.csv")).set_index("rec")
for r, g in G.groupby("rec"):
    if int(r[4:7]) < 179 or status(r) != "has": continue
    hs = hidden[int(g.fold.iloc[0])]; a = V7.get(r)
    if a is not None and a not in hs: continue
    why = "v7 prev hidden by DIAG10 fold" if a is not None else ("truth hidden by DIAG10 fold" if TP[r] in hs else "linker not confident")
    rm_ = float(np.sqrt(np.mean((g.sg.values - g.y.values) ** 2)))
    print(f"  {r} rmse {rm_:.3f} sse {np.sum((g.sg.values - g.y.values) ** 2):.2f} ymean {g.y.mean():.2f} | {why} | v15 top {D15.prev0.get(r)} p0 {D15.p0.get(r, np.nan):.2f} truth {TP[r]}")
