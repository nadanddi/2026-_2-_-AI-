# -*- coding: utf-8 -*-
"""DG1 (2026-10-12 집 클로드; 날짜 관문 — 앵커 전날의 달력 날짜가 대상 날짜 − 1 이 아니면 앵커 제거).  사전 고정 — 실행 전 커밋.
근거(서술 DC1): 거울 CV 에서 날짜 틀린 선택의 정확도 .05(날짜 맞음 .86).  달력 DM11 = 외기 묶음 + 학습 정답 확실 연결 (train 만 사용, 규정 안).
EV: EA2 앵커 기록 중 v7m 전날의 날짜 ≠ 대상 − 1 인 기록 → SG2 로 되돌림(앵커 없음).  대상 날짜·전날 날짜 중 하나라도 모르면 그대로 둠.
DIAG10: SH2 와 같은 방식(p ≥ .8, 정답 있는 같은 ID, 숨김 아님)으로 sg 에 EA2 식 앵커를 붙인 판 vs 같은 판에 날짜 관문.
판정(검증기마다): 바뀐 기록 RMSE 3시드 모두 개선 AND 부트스트랩 P(worse) < .025(EV 덩어리, DIAG10 기록) AND 최대 기록 몫 ≤ .5; 두 검증기 모두 PASS 여야 채택 후보.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
def date_bad(t):
    if t not in L.index: return False
    dt, dp = DAY.get(t, np.nan), DAY.get(L.at[t, "prev"], np.nan)
    return bool(np.isfinite(dt) and np.isfinite(dp) and dp != dt - 1)
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e)))); rng = np.random.default_rng(20261244)
def judge(name, base, new, Y, groups):
    recs = sorted(set.union(*[set(r for r in new[s].index if not np.allclose(new[s].loc[r].values, base[s].loc[r].values, equal_nan=True)) for s in new]))
    if not recs: print(name, "no changed records"); return
    cells = [rm((new[s].loc[recs] - Y.loc[recs]).values) / rm((base[s].loc[recs] - Y.loc[recs]).values) - 1 for s in new]
    per = pd.Series(np.mean([(((new[s].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for s in new], 0), index=recs)
    cl = per.groupby(groups(recs)).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
    share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
    print(f"{name}: changed {len(recs)} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | P {pw:.4f} | top share {share:.2f}"
          f" -> {'PASS' if all(x < 0 for x in cells) and pw < .025 and share <= .5 else 'FAIL'}")
    print("   per record:", ", ".join("%s %+.3f" % (a, b) for a, b in per.sort_values().items()))
# ---- EV ----
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); ch = Fv[Fv.role == "val"].set_index("rec").chunk
anc = R.groupby("rec").anchored.first(); bad = [t for t in anc.index if bool(anc[t]) and date_bad(t)]
print("EV anchored records with date-wrong predecessor:", bad)
Y = R.groupby(["rec", "hour"]).y.first().unstack(); base, new = {}, {}
for sd, G in R.groupby("seed"):
    E = G.pivot_table(index="rec", columns="hour", values="EA2"); S = G.pivot_table(index="rec", columns="hour", values="SG2")
    N_ = E.copy(); N_.loc[bad] = S.loc[bad]; base[sd], new[sd] = E, N_
judge("EV date gate", base, new, Y, lambda recs: ch.reindex(recs).values)
# ---- DIAG10 ----
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
Yl = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Yl["rec"] = Yl.row_id.str[:7]; Yl["h"] = Yl.row_id.str[8:].astype(int); EC23 = Yl[Yl.h == 23].set_index("rec").sub_ec
YD = O.groupby(["rec", "hour"]).sub_ec.first().unstack(); bD, nD = {}, {}; bad_d = set()
for sd, G in O.groupby("seed"):
    Q = G.pivot_table(index="rec", columns="hour", values="sg"); fo = G.groupby("rec").fold.first().reindex(Q.index)
    hidden = {k: {(r[:3], int(r[4:7]) + j) for r in fo[fo == k].index for j in (-1, 0, 1)} for k in fo.unique()}
    cm = Q.cumsum(1) / np.arange(1, 25); A1 = Q.copy(); A2 = Q.copy()
    for r in Q.index:
        if r not in L.index: continue
        pr = L.loc[r]
        if pr.xf or pr.p < .8 or not pr.prev_is_labelled or pr.prev not in EC23.index or (pr.prev[:3], int(pr.prev[4:7])) in hidden[fo[r]]: continue
        v = Q.loc[r].values + (EC23[pr.prev] - cm.loc[r].values); A1.loc[r] = v
        if date_bad(r): bad_d.add(r)
        else: A2.loc[r] = v
    bD[sd], nD[sd] = A1, A2
print("DIAG10 anchored (p>=.8) with date-wrong predecessor:", sorted(bad_d))
judge("DIAG10 date gate", bD, nD, YD, lambda recs: np.array(recs))
