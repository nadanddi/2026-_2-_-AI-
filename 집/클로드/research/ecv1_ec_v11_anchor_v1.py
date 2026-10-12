# -*- coding: utf-8 -*-
"""ECV1 (2026-10-12 집 클로드; 자율 — 근거: EC 앵커 확장 실패의 주원인은 틀린 짝(F47_146 같은 다른 ID 오연결, 6.716), v11 은 다른 ID 전날 정확 .77 → .87·2차 확신 ≥ .6 정확 .97).  사전 고정(이 파일 커밋 후 실행).
EC 검증기 DIAG10·P2LOO·EL1(연구실/클로드/results/ec_rv1_preds_v1.csv, CUR, sg, 시드 47/1414/6464).  기준 NONE = sg(앵커 없음 = 12회차 EC 구성에 해당; W2 는 평가 순수 창 전용).
안: V7 = v7 거울 CV 앵커(p ≥ .8·정답·|dT| ≤ 1.5; 13회차 EC 구성) / V11 = v11 거울 CV 짝(p ≥ .6·정답·날짜 일치, 실내 차 제한 없음 — EC) / V7V11 = V7 ∪ V11(V7 우선).
  EA2 식: sg_h + (A − sg 0..h 누적평균), A = 짝 23시 EC 정답; 짝이 같은 검증기·시드·폴드의 숨김(val) 기록이면 버림; 학습 범위로 자름.
판정(본페로니 2안 α .0125, 비교 대상 = NONE 과 V7 각각에 대해 V11·V7V11): 3검증기 x 3시드 모두 2차 행 RMSE 개선 AND DIAG10 2차 기록 부트스트랩 P(악화) < .0125(시드마다) AND 상위 1기록 몫 ≤ .5.
서술: 전체 행, 앵커 수.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, X as OUTX, TRAIN  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(HERE, "ec0h", "rise", "dm11_calendar_v1.csv")); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
P11 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v11_mirror_picks.csv")); P11 = P11[P11.p >= .6]
def mirror_lab(b):
    f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
def tdate(b):
    lab = mirror_lab(b); vi = np.where(lab)[0]; eq = np.all(np.abs(OUTX[vi, 0] - OUTX[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    return pd.Series(ds).mode().iloc[0] if len(ds) else np.nan
V11 = {}
for r, p in zip(P11.rec, P11.prev):
    g = tdate(idx[r]) - DAY[idx[p]]
    if np.isfinite(g) and g != 1: continue
    V11[r] = p
ARMS = {"V7": V7, "V11": V11, "V7V11": {**V11, **V7}}
print({k: len(v) for k, v in ARMS.items()})
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
lo, hi = O.sub_ec.min(), O.sub_ec.max(); rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261024)
res = []; dsse = {}
for (v, sd), G in O.groupby(["validator", "seed"]):
    G = G.sort_values(["rec", "hour"]).reset_index(drop=True); y = G.sub_ec.values; sg = G.sg.values; p2 = G.day.values >= 179
    cm = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()).values
    hidden = G.groupby("fold").rec.apply(set).to_dict(); fold_of = G.groupby("rec").fold.first().to_dict()
    preds = {"NONE": sg}
    for arm, A in ARMS.items():
        new = sg.copy()
        for rec, Gr in G.groupby("rec"):
            if rec not in A: continue
            pv = A[rec]
            if pv in hidden.get(fold_of[rec], set()): continue
            ix = Gr.index.values; new[ix] = np.clip(sg[ix] + (EC[idx[pv], 23] - cm[ix]), lo, hi)
        preds[arm] = new
    for arm in ("V7", "V11", "V7V11"):
        for basen in ("NONE", "V7"):
            if arm == basen: continue
            b, n = preds[basen], preds[arm]
            res.append(dict(val=v, seed=sd, arm=arm, base=basen, all_pct=100 * (rm(n - y) / rm(b - y) - 1), p2_pct=100 * (rm((n - y)[p2]) / rm((b - y)[p2]) - 1)))
            if v == "DIAG10":
                d = pd.Series(((n - y) ** 2 - (b - y) ** 2)[p2]).groupby(G.rec.values[p2]).sum(); dsse[(arm, basen, sd)] = d
S = pd.DataFrame(res); pd.set_option("display.width", 220)
print(S.pivot_table(index=["arm", "base"], columns=["val", "seed"], values="p2_pct").round(2).to_string()); print()
print(S.pivot_table(index=["arm", "base"], columns=["val"], values="all_pct", aggfunc="mean").round(2).to_string())
for arm in ("V11", "V7V11"):
    for basen in ("NONE", "V7"):
        g = S[(S.arm == arm) & (S.base == basen)]; Ps = []; shares = []
        for sd in (47, 1414, 6464):
            d = dsse[(arm, basen, sd)].values; Ps.append(float((d[rng.integers(0, len(d), (20000, len(d)))].sum(1) >= 0).mean()))
            gain = -d; shares.append(gain.max() / gain.sum() if gain.sum() > 0 else np.nan)
        ok = (g.p2_pct < 0).all() and all(p < .0125 for p in Ps) and np.nanmean(shares) <= .5
        print(f"{arm} vs {basen}: all p2 better {bool((g.p2_pct < 0).all())} | DIAG10 P {np.round(Ps, 4).tolist()} | top share {np.nanmean(shares):.2f} -> {'ADOPT' if ok else 'FAIL'}")
