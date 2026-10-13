# -*- coding: utf-8 -*-
"""TL1-T4 (2026-10-14 집 클로드; 비평 T4).  사전 고정(커밋 후 실행).  EC.  대리 모델(P2T2 와 같은 ET + DP1 + 계절, 시드 47·1414·6464).
비평(6.870): TL1 증거는 2차 3기록뿐 → '목표 날짜 가까운 시기 자료로만 학습하면 보통 날이 나아진다'가 진짜라면 1차 보통 기록 300여 개에서도 보여야 함.
설계: 1차 학습 기록을 농장 x 번호 10개 묶음으로 폴드(검증).  학습 제외 = 같은 농장 ±1, 다른 농장 −3..+3, 잠금 ±1(TL1 과 같음).
  REF = 남은 전체 학습.  NEAR_W = 남은 학습 중 학습 달력(DMT3e)에서 검증 날짜 중 하나와 같은 구간·날짜 차 <= W 인 날 (W 후보 5·10·20; 연구자 범위).
  관문 = REF 0..h 누적평균 >= 1.0 → REF, 아니면 NEAR_W (TL);  '내리기만' 판 = 관문 아니면 min(REF, NEAR_W) (TL-low).
  NEAR_W 학습 날 < 30 이면 그 폴드는 REF 그대로(작은 학습 방지).
선택/판정 분리: W 와 판(TL/TL-low)은 F13 검증 폴드에서 3시드 평균 RMSE 가 가장 낮은 것으로 고르고, 판정은 F47 검증 폴드에서만.
판정(미리 정함): F47 1차 검증 행에서 3시드 모두 후보 < REF AND 폴드 부트스트랩(시드 평균) P(낫지 않음) < .025.
  통과 → T2(TL1-D 실제 파이프라인 확인) 진행 허가;  실패 → TL1 계열 종료.
서술: 보통/고EC 날 분해, 다른 농장 고EC ±4 번호 근처 기록 제외 민감도.
"""
import env  # noqa: F401
import importlib.util, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dp1", os.path.join(HERE, "ec3_DP1_daily_operation_pattern_v1.py"))
dp1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dp1)
dc4, p3, core = dp1.dc4, dp1.p3, dp1.core
raw, full, lab, lock, signatures, fds = p3.prepare()
lab = lab.join(pd.concat([dp1.day_feats(g) for _, g in lab.groupby(["farm", "day"])]))
wv = dc4.weather_vectors(full); FS = [c for c in core.FULL if c != "day"] + ["season"] + dp1.NEW
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); OTHER = {"F13": "F47", "F47": "F13"}; SEEDS = (47, 1414, 6464); WS = (5, 10, 20)
key = lambda f, d: f"{f}_{int(d):03d}"
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAY = {r: (s_, d_) for r, s_, d_ in zip(CAL.rec, CAL.seg, CAL.day)}
days = lab[["farm", "day"]].drop_duplicates(); p1 = days[days.day < 179].sort_values(["farm", "day"])
folds = []
for f, g in p1.groupby("farm"):
    dd = list(g.day)
    for k in range(0, len(dd), 10): folds.append((f, {(f, int(d)) for d in dd[k:k + 10]}))
print("folds", len(folds), flush=True)
out = []
for fi, (vf, vd) in enumerate(folds):
    va_m = np.array([(f, int(d)) in vd for f, d in zip(lab.farm, lab.day)])
    forb = {(f, d + j) for f, d in vd for j in (-1, 0, 1)} | {(OTHER[f], d + j) for f, d in vd for j in range(-3, 4)} | {(f, d + j) for f, d in lock for j in (-1, 0, 1)}
    tr = lab[np.array([(f, int(d)) not in forb for f, d in zip(lab.farm, lab.day)])].copy(); va = lab[va_m].copy()
    vdays = va[["farm", "day"]].drop_duplicates().reset_index(drop=True)
    season, vq = dc4.season_index(tr[["farm", "day"]].drop_duplicates(), vdays, wv)
    tr["season"] = [season[(f, d)] for f, d in zip(tr.farm, tr.day)]
    va = va.merge(vdays.assign(season=vq), on=["farm", "day"], how="left").set_index(va.index)
    vdates = [DAY[key(f, d)] for f, d in vd if key(f, d) in DAY]
    kk = [key(f, d) for f, d in zip(tr.farm, tr.day)]
    G = pd.DataFrame(dict(fold=fi, vfarm=vf, farm=va.farm.values, day=va.day.values, hour=va.hour.values, y=va.sub_ec.values))
    for sd in SEEDS:
        G[f"REF_{sd}"] = p3.final(core.predict_model(core.et(sd), tr, va, FS), tr, va)
    for W in WS:
        m = np.array([k in DAY and any(DAY[k][0] == s2 and abs(DAY[k][1] - d2) <= W for s2, d2 in vdates) for k in kk]); nd = tr[m][["farm", "day"]].drop_duplicates().shape[0]
        G[f"nd_{W}"] = nd
        for sd in SEEDS:
            G[f"N{W}_{sd}"] = p3.final(core.predict_model(core.et(sd), tr[m], va, FS), tr[m], va) if nd >= 30 else G[f"REF_{sd}"]
    out.append(G); print(f"fold {fi} {vf} done | NEAR days {[int(G[f'nd_{W}'].iloc[0]) for W in WS]} of {tr[['farm', 'day']].drop_duplicates().shape[0]}", flush=True)
D = pd.concat(out, ignore_index=True).sort_values(["fold", "farm", "day", "hour"]).reset_index(drop=True)
D["hi"] = D.groupby(["farm", "day"]).y.transform("mean") >= 1.0
for sd in SEEDS:
    cm = D.groupby(["fold", "farm", "day"])[f"REF_{sd}"].transform(lambda x: x.expanding().mean()); D[f"gate_{sd}"] = cm >= 1.0
    for W in WS:
        D[f"TL{W}_{sd}"] = np.where(D[f"gate_{sd}"], D[f"REF_{sd}"], D[f"N{W}_{sd}"])
        D[f"TLlow{W}_{sd}"] = np.where(D[f"gate_{sd}"], D[f"REF_{sd}"], np.minimum(D[f"REF_{sd}"], D[f"N{W}_{sd}"]))
D.to_csv(os.path.join(HERE, "tl1t4_preds_v1.csv.gz"), index=False)
arms = [f"{a}{W}" for a in ("TL", "TLlow") for W in WS]
def score(g, a): return np.mean([rm(g[f"{a}_{sd}"] - g.y) for sd in SEEDS])
F13 = D[D.vfarm == "F13"]; F47 = D[D.vfarm == "F47"]
sel = pd.Series({a: score(F13, a) for a in arms}); ref13 = score(F13, "REF"); print(f"\nF13 selection: REF {ref13:.4f} | " + " ".join(f"{a} {v:.4f}" for a, v in sel.items()))
best = sel.idxmin(); print("selected:", best)
pct = {sd: 100 * (rm(F47[f"{best}_{sd}"] - F47.y) / rm(F47[f"REF_{sd}"] - F47.y) - 1) for sd in SEEDS}
dd = np.mean([((F47[f"{best}_{sd}"] - F47.y) ** 2 - (F47[f"REF_{sd}"] - F47.y) ** 2).groupby(F47.fold).sum().values for sd in SEEDS], 0)
ix = np.random.default_rng(20261217).integers(0, len(dd), (20000, len(dd))); P = float((dd[ix].sum(1) >= 0).mean())
ok = all(v < 0 for v in pct.values()) and P < .025
print(f"F47 judge {best} vs REF: seeds {[round(v, 2) for v in pct.values()]}% | fold-bootstrap P(not better) {P:.4f} | folds better {int((dd < 0).sum())}/{len(dd)} -> {'PASS' if ok else 'FAIL'}")
for nm, msk in (("normal", ~F47.hi), ("high", F47.hi)):
    g = F47[msk]; print(f"   F47 {nm}: REF {score(g, 'REF'):.4f} {best} {score(g, best):.4f}")
print("all arms on F47 (descriptive):", {a: round(100 * (score(F47, a) / score(F47, 'REF') - 1), 2) for a in arms})
# sensitivity: drop records within ±4 numbers of an other-farm high-EC labelled day
hiset = {(f, int(d)) for (f, d), v in lab.groupby(["farm", "day"]).sub_ec.mean().items() if v >= 1.0}
near_hi = np.array([any((OTHER[f], int(d) + j) in hiset for j in range(-4, 5)) for f, d in zip(F47.farm, F47.day)])
g = F47[~near_hi]; print(f"sensitivity (drop near other-farm high, n rows {len(g)}): {best} {100 * (score(g, best) / score(g, 'REF') - 1):+.2f}%")
print("TL1-T4:", "PASS -> T2 allowed" if ok else "FAIL -> stop TL1 line")
