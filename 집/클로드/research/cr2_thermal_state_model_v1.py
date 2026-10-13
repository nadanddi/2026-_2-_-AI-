# -*- coding: utf-8 -*-
"""CR2 (2026-10-13 집 클로드; 사용자: 'RNN 맞춤으로 모델 새로 만들어 보자').  사전 고정(커밋 후 실행).  온도.
CR1(GRU) 실패 원인(과적합·추운 외삽 폭주)에 맞춘 '작은 물리형 순환 모델' — 배지 온도 상태 S 를 시간마다 갱신:
   S_t = S_{t-1} + α·(Tin_t − S_{t-1}) + β·rad_t/100 + γ·heat_t/100 + δ·thermal_t/100 + ε,   α ∈ (0,1) (logit 매개)
   초기 상태 S_{-1}: 같은 동 전날(v15R 일치/고확신 짝 → v7 앵커 짝) 23시 배지 정답이 보이면 그 값 − .36(앵커 편향, 13회차와 같음), 아니면 기준 예측 q_0.
   매개변수: 온실(F13/F47)별 5개 = 10개.  적합: 학습 폴드 기록들의 시간별 제곱오차 최소(Nelder–Mead/L-BFGS, 시작 α .3, 나머지 0).
   입력은 그 기록 0..t 시 입력(인과) + 학습 정답(전날).  규정 안.
학습원: TC1b·CR1 과 같이 DIAG10 기록 공통, 배치·폴드 k 의 시험 기록과 같은 온실 ±1·다른 온실 ±5 번호 기록 제외 → 폴드별 매개변수.
안(2개, α .0125): COLD = base + .5·g·(S − base), g = clip((10 − in_temp)/2, 0, 1)(결측 0)  [범위 COLD → 주 EXT10];
                 ALL  = base + .2·(S − base)  [범위 ALL → 주 DIAG10·EXT10].   .5/.2 는 고정(맞추지 않음).
기준 base = 14회차형(셀 + v7 앵커 + TNA8 앵커, 측정 감쇠).  판정: 주 배치 4칸 모두 개선 & 농장×5일 블록 부트스트랩 P(악화) < .0125;
  나머지 배치는 칸 평균 악화이면서 '후보가 나음' 비율 < .0125 이면 FAIL.  서술: 물리 모델 단독 RMSE(배치별), 추운 행(≤ 8℃) RMSE.
"""
import os, sys, importlib.util
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
src = open(os.path.join(HERE, "cr1_chain_gru_temp_v1.py"), encoding="utf-8").read()
head = src[:src.index("class Net(nn.Module):")].replace("import env_extra_gpu  # noqa: F401\n", "").replace("import torch, torch.nn as nn\n", "").replace('DEV = "cuda" if torch.cuda.is_available() else "cpu"', 'DEV = None')
exec(head)   # A7, LC (v7 + TNA8), PREV, ST, idx, TRAIN, FARM, NUM, CUBE (in_temp,in_hum,out_temp,out_rad,act_heating,act_thermal,act_vent)
rng = np.random.default_rng(20261123)
IT, RAD, HEAT, THERM = 0, 3, 4, 5
def simulate(th, recs, q0map, hidden):
    out = {}
    for r in recs:
        b = idx[r]; f = 0 if FARM[b] == "F13" else 1; a, bb, gg, dd, ee = th[f * 5:(f + 1) * 5]; al = 1 / (1 + np.exp(-a))
        pv = PREV.get(r); s = q0map[r]
        if pv is not None and TRAIN[idx[pv]] and pv not in hidden and np.isfinite(ST[idx[pv], 23]): s = ST[idx[pv], 23] - .36
        x = np.nan_to_num(CUBE[b]); traj = np.zeros(24)
        for t in range(24):
            tin = x[t, IT] if np.isfinite(CUBE[b, t, IT]) else s
            s = s + al * (tin - s) + bb * x[t, RAD] / 100 + gg * x[t, HEAT] / 100 + dd * x[t, THERM] / 100 + ee; traj[t] = s
        out[r] = traj
    return out
def fit(recs, q0map, ymap, hidden):
    def loss(th):
        sim = simulate(th, recs, q0map, hidden); e = np.concatenate([sim[r] - ymap[r] for r in recs]); return float(np.nanmean(e ** 2))
    th0 = np.array([np.log(.3 / .7), 0, 0, 0, 0] * 2, float)
    return minimize(loss, th0, method="L-BFGS-B").x
FR = A7.A2.A1.frames()
def layout_maps(M, cells):
    key = (M.farm + "_" + M.day.map("{:03d}".format)).values; h = M.hour.values; y = M.sub_temp.values
    kb, Ab = A7.build(M, LC); anc = [A7.correct(M, c, kb, Ab, .8, "const", "meas", 2, 1.5) for c in cells]
    recs = sorted(set(key)); pos = {r: np.where(key == r)[0] for r in recs}
    def per_rec(arr): return {r: np.array([arr[pos[r]][h[pos[r]] == t][0] if (h[pos[r]] == t).any() else np.nan for t in range(24)]) for r in recs}
    return key, h, y, anc, recs, pos, per_rec(np.mean(anc, 0)), per_rec(y)
_, _, _, _, DREC, _, DB, DY = layout_maps(*FR["DIAG10"])
def purge(te):
    fd = {(FARM[idx[r]], int(NUM[idx[r]])) for r in te}; out = set()
    for r in DREC:
        f, d = FARM[idx[r]], int(NUM[idx[r]])
        if any((f == f2 and abs(d - d2) <= 1) or (f != f2 and abs(d - d2) <= 5) for f2, d2 in fd): out.add(r)
    return out
res, blocks, phys = [], {}, []
for L, (M, cells) in FR.items():
    key, h, y, anc, recs, pos, B, Y = layout_maps(M, cells)
    fold_of = {r: M.fold.values[pos[r][0]] for r in recs}; S = {}
    for k in sorted(set(fold_of.values())):
        te = [r for r in recs if fold_of[r] == k]; hid = set(te); tr = [r for r in DREC if r not in purge(te)]
        th = fit(tr, {r: DB[r][0] for r in tr}, DY, hid)
        S.update(simulate(th, te, {r: B[r][0] for r in te}, hid))
    srow = np.zeros(len(M))
    for r in recs:
        ix = pos[r]; srow[ix] = S[r][h[ix]]
    inT = M.in_temp.values; g = np.nan_to_num(np.clip((10 - inT) / 2, 0, 1), nan=0.0); cold = inT <= 8
    phys.append(dict(layout=L, phys_rmse=rm(srow - y), base_rmse=rm(np.mean(anc, 0) - y), phys_cold=rm((srow - y)[cold]) if cold.any() else np.nan, base_cold=rm((np.mean(anc, 0) - y)[cold]) if cold.any() else np.nan, n_cold=int(cold.sum())))
    for ci, base in enumerate(anc):
        for a, new in (("COLD", base + .5 * g * (srow - base)), ("ALL", base + .2 * (srow - base))):
            p2 = M.day.values >= 179
            res.append(dict(layout=L, cell=ci, arm=a, pct=100 * (rm(new - y) / rm(base - y) - 1), p2=100 * (rm((new - y)[p2]) / rm((base - y)[p2]) - 1)))
            if ci == 0:
                bk = M.farm + "_" + (M.day // 5).astype(str); blocks[(L, a)] = pd.Series((new - y) ** 2 - (base - y) ** 2).groupby(bk.values).mean()
    print("done", L, flush=True)
pd.set_option("display.width", 220)
print(pd.DataFrame(phys).round(3).to_string(index=False))
Sx = pd.DataFrame(res); print(Sx.pivot_table(index=["arm", "layout"], columns="cell", values="pct").round(3).to_string())
def bootP(L, a):
    v = blocks[(L, a)].values; ix = rng.integers(0, len(v), (20000, len(v))); m = v[ix].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
for a, prim in (("COLD", ["EXT10"]), ("ALL", ["DIAG10", "EXT10"])):
    ok = True; msg = []
    for L in sorted(set(Sx.layout)):
        c_ = Sx[(Sx.arm == a) & (Sx.layout == L)].pct; pw, pb = bootP(L, a)
        if L in prim:
            c = bool((c_ < 0).all()) and pw < .0125; ok &= c; msg.append(f"{L}[primary] cells {int((c_ < 0).sum())}/4 P(worse) {pw:.4f} -> {'ok' if c else 'FAIL'}")
        else:
            bad = c_.mean() > 0 and pb < .0125; ok &= not bad; msg.append(f"{L} mean {c_.mean():+.3f}% P(better) {pb:.3f}{' SIGNIF WORSE' if bad else ''}")
    print(f"\n== CR2 {a}:"); print("\n".join("   " + m for m in msg)); print(f"CR2 {a}:", "ADOPT" if ok else "FAIL")
