# -*- coding: utf-8 -*-
"""HR5 (2026-10-14 집 클로드; 사용자: '정식 시험 진행해봐').  사전 고정(커밋 후 실행).  EC.  자기 기록 입력 + 기본 예측만(일반 방법; 섞인 기록 복원 안 씀).
HR2~HR4(6.887~6.889, 비인과 상한): 하루 수준 잔차를 TabPFN(요약 45)으로 맞히면 전체 −11%, 보통 날까지 개선하는 유일한 모델.
인과판: 기준 시각 h' ∈ {0,3,6,9,12,15,18,21} 마다 TabPFN v2(random_state 0) 하나 —
  특징(45) = [입력 14종 + 기본 예측] 각각의 0..h' 누적 평균, 0..min(h',7) 평균, 9..h' 평균(h' < 9 면 누적 평균으로 채움);  목표 = 하루 수준 잔차 d = 실제 하루 평균 − 기본 하루 평균.
  시각 h 예측 = 기본_h + λ · d̂_{h'}, h' = h 이하 가장 큰 기준 시각 (0..h 입력만 → 인과).
기준(기본) = 연구실 ec_rv1 CUR sg(지금 EC 파이프라인 SG2 단계, 시드 47·1414·6464; EG1 과 같은 기준).
학습 행 = cp2 학습 정답 기록(잠금 제외, 기본 = cp2 base_oof DIAG10 OOF 시드 평균) 중 그 폴드 숨김 아닌 것:
  숨김 = 검증 기록 + 같은 농장 ±1 번호 + 다른 농장 −3..+3 번호.  [한계: 학습 행 base_oof 를 만든 DIAG10 모델이 검증 정답을 봤을 수 있음(2차 노출)]
λ 선택: {.5, 1} 중 DIAG10(전체 행, 3시드 평균 RMSE)에서 낮은 것 — DIAG10 은 판정에서 손해 감시로만.
판정(EC 분야 배정 규칙, 링커 안 씀 → EV 해당 없음, k = 1): (1) P2LOO·EL1 3시드 모두 2차 dSSE < 0 AND (P2LOO 시드 평균 농장 x 5일 블록 P(악화) < .025 OR 나빠진 2차 기록 0);
  (3) DIAG10 손해 감시: 시드 평균 악화 AND '후보 나음' 비율 < .025 이면 FAIL.
서술: 보통/고EC, 시각대, 장부 기록(F13_231·233·F47_216·229·241·F13_217·243·F47_231) 제외 dSSE.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import env_extra_gpu  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
DEV = "cuda" if torch.cuda.is_available() else "cpu"
USE = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
HS = (0, 3, 6, 9, 12, 15, 18, 21); LEDGER = {"F13_231", "F13_233", "F47_216", "F47_229", "F47_241", "F13_217", "F13_243", "F47_231"}
OTHER = {"F13": "F47", "F47": "F13"}
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); X = X[X.row_id.str[:3].isin(["F13", "F47"])]
X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
INP = {r: g.sort_values("h")[USE].values.astype(float) for r, g in X.groupby("rec")}
def feat(inp, base, h):
    Z = np.concatenate([inp, base[:, None]], 1)
    with np.errstate(all="ignore"):
        cm = np.nanmean(Z[:h + 1], 0); nt = np.nanmean(Z[:min(h, 7) + 1], 0); dy = np.nanmean(Z[9:h + 1], 0) if h >= 9 else cm
    return np.concatenate([cm, nt, dy])
A = pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz")); A = A[A.is_train & ~A.lock].sort_values(["rec", "hour"])
TR = {}
for r, g in A.groupby("rec"):
    if len(g) == 24 and np.isfinite(g.base_oof.values).all() and np.isfinite(g.y.values).sum() >= 12 and r in INP:
        TR[r] = (g.base_oof.values, float(np.nanmean(g.y.values) - g.base_oof.mean()))
TRK = sorted(TR); TF = {h: np.stack([feat(INP[r], TR[r][0], h) for r in TRK]) for h in HS}; TD = np.array([TR[r][1] for r in TRK])
for h in HS: TF[h] = np.where(np.isfinite(TF[h]), TF[h], np.nanmean(TF[h], 0))
tfarm = np.array([r[:3] for r in TRK]); tnum = np.array([int(r[4:]) for r in TRK]); print("corrector training records", len(TRK), flush=True)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
DH = {}   # (validator, fold, seed, rec) -> 24 d-hat
for v in ("DIAG10", "EL1", "P2LOO"):
    for k, Gk in O[O.validator == v].groupby("fold"):
        val = set(Gk.rec); hide = np.zeros(len(TRK), bool)
        for r in val:
            f, n = r[:3], int(r[4:]); hide |= ((tfarm == f) & (np.abs(tnum - n) <= 1)) | ((tfarm == OTHER[f]) & (np.abs(tnum - n) <= 3))
        tr = ~hide
        models = {}
        for h in HS:
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=DEV, n_estimators=4, random_state=0, ignore_pretraining_limits=True)
            m.fit(TF[h][tr], TD[tr]); models[h] = m
        for (sd, r), g in Gk.groupby(["seed", "rec"]):
            g = g.sort_values("hour"); base = g.sg.values
            if len(base) != 24: continue
            Q = np.stack([feat(INP[r], base, h) for h in HS]); Q = np.where(np.isfinite(Q), Q, np.stack([TF[h].mean(0) for h in HS]))
            dh = {h: float(models[h].predict(Q[j:j + 1])[0]) for j, h in enumerate(HS)}
            DH[(v, k, sd, r)] = np.array([dh[max(x for x in HS if x <= hh)] for hh in range(24)])
        print(v, k, "done", flush=True)
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
def build(lam):
    rows = []
    for (v, k, sd), G in O.groupby(["validator", "fold", "seed"]):
        for r, g in G.groupby("rec"):
            g = g.sort_values("hour")
            if (v, k, sd, r) not in DH: continue
            rows.append(pd.DataFrame(dict(validator=v, seed=sd, rec=r, farm=g.farm.values, day=g.day.values, hour=g.hour.values, y=g.sub_ec.values, base=g.sg.values, new=g.sg.values + lam * DH[(v, k, sd, r)])))
    return pd.concat(rows, ignore_index=True)
sel = {}
for lam in (.5, 1.0):
    B = build(lam); g = B[B.validator == "DIAG10"]; sel[lam] = np.mean([rm(x.new - x.y) for _, x in g.groupby("seed")])
LAM = min(sel, key=sel.get); print("lambda selection on DIAG10 (all rows, seed-mean RMSE):", {k: round(v, 5) for k, v in sel.items()}, "-> lambda", LAM)
B = build(LAM); B["d"] = (B.new - B.y) ** 2 - (B.base - B.y) ** 2; B["p2"] = B.day >= 179; B["hi"] = B.groupby(["validator", "seed", "rec"]).y.transform("mean") >= 1.0
res = []
for (v, sd), g in B.groupby(["validator", "seed"]):
    p2 = g.p2.values; f = lambda m: 100 * (rm((g.new - g.y)[m]) / rm((g.base - g.y)[m]) - 1) if m.any() else np.nan
    res.append(dict(val=v, seed=sd, dSSE_p2=float(g.d[p2].sum()), p2=f(p2), all=f(np.ones(len(g), bool)), p2_normal=f(p2 & ~g.hi.values), p2_high=f(p2 & g.hi.values),
                    p2_noledger_dSSE=float(g.d[p2 & ~g.rec.isin(LEDGER).values].sum())))
S = pd.DataFrame(res); pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
def boot(v, p2only):
    g = B[(B.validator == v) & (B.p2 if p2only else True)]
    s = g.groupby(["seed", g.farm + "_" + (g.day // 5).astype(str)]).d.mean().groupby(level=1).mean().values
    m = s[np.random.default_rng(20261223).integers(0, len(s), (20000, len(s)))].mean(1); return float((m >= 0).mean()), float((m < 0).mean())
main = S[S.val.isin(["P2LOO", "EL1"])]; c1 = bool((main.dSSE_p2 < 0).all()); pP2, _ = boot("P2LOO", True)
RD = B[B.validator.isin(["P2LOO", "EL1"]) & B.p2].groupby(["validator", "seed", "rec"]).d.sum(); nh = int((RD > 1e-6).sum()); c1b = pP2 < .025 or nh == 0
print(f"\n(1) main P2LOO+EL1: cells better {int((main.dSSE_p2 < 0).sum())}/{len(main)} | P2LOO block P(worse) {pP2:.4f} | harmed record-cells {nh} -> {'PASS' if c1 and c1b else 'FAIL'}")
dg = S[S.val == "DIAG10"]; _, pb = boot("DIAG10", False); c3 = not (B[B.validator == "DIAG10"].groupby("seed").d.sum().mean() > 0 and pb < .025)
print(f"(3) DIAG10 harm guard: all-rows change {dg['all'].round(2).tolist()}% share(better) {pb:.3f} -> {'PASS' if c3 else 'FAIL'}")
hrs = B[B.validator == "P2LOO"]; print("P2LOO pass-2 by hour block (dSSE, seed mean):", hrs[hrs.p2].groupby(hrs.hour // 6).d.sum().div(3).round(3).to_dict())
print("P2LOO pass-2 record dSSE (seed mean), top:", RD.loc["P2LOO"].groupby(level="rec").mean().sort_values().round(3).head(6).to_dict(), "| worst:", RD.loc["P2LOO"].groupby(level="rec").mean().sort_values().round(3).tail(5).to_dict())
print("HR5:", "ADOPT" if (c1 and c1b and c3) else "FAIL")
B.to_csv(os.path.join(HERE, "hr5_rows_v1.csv.gz"), index=False)
