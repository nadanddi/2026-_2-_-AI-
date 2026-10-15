# -*- coding: utf-8 -*-
"""A3 (2026-10-15 집 클로드; 사용자: 5점 감사 후 '1,2,3 순서로' 의 3번 — 절제 표).  서술(판정 없음), 실행 전 커밋.  온도 + EC.
목적: 발표 기준 '제외 시 성능과 영향' + '현장 조건 성능' 을 한 표로.
[온도] 판 DIAG10·EL1·EXT10·EXT12(G5 와 같은 TK2 폴드), 기본 = G5 의 REF / NX / NXL(X 표시 제거 / + 기록 넘는 특징 제거) 각각 위에:
  Y0 = 기본만(W40G-S 꼴)
  Y1 = 지금 '어제' 부품: 사슬 새벽 보정(특징에 번호상 앞 기록 lit1·lit2 포함, 사슬표 CT) + 앵커 LV(v7 + TNA8)
  Y2 = 현장형 복원: 사슬 보정에서 lit1·lit2(번호 규칙) 뺌, 사슬표·앵커는 링커(CT, LV) 그대로
  Y3 = 진짜 전날: Y2 꼴 + 사슬표·앵커를 CLEAN(정답 연속 ∩ 달력 차 1, 낙관 부분집합)
  사슬 보정기 학습 틀 = 같은 기본 판의 DIAG10 프레임(G5 체크포인트로 판마다 다시).
[EC] SZ2 체크포인트(새 시드, DIAG10(2차 포함 폴드)·EL1·P2LOO) 위:
  E0 = pre_NOS(그날 입력만, 날짜 없음) / E1 = pre_SEA(+ 날짜) / E2 = NH(+ HR5) / E3 = F5E(+ EV3L 링커 전날 입력 창) /
  E4 = F5E + EA 앵커(링커 v7 확신 짝의 23시 EC, 그 폴드 검증 기록 정답이면 안 씀) / E5 = F5E + EA 앵커(CLEAN 진짜 전날, 현장 조건 — 숨김 무시) /
  참고: S = fin_SEA(SG2 구조판), SH = S + HR5(15회차형, W2·B5 없음)
보고: RMSE(전체·2차·고EC), 시드 평균.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u a3_field_ablation_table_v1.py
"""
import os, sys, pickle, glob
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "c1_true_yesterday_field_condition_v1.py"), encoding="utf-8").read(); __file__ = os.path.join(HERE, "c1_true_yesterday_field_condition_v1.py")
exec(src[:src.index('print("[TEMP]", flush=True)')])   # tv3 harness + TRUE (CLEAN), CTT, LT
__file__ = os.path.join(HERE, "a3_field_ablation_table_v1.py")
pd.set_option("display.width", 240)
# ---------------- temperature ----------------
G5 = os.path.join(env.LOCAL, "g5_ckpt"); FEAT0 = list(C2.T.FEAT); FEATNL = [f for f in FEAT0 if f not in ("lit1_in23", "lit2_in23")]
for arm in ("REF", "NX", "NXL"):
    d = os.path.join(env.LOCAL, f"a3_{arm}_ckpt"); os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(G5, "*.csv")):
        X = pd.read_csv(f); Y = X[["row_id", "farm", "day", "hour", "sub_temp", "in_temp", "validator", "fold"]].copy()
        Y["base_REF_7"], Y["base_REF_101"], Y["codex_REF"] = X[f"base_{arm}_7"], X[f"base_{arm}_101"], X[f"codex_{arm}"]; Y["pfn_A"] = Y["pfn_B"] = X[f"pfn_{arm}"]
        Y.to_csv(os.path.join(d, os.path.basename(f)), index=False)
    for L in ("DIAG10", "EL1", "EXT10", "EXT12"): T.LAY[f"{arm}_{L}"] = (f"a3_{arm}_ckpt", L + "_", ("base_REF_7", "base_REF_101"), "codex_REF")
rows = []
for arm in ("REF", "NX", "NXL"):
    D_ = T.layout_frame(f"{arm}_DIAG10", CT); DT_ = T.layout_frame(f"{arm}_DIAG10", CTT)
    for L in ("DIAG10", "EL1", "EXT10", "EXT12"):
        key = f"{arm}_{L}"; d_, pre, bs, cx = T.LAY[key]; M = T.layout_frame(key, CT); MT = T.layout_frame(key, CTT); y = M.sub_temp.values; p2 = M.day.values >= 179
        P = {k: [] for k in ("Y0", "Y1", "Y2", "Y3")}
        for bi, b in enumerate(bs):
            for fam in "A":
                a = B.w40(M, b, cx, fam); P["Y0"].append(a)
                C2.T.FEAT[:] = FEAT0; P["Y1"].append(anch(M, C2.corrected(M, a, D_, bi, fam, 7), LV))
                C2.T.FEAT[:] = FEATNL; P["Y2"].append(anch(M, C2.corrected(M, a, D_, bi, fam, 7), LV))
                aT = B.w40(MT, b, cx, fam); P["Y3"].append(anch(MT, C2.corrected(MT, aT, DT_, bi, fam, 7), LT))
                C2.T.FEAT[:] = FEAT0
        for k, v in P.items():
            rows.append(dict(target="temp", base=arm, layer=k, val=L, all=np.mean([rm(p - y) for p in v]), p2=np.mean([rm((p - y)[p2]) for p in v]) if p2.any() else np.nan))
        print("temp", arm, L, "done", flush=True)
RT = pd.DataFrame(rows); print("\n[TEMP] RMSE (cell mean)"); print(RT.pivot_table(index=["val", "base"], columns="layer", values="all").round(4).to_string())
print("\n[TEMP] pass-2 RMSE"); print(RT.pivot_table(index=["val", "base"], columns="layer", values="p2").round(4).to_string())
# ---------------- EC ----------------
SEEDS = (1357, 2468, 3579); R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); MTp = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb")); EP = pickle.load(open(os.path.join(env.LOCAL, "nh2_ev3l_v1.pkl"), "rb"))
CK = os.path.join(env.LOCAL, "sz2_ckpt"); Z = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(CK, "*.csv"))], ignore_index=True)
Z = Z.rename(columns={"validation_fold": "fold"}); Z["rec"] = Z.farm + "_" + Z.day.map("{:03d}".format)
R = R.merge(Z[["validator", "fold", "rec", "hour"] + [f"pre_NOS_{s}" for s in SEEDS] + [f"pre_SEA_{s}" for s in SEEDS]], on=["validator", "fold", "rec", "hour"], how="left")
mt = np.array([MTp[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)]); ev = np.array([EP[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
CURE = {r: p for r, p, q, l, dT in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if isinstance(p, str) and q >= .8 and l and abs(dT) <= 1.5 and r in v7set}
def ea_col(base_col, links, respect_hidden):
    out = R[base_col].values.copy()
    for (v, k, r), g in R.groupby(["validator", "fold", "rec"]):
        p = links.get(r)
        if p is None or not np.isfinite(ECA[idx[p], 23]): continue
        if respect_hidden and p in set(R.rec[(R.validator == v) & (R.fold == k)]): continue
        ii = g.sort_values("hour").index.values; q = R.loc[ii, base_col].values; pm = np.cumsum(q) / np.arange(1, len(q) + 1); out[ii] = q + (ECA[idx[p], 23] - pm)
    return out
for s in SEEDS:
    R[f"E0_{s}"] = R[f"pre_NOS_{s}"]; R[f"E1_{s}"] = R[f"pre_SEA_{s}"]; R[f"E2_{s}"] = (R[f"A1_{s}"] - .1 * mt) / .9; R[f"E3_{s}"] = .9 * R[f"E2_{s}"] + .1 * ev
    R[f"E4_{s}"] = ea_col(f"E3_{s}", CURE, True); R[f"E5_{s}"] = ea_col(f"E3_{s}", TRUE, False); R[f"S_{s}"] = R[f"S_{s}"]; R[f"SH_{s}"] = R[f"SH_{s}"]
rows = []
for v in ("DIAG10", "EL1", "P2LOO"):
    g = R[R.validator == v]; y = g.y.values; p2 = g.p2.values; hi = g.hi.values
    for k in ("E0", "E1", "E2", "E3", "E4", "E5", "S", "SH"):
        rows.append(dict(val=v, layer=k, all=np.mean([rm(g[f"{k}_{s}"].values - y) for s in SEEDS]), p2=np.mean([rm((g[f"{k}_{s}"].values - y)[p2]) for s in SEEDS]),
                         p2_high=np.mean([rm((g[f"{k}_{s}"].values - y)[p2 & hi]) for s in SEEDS]) if (p2 & hi).any() else np.nan))
RE = pd.DataFrame(rows)
for c in ("all", "p2", "p2_high"): print(f"\n[EC] {c} RMSE"); print(RE.pivot(index="val", columns="layer", values=c).round(4).to_string())
pd.concat([RT, RE.assign(target="ec")]).to_csv(os.path.join(HERE, "a3_table_v1.csv"), index=False)
