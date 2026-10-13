# -*- coding: utf-8 -*-
"""HX9-OPS v5 (2026-10-13 집 클로드, 사용량 한도 뒤 재개): 다변량 '비인과 상한'(v1 4단계)의 기준선·누수 점검.
v1 결과를 본 뒤의 사후 점검임을 명시. v1 의 다변량 숫자 두 가지가 해석에 기준선이 빠져 있었음:
  (i)  Y2h(고EC 31) 의 R2_vs_nocorr(+.32~.40)는 '상수 하나(고EC 평균 잔차 +.28)만 더해도' 얻는 몫을 포함.
  (ii) Y2a(전체 360, DIAG10 폴드) GBM(family+sg) R2 +.165, 2차 +.383 은 sg 단독 GBM 기준선이 없고, DIAG10 폴드는
       고EC 에피소드를 여러 폴드로 쪼갬(ep0 → 폴드 2·3·8 등) → 운영 지문이 '같은 에피소드 이웃의 잔차'를 외우는 누수 가능.
========================  계산 전에 적어 둔 기준 (사전 고정)  ========================
Q1 (Y2h, LOCO=에피소드 빼기): 상한 비교 기준 = 상수 전용(훈련 사슬 잔차 평균) 예측. family 모델이 상수 전용보다
    SSE 가 작아야(이득 > 0) '가족 정보가 있음'. 개선 기록 수도 상수 전용 대비로 다시 셈.
Q2 (Y2a, 전체 잔차 360): family 이득이 '진짜'이려면 아래 다섯 모두:
    (a) DIAG10 폴드에서 family+sg GBM 이 sg 단독 GBM(같은 설정·시드) 대비 SSE 이득 > 0
    (b) 달력 블록 묶음 교차적합(DM11 13일 블록 10개; 에피소드가 거의 한 블록 안에 들어감. DM11 은 묶음 설계에만 쓰는 서술용)에서도 이득 > 0
    (c) (b) 의 이득이 family 행을 섞은 위약 30회의 95백분위보다 큼
    (d) 1차 학습 → 2차(46기록) 시험에서 이득 > 0
    (e) 고EC 기록 SSE 가 sg 단독 대비 나빠지지 않음 (DIAG10·달력 블록 둘 다)
   하나라도 실패하면 'DIAG10 폴드 누수 또는 sg 재보정 몫'으로 판정하고 가족 신호로 세지 않음.
   변형: usable 41개 / 목표 부호화 4개(te_clu_*, proto_sim_hi, knn3_hi_mag) 뺀 37개 — 목표 부호화는 다른 폴드 정답을 담아 2차 누수 가능.
예측(솔직히): (a) 는 양수일 수 있으나 (b)(c)(d) 중 하나 이상에서 무너질 것으로 예상 (6.794 CC1 잔차 보정기 실패, 6.175 다일 특징 날짜 식별 과적합).
"""
import os, sys, json
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
rng = np.random.default_rng(20261013)

R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec").loc[R.index]
M = pd.read_csv(os.path.join(HERE, "vars_v1_meta.csv")).set_index("variable")
VARS = [v for v in M.index if v != "op_cluster"]
USE = [v for v in VARS if M.at[v, "usable"] == "usable"]
TE = ["te_clu_hirate", "te_clu_mag", "proto_sim_hi", "knn3_hi_mag"]
USE_NOTE = [v for v in USE if v not in TE]
R = R.join(F[VARS])
print(f"variables: all {len(VARS)}, usable {len(USE)}, usable-noTE {len(USE_NOTE)}")

def gbm(s):
    return GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=3, random_state=s)

def impute(X, tr):
    med = np.nanmedian(X[tr], 0); med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(X), X, med)

def cv_pred(X, y, g, kind, seeds=(0, 1, 2), train_mask=None):
    """grouped OOF predictions. kind: 'gbm', 'const', 'lin'. train_mask: if given, single split train=train_mask, test=~train_mask."""
    p = np.full(len(y), np.nan)
    splits = [(~(g == u), g == u) for u in np.unique(g)] if train_mask is None else [(train_mask, ~train_mask)]
    for tr, te in splits:
        if kind == "const":
            p[te] = y[tr].mean(); continue
        Xi = impute(X, tr)
        if kind == "lin":
            A = np.column_stack([np.ones(len(y)), Xi]); b = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]; p[te] = A[te] @ b; continue
        pp = np.zeros(te.sum())
        for s in seeds:
            pp += gbm(s).fit(Xi[tr], y[tr]).predict(Xi[te]) / len(seeds)
        p[te] = pp
    return p

def sse(y, p, m=None):
    m = np.ones(len(y), bool) if m is None else m
    k = m & np.isfinite(p)
    return float(np.sum((y[k] - p[k]) ** 2))

OUT = {}
# ======================= Q1: Y2h (high records, LOCO by episode) =======================
S2 = R[(R.hi10 == 1) & (R.has_sg == 1)].copy()
y2 = S2.resid_mean.values; g2 = S2.ep_id.values.astype(int); sg2 = S2.sg_mean.values[:, None]
p_const = cv_pred(None, y2, g2, "const")
p_lin = cv_pred(sg2, y2, g2, "lin")
sse0 = float(np.sum(y2 ** 2)); sst = float(np.sum((y2 - y2.mean()) ** 2))
q1 = {"n": len(y2), "chains": int(len(np.unique(g2))), "SSE_nocorr": sse0, "SST": sst,
      "const_R2_vs_nocorr": 1 - sse(y2, p_const) / sse0, "const_R2_vs_mean": 1 - sse(y2, p_const) / sst,
      "sg_lin_R2_vs_nocorr": 1 - sse(y2, p_lin) / sse0}
for nm, vs in [("all48", VARS), ("usable41", USE), ("usable_noTE37", USE_NOTE)]:
    for addsg in [False, True]:
        X = S2[vs].values.astype(float)
        if addsg: X = np.column_stack([X, sg2])
        p = cv_pred(X, y2, g2, "gbm")
        key = f"{nm}{'+sg' if addsg else ''}|gbm"
        q1[key] = dict(R2_vs_nocorr=1 - sse(y2, p) / sse0, gain_vs_const=(sse(y2, p_const) - sse(y2, p)) / sst,
                       n_improved_vs_const=int(np.sum(np.abs(y2 - p) < np.abs(y2 - p_const))),
                       n_improved_vs_nocorr=int(np.sum(np.abs(y2 - p) < np.abs(y2))))
print("\n=== Q1 Y2h (31 high records, LOCO episodes) ===")
for k, v in q1.items():
    print(" ", k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()} if isinstance(v, dict) else round(v, 3) if isinstance(v, float) else v)
OUT["Q1_Y2h"] = q1

# ======================= Q2: Y2a (all 360 OOF records) =======================
SA = R[R.has_sg == 1].copy()
yA = SA.resid_mean.values; sgA = SA.sg_mean.values[:, None]; hm = SA.hi10.values == 1; p2m = SA.pass2.values == 1
gD = SA.fold.values.astype(int)
blk = np.floor((SA.dm11_day.values - 1) / 13)
gC = np.where(np.isfinite(blk), blk, -1).astype(int)          # 4 records without DM11 day -> one extra group
sstA = float(np.sum((yA - yA.mean()) ** 2)); sse0A = float(np.sum(yA ** 2))
print(f"\n=== Q2 Y2a (n {len(yA)}, high {hm.sum()}, pass2 {p2m.sum()}) | groups DIAG10 {len(np.unique(gD))}, calendar {len(np.unique(gC))} ===")
q2 = {}
base_cache = {}
for gname, g in [("DIAG10", gD), ("CAL13", gC)]:
    pb = cv_pred(sgA, yA, g, "gbm"); pl = cv_pred(sgA, yA, g, "lin"); pc = cv_pred(None, yA, g, "const")
    base_cache[gname] = pb
    q2[f"{gname}|sg_only_gbm"] = dict(R2_vs_nocorr=1 - sse(yA, pb) / sse0A, hi_SSE_ratio_vs_nocorr=sse(yA, pb, hm) / sse(yA, np.zeros_like(yA), hm))
    q2[f"{gname}|sg_only_lin"] = dict(R2_vs_nocorr=1 - sse(yA, pl) / sse0A)
    q2[f"{gname}|const"] = dict(R2_vs_nocorr=1 - sse(yA, pc) / sse0A)
    for nm, vs in [("all48", VARS), ("usable41", USE), ("usable_noTE37", USE_NOTE)]:
        X = np.column_stack([SA[vs].values.astype(float), sgA])
        pf = cv_pred(X, yA, g, "gbm")
        q2[f"{gname}|{nm}+sg_gbm"] = dict(
            R2_vs_nocorr=1 - sse(yA, pf) / sse0A,
            gain_vs_sgonly=(sse(yA, pb) - sse(yA, pf)) / sstA,
            n_improved_vs_sgonly=int(np.sum(np.abs(yA - pf) < np.abs(yA - pb))),
            hi_SSE_ratio_vs_sgonly=sse(yA, pf, hm) / sse(yA, pb, hm),
            hi_n_improved_vs_sgonly=int(np.sum(np.abs(yA[hm] - pf[hm]) < np.abs(yA[hm] - pb[hm]))),
            pass1_gain_vs_sgonly=(sse(yA, pb, ~p2m) - sse(yA, pf, ~p2m)) / float(np.sum((yA[~p2m] - yA[~p2m].mean()) ** 2)),
            pass2_gain_vs_sgonly=(sse(yA, pb, p2m) - sse(yA, pf, p2m)) / float(np.sum((yA[p2m] - yA[p2m].mean()) ** 2)))
# (c) permutation null on CAL13 for usable41 and noTE37 (seed 0 only, observed recomputed with seed 0)
for nm, vs in [("usable41", USE), ("usable_noTE37", USE_NOTE)]:
    Xf = SA[vs].values.astype(float)
    pb0 = cv_pred(sgA, yA, gC, "gbm", seeds=(0,))
    obs = (sse(yA, pb0) - sse(yA, cv_pred(np.column_stack([Xf, sgA]), yA, gC, "gbm", seeds=(0,)))) / sstA
    null = []
    for k in range(30):
        Xp = Xf[rng.permutation(len(yA))]
        null.append((sse(yA, pb0) - sse(yA, cv_pred(np.column_stack([Xp, sgA]), yA, gC, "gbm", seeds=(0,)))) / sstA)
    q2[f"CAL13|{nm}|perm"] = dict(obs_gain_seed0=obs, null_mean=float(np.mean(null)), null_q95=float(np.percentile(null, 95)),
                                  p=(np.sum(np.array(null) >= obs) + 1) / 31)
# (d) pass1 -> pass2 transfer
tm = ~p2m
pbT = cv_pred(sgA, yA, gD, "gbm", train_mask=tm)
for nm, vs in [("all48", VARS), ("usable41", USE), ("usable_noTE37", USE_NOTE)]:
    pfT = cv_pred(np.column_stack([SA[vs].values.astype(float), sgA]), yA, gD, "gbm", train_mask=tm)
    q2[f"P1toP2|{nm}+sg_gbm"] = dict(sgonly_R2_vs_nocorr_p2=1 - sse(yA, pbT, p2m) / sse(yA, np.zeros_like(yA), p2m),
                                    fam_R2_vs_nocorr_p2=1 - sse(yA, pfT, p2m) / sse(yA, np.zeros_like(yA), p2m),
                                    gain_vs_sgonly_p2=(sse(yA, pbT, p2m) - sse(yA, pfT, p2m)) / float(np.sum((yA[p2m] - yA[p2m].mean()) ** 2)),
                                    n_improved_p2=int(np.sum(np.abs(yA[p2m] - pfT[p2m]) < np.abs(yA[p2m] - pbT[p2m]))), n_p2=int(p2m.sum()),
                                    hi_p2_SSE_ratio=sse(yA, pfT, p2m & hm) / sse(yA, pbT, p2m & hm))
for k, v in q2.items():
    print(" ", k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})

# verdict per pre-fixed rule
verd = {}
for nm in ["usable41", "usable_noTE37"]:
    a = q2[f"DIAG10|{nm}+sg_gbm"]["gain_vs_sgonly"] > 0
    b = q2[f"CAL13|{nm}+sg_gbm"]["gain_vs_sgonly"] > 0
    c = q2[f"CAL13|{nm}|perm"]["obs_gain_seed0"] > q2[f"CAL13|{nm}|perm"]["null_q95"]
    d = q2[f"P1toP2|{nm}+sg_gbm"]["gain_vs_sgonly_p2"] > 0
    e = (q2[f"DIAG10|{nm}+sg_gbm"]["hi_SSE_ratio_vs_sgonly"] <= 1) and (q2[f"CAL13|{nm}+sg_gbm"]["hi_SSE_ratio_vs_sgonly"] <= 1)
    verd[nm] = dict(a=bool(a), b=bool(b), c=bool(c), d=bool(d), e=bool(e), real=bool(a and b and c and d and e))
print("\nQ2 verdict (pre-fixed a..e):", verd)
OUT["Q2_Y2a"] = q2; OUT["Q2_verdict"] = verd

# importance (descriptive): fit on all 360 with usable41+sg
X = impute(np.column_stack([SA[USE].values.astype(float), sgA]), np.ones(len(yA), bool))
imp = np.zeros(X.shape[1])
for s in (0, 1, 2):
    imp += gbm(s).fit(X, yA).feature_importances_ / 3
imp = pd.Series(imp, index=USE + ["sg_mean"]).sort_values(ascending=False)
print("\nGBM importance (usable41+sg, all 360, descriptive):\n", imp.head(12).round(3).to_string())
OUT["importance_top12"] = imp.head(12).round(4).to_dict()
json.dump(OUT, open(os.path.join(HERE, "test_v5_baselines.json"), "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
print("saved test_v5_baselines.json")
