# -*- coding: utf-8 -*-
"""LINK3 — 정답 없는 기록(평가형)끼리 사슬로 잇고, 사슬 앞 정답 기록의 23시 EC를 전달 (진단, 실행 전 기준 고정)
2026-10-06 연구실 클로드

배경: LINK2(평가 조건)에서 2차 대상의 39%는 진짜 앞날이 정답 없는 기록이라 어떤 정답 기록을 골라도 틀림.
이번: 정답 없는 기록도 앞날 후보에 넣고, 고른 앞날이 정답 없는 기록이면 그 기록의 '추정값'을 이어받는다.

설정 (DIAG10 폴드 k 마다):
  L = 정답 있는 기록 중 폴드 k 밖 (학습 자료: 입력·정답 모두 참조 가능)
  U = 폴드 k 의 2차 기록(정답 가림) + 실제 평가 기록 60개(test_X)  ← 정답 없는 기록
  폴드 k 의 1차 기록은 가린 채 후보에서도 뺌(실제 평가엔 정답 없는 1차 기록이 없음)
  규정 맞춤: U 기록 t 의 앞날 후보 = L 전체 ∪ {U 중 기록 번호가 t 보다 작은 것}(2차는 달력 순서 → 이전 입력만)
  판별기: LINK1과 같은 입력 특징·같은 LightGBM, 폴드 k 마다 L 안의 쌍(정답 연결 = CH2 오라클)으로 학습
  추정: U 를 기록 번호 순서로 처리. top-1 앞날 q 의 확률 ≥ .5 이고 q 의 23시 값이 있으면
        level(t) = v23(q),  v23(q) = q 가 L 이면 정답 23시 EC, U 이면 level(q) + (모델23 − 모델하루평균)(q)
        아니면 level(t) = 모델 하루 평균(전달 없음 → t 를 앞날로 고른 다음 기록도 모델값 전달)
  시간별 예측: pred_h = 모델_h − 모델하루평균 + level
  모델: 폴드 k 기록 = DIAG10 dp(계절 R3+DP1) 시드평균, 실제 평가 기록 = submission_14 EC 열(모양·전달용)
변형(k=2, 고정): V1 보호 없음 / V2 보호: |level − 모델하루평균| > .30 이면 모델값(SG2와 같은 폭)
[고정 기준] 2차 46일 시간 행 RMSE가 모델보다 낮고, 두 온실 모두 낮고, 온실×5기록 묶음 부트스트랩 2000
            P(worse) < .0125 (.025/2). 진단 기준이며 제출 채택 판정 아님.
함께 보고: 앞날로 U 를 고른 비율, 정답 앞날이 L 에 있을 때 적중률, 일반/고EC 나눔.
"""
import os, sys, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import linear_sum_assignment

R = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
print("source sha256", hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16])

TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
TR = TR.merge(Y[["row_id", "sub_ec"]], on="row_id", how="left"); TE["sub_ec"] = np.nan
TR["is_test"] = False; TE["is_test"] = True
X = pd.concat([TR, TE], ignore_index=True)
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; IN = ["in_temp", "in_hum", "in_co2"]
ACT = ["act_vent", "act_side", "act_shade", "act_thermal", "act_valve", "act_heating", "act_circfan", "act_co2", "act_fog"]

# 모델 예측 (시간별)
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); DP = DP[DP.validator == "DIAG10"].copy()
DP["m"] = DP[["dp_7", "dp_101", "dp_2024"]].mean(axis=1)
fold = DP.groupby(["farm", "day"]).validation_fold.first()
S14 = pd.read_csv(os.path.join(env.SUBMIT, u"09회차_2026-10-06(팀)", "submission_14.csv"))
S14["m"] = S14.sub_ec
MOD = pd.concat([DP[["row_id", "m"]], S14[["row_id", "m"]]]).set_index("row_id").m
X["m"] = X.row_id.map(MOD)

rows = []; truth = {}; REC = {}
for f in ("F13", "F47"):
    Z = X[X.farm == f]
    D = np.array(sorted(Z.day.unique())); n = len(D)
    piv = {v: Z.pivot(index="day", columns="hour", values=v).reindex(D).values for v in W + IN + ACT + ["sub_ec", "m"]}
    isT = Z.groupby("day").is_test.first().reindex(D).values
    REC[f] = dict(D=D, piv=piv, isT=isT)
    lab = ~isT
    def sd(v):
        return np.nanstd(np.diff(piv[v][lab], axis=1))
    sc = {v: sd(v) for v in W + IN}
    def J(v):
        A23, A22, B0, B1 = piv[v][:, 23], piv[v][:, 22], piv[v][:, 0], piv[v][:, 1]
        return (B0[None, :] - A23[:, None]) - .5 * ((A23 - A22)[:, None] + (B1 - B0)[None, :])
    jw = {v: J(v) / sc[v] for v in W}; ji = {v: J(v) / sc[v] for v in IN}
    datec = sum(wt * jw[v] ** 2 for v, wt in zip(W, (1, 1, .3, 1)))
    act_jump = {v: np.abs(piv[v][:, 0][None, :] - piv[v][:, 23][:, None]) for v in ACT}
    hand = datec + sum(ji[v] ** 2 for v in IN) / 4 + sum(act_jump[v] for v in ("act_heating", "act_thermal", "act_circfan", "act_vent")) / 50 / 4
    # 정답 연결: 정답 있는 기록끼리 CH2 (LINK1과 동일)
    li = np.where(lab)[0]; nl = len(li)
    jec = J("sub_ec")[np.ix_(li, li)]
    M = hand[np.ix_(li, li)] + (jec / .02) ** 2
    M = np.where(np.isfinite(M), M, 1e6); np.fill_diagonal(M, 1e6)
    big = np.full((2 * nl, 2 * nl), 1e6); big[:nl, :nl] = M
    big[:nl, nl:] = np.where(np.eye(nl) == 1, 12.0, 1e6); big[nl:, :nl] = np.where(np.eye(nl) == 1, 12.0, 1e6); big[nl:, nl:] = 0
    r_, c_ = linear_sum_assignment(big)
    for i, j in zip(r_, c_):
        if i < nl and j < nl and abs(jec[i, j]) < .05:
            truth[(f, D[li[j]])] = D[li[i]]
    with np.errstate(all="ignore"):
        dmean = {v: np.nanmean(piv[v], axis=1) for v in ACT}
        nightT = np.nanmean(piv["in_temp"][:, :6], axis=1)
    for j, b in enumerate(D):
        for i, a in enumerate(D):
            if i == j:
                continue
            row = dict(farm=f, a=a, b=b, ta=isT[i], tb=isT[j], hand=hand[i, j], datec=datec[i, j], gap=b - a,
                       late_a=int(a >= 179), late_b=int(b >= 179))
            for v in W:
                row["jw_" + v] = abs(jw[v][i, j])
            for v in IN:
                row["ji_" + v] = abs(ji[v][i, j])
            for v in ACT:
                row["ja_" + v] = act_jump[v][i, j]; row["sd_" + v] = abs(dmean[v][i] - dmean[v][j])
            row["sd_nightT"] = abs(nightT[i] - nightT[j])
            rows.append(row)
P = pd.DataFrame(rows)
P["y"] = [int(truth.get((f, b)) == a) for f, a, b in zip(P.farm, P.a, P.b)]
FEAT = [c for c in P.columns if c.startswith(("jw_", "ji_", "ja_", "sd_"))] + ["datec", "hand", "gap", "late_a", "late_b"]
P["fa"] = [fold.get((f, a), -1) for f, a in zip(P.farm, P.a)]
P["fb"] = [fold.get((f, b), -1) for f, b in zip(P.farm, P.b)]
print("pairs %d, 정답 연결 %d, 평가 기록 %d" % (len(P), len(truth), int(X.groupby(["farm", "day"]).is_test.first().sum())))

def v23_of(f, d, lev, isU):
    r = REC[f]; i = int(np.where(r["D"] == d)[0][0])
    if not isU:
        return r["piv"]["sub_ec"][i, 23]
    if d not in lev or lev[d] is None:
        return None
    return lev[d] + (r["piv"]["m"][i, 23] - np.nanmean(r["piv"]["m"][i]))

out = []
for k in sorted(DP.validation_fold.unique()):
    tgt = [(f, d) for (f, d), kk in fold.items() if kk == k and d >= 179]
    if not tgt:
        continue
    trm = (P.fa != k) & (P.fb != k) & ~P.ta & ~P.tb
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=15, min_child_samples=20,
                             subsample=.8, subsample_freq=1, colsample_bytree=.8, scale_pos_weight=20, verbose=-1, random_state=7)
    mdl.fit(P.loc[trm, FEAT], P.loc[trm, "y"])
    for f in ("F13", "F47"):
        r = REC[f]
        hidden = {d for d in r["D"] if fold.get((f, d), -9) == k}
        Uset = {d for d in r["D"] if (d in hidden and d >= 179) or r["isT"][np.where(r["D"] == d)[0][0]]}
        Lset = {d for d in r["D"] if (d not in hidden) and not r["isT"][np.where(r["D"] == d)[0][0]]}
        Pf = P[(P.farm == f) & P.b.isin(Uset)].copy()
        Pf = Pf[Pf.a.isin(Lset) | (Pf.a.isin(Uset) & (Pf.a < Pf.b))]
        Pf["prob"] = mdl.predict_proba(Pf[FEAT])[:, 1]
        for guard, tag in ((None, "V1"), (.30, "V2")):
            lev = {}
            for d in sorted(Uset):
                g = Pf[Pf.b == d].sort_values("prob", ascending=False)
                i = int(np.where(r["D"] == d)[0][0]); mday = np.nanmean(r["piv"]["m"][i])
                q = g.a.iloc[0]; pq = g.prob.iloc[0]; qU = q in Uset
                v = v23_of(f, q, lev, qU) if pq >= .5 else None
                if v is not None and guard is not None and abs(v - mday) > guard:
                    v = None
                lev[d] = v if v is not None else mday
                if (f, d) in tgt:
                    t = truth.get((f, d))
                    out.append(dict(var=tag, farm=f, day=d, fold=k, q=q, p=pq, qU=qU, used=v is not None, level=lev[d], mday=mday,
                                    t_inL=(t is not None) and (t in Lset), hit=(t is not None) and q == t))
            # 전달 없는 경우 다음 기록에 '모델값'이 전달되지 않도록: 모델값으로 채운 날은 lev 에서 None 처리
            # (위 v23_of 는 lev 값을 쓰므로, 모델값 대체 날도 모델 23시 값으로 이어짐 — 고정 규칙대로 둠)
O = pd.DataFrame(out)

def rm(x):
    return np.sqrt(np.mean(np.square(x)))
H2 = DP[DP.day >= 179].copy()
H2["mday"] = H2.groupby(["farm", "day"]).m.transform("mean"); H2["ydm"] = H2.groupby(["farm", "day"]).sub_ec.transform("mean")
for tag in ("V1", "V2"):
    o = O[O["var"] == tag]
    Q = H2.merge(o[["farm", "day", "level"]], on=["farm", "day"])
    Q["pred"] = Q.m - Q.mday + Q.level
    print("\n=== %s (%s) 대상 %d일 ===" % (tag, "보호 없음" if tag == "V1" else "보호 .30", len(o)))
    print("  전달 사용 %d일 (앞날로 정답 없는 기록을 고른 날 %d, 정답 기록 %d) | 정답 앞날이 L 에 있는 %d일 중 적중 %.2f" % (
        o.used.sum(), (o.used & o.qU).sum(), (o.used & ~o.qU).sum(), o.t_inL.sum(), o[o.t_inL].hit.mean() if o.t_inL.any() else np.nan))
    for nm, mm in (("2차 전체", Q.day > 0), ("2차 일반", Q.ydm < 1), ("2차 고EC", Q.ydm >= 1), ("F13", Q.farm == "F13"), ("F47", Q.farm == "F47")):
        q = Q[mm]
        print("  %-7s 모델 %.4f | 사슬 %.4f (%+.1f%%)" % (nm, rm(q.m - q.sub_ec), rm(q.pred - q.sub_ec), 100 * (rm(q.pred - q.sub_ec) / rm(q.m - q.sub_ec) - 1)))
    Q["blk"] = Q.farm + "_" + (Q.day // 5).astype(str)
    B = Q.groupby("blk").apply(lambda g: pd.Series({"a": ((g.m - g.sub_ec) ** 2).sum(), "b": ((g.pred - g.sub_ec) ** 2).sum()}))
    idx = np.random.default_rng(1).integers(0, len(B), (2000, len(B)))
    pw = np.mean(B.b.values[idx].sum(1) > B.a.values[idx].sum(1))
    ok = (rm(Q.pred - Q.sub_ec) < rm(Q.m - Q.sub_ec)) and all(
        rm(Q[Q.farm == f].pred - Q[Q.farm == f].sub_ec) < rm(Q[Q.farm == f].m - Q[Q.farm == f].sub_ec) for f in ("F13", "F47"))
    print("  P(worse) %.4f → 고정 기준 %s" % (pw, "통과" if ok and pw < .0125 else "불합격"))
O.to_csv(os.path.join(R, "..", "results", "ec_link03_days_v1.csv"), index=False, encoding="utf-8-sig")
