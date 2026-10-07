# -*- coding: utf-8 -*-
"""[v2: v1이 입력 요약 merge 열 이름 충돌(in_temp)로 중단 → 기존 열 제거 후 merge, 사슬 결과 캐시. 정의·평가 동일]
고EC 날 '크기'를 가르는 정보원 지도 (진단 전용, 채택 판정 아님) — 2026-10-06 연구실 클로드

질문: 진짜 고EC 날(하루 평균 정답 ≥ 1) 안에서, 예측이 못 가르는 크기 차이를 어떤 정보가 아는가?
대상: r = 실제 하루 평균 − 예측 하루 평균 (예측 = 계절 R3 + DP1 시드 평균, DIAG10 OOF; ec_err01 날 표)
평가(고정): 31일 하루씩 빼기(LOO)로
  B0  보정 없음
  B1  상수 이동만 (r 의 LOO 평균) ← 공통 과소(48%)만 고침
  Sx  r ~ a + b·(x − p) 를 30일로 적합해 빠진 날 예측 ← '크기 구별' 정보
  핵심 지표 = B1 대비 RMSE 변화(크기 구별 몫), 순열 p(정보원 x 를 31일 안에서 섞어 2000회)
정보원(각각 따로):
  A1  SG2 서명 kNN 1순위 이웃의 하루 평균 정답(23시, 폴드 밖 학습 정답만; hk0_rows) — 합법
  A12 1·2순위 이웃 평균 — 합법
  PR  같은 온실 기록 순서상 바로 앞 정답 기록의 하루 평균 — 합법(이전 기록 정답)
  PRD 같은 '동'(st_dong_assign, 입력 분류) 이전 정답 기록 3개 최소 — 합법(HC5 재현)
  CHo CH2 사슬의 앞 기록 23시 EC (사슬이 '이 날 0시 정답'을 비용에 씀 → 오라클 상한, 불법)
  CHi 같은 사슬을 EC 항 없이(입력만) 만든 앞 기록의 23시 EC — 합법(정답 쪽은 앞 기록 것만)
  IN  하루 입력 요약 22개 중 r 과 |ρ| 최대 1개 (선택 자체를 LOO 안에서 다시 함)
두 집합: 진짜 고EC 31일(정답으로 선택, 진단) / 예측 ≥ .9 40일(현실적 집합, 일반 날 10 섞임)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

R = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
MYLOCAL = os.path.join(R, "..", "local")
rng = np.random.default_rng(20261006)

day = pd.read_csv(os.path.join(R, "..", "results", "ec_err01_days_DIAG10_v1.csv"))
day["r"] = day.y - day.p

# ---------- A1/A12: SG2 이웃 (23시 행) ----------
hk = pd.read_csv(os.path.join(H, "hk0_rows_v1.csv"), usecols=["farm", "day", "hour", "a1", "a2"])
hk = hk[hk.hour == 23].drop(columns="hour")
day = day.merge(hk, on=["farm", "day"], how="left")
day["A1"] = day.a1; day["A12"] = day[["a1", "a2"]].mean(axis=1)

# ---------- 정답 기록 ----------
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
for T in (X, Y):
    T["farm"], T["day"], T["hour"] = T.row_id.str[:3], T.row_id.str[4:7].astype(int), T.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]; Y = Y[Y.farm.isin(["F13", "F47"])]
X = X.merge(Y[["row_id", "sub_ec"]], on="row_id", how="left")
dm = Y.groupby(["farm", "day"]).sub_ec.mean()
lab = set(dm.index)

# PR: 기록 순서상 직전 정답 기록 / PRD: 같은 동 직전 정답 3개 최소
dg = pd.read_csv(os.path.join(H, "st_dong_assign_v1.csv")).set_index(["farm", "day"]).dong
def prev_lab(f, d):
    c = [e for (ff, e) in lab if ff == f and e < d]
    return dm[(f, max(c))] if c else np.nan
def prev_dong3(f, d):
    c = sorted([e for (ff, e) in lab if ff == f and e < d and dg.get((f, e)) == dg.get((f, d))])[-3:]
    return min(dm[(f, e)] for e in c) if c else np.nan
day["PR"] = [prev_lab(f, d) for f, d in zip(day.farm, day.day)]
day["PRD"] = [prev_dong3(f, d) for f, d in zip(day.farm, day.day)]

# ---------- CH2 사슬 재구성 (원 스크립트와 같은 비용; EC 항 있음/없음 두 판) ----------
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; WT = {"out_temp": 1, "out_hum": 1, "out_wspd": .3, "out_rad": 1}
IN = {"in_temp": 1.0, "in_hum": 5.0, "in_co2": 40.0}; ACT = ["act_heating", "act_thermal", "act_circfan", "act_vent"]
C0 = 12.0
XL = X[X.sub_ec.notna()]
P = XL.pivot_table(index=["farm", "day"], columns="hour", values=W + list(IN) + ACT + ["sub_ec"])
SC = {(f, v): np.nanstd(XL[XL.farm == f].sort_values(["day", "hour"]).groupby("day")[v].diff()) for f in ("F13", "F47") for v in W}

def jump(f, a, b, v):
    A23, A22, B0, B1 = (P.loc[(f, a), (v, 23)], P.loc[(f, a), (v, 22)], P.loc[(f, b), (v, 0)], P.loc[(f, b), (v, 1)])
    return (B0 - A23) - .5 * ((A23 - A22) + (B1 - B0))

def chains(use_ec):
    pred = {}
    for f in ("F13", "F47"):
        D = sorted(P.loc[f].index); n = len(D); M = np.full((n, n), 1e6)
        for i, a in enumerate(D):
            for j, b in enumerate(D):
                if a == b:
                    continue
                c = sum(WT[v] * ((jump(f, a, b, v) / SC[(f, v)]) ** 2) for v in W)
                c += (sum(((jump(f, a, b, v) / s) ** 2) for v, s in IN.items())
                      + sum(abs(P.loc[(f, b), (v, 0)] - P.loc[(f, a), (v, 23)]) / 50 for v in ACT)) / 4
                if use_ec:
                    c += (jump(f, a, b, "sub_ec") / .02) ** 2
                M[i, j] = c if np.isfinite(c) else 1e6
        big = np.full((2 * n, 2 * n), 1e6); big[:n, :n] = M
        big[:n, n:] = np.where(np.eye(n) == 1, C0, 1e6); big[n:, :n] = np.where(np.eye(n) == 1, C0, 1e6); big[n:, n:] = 0
        r_, c_ = linear_sum_assignment(big)
        for i, j in zip(r_, c_):
            if i < n and j < n:
                pred[(f, D[j])] = D[i]
    return pred

os.makedirs(MYLOCAL, exist_ok=True)
import pickle
for tag, use in (("CHo", True), ("CHi", False)):
    ck = os.path.join(MYLOCAL, "ec_err03_chain_%s.pkl" % tag)
    if os.path.exists(ck):
        pr = pickle.load(open(ck, "rb"))
    else:
        pr = chains(use); pickle.dump(pr, open(ck, "wb"))
    day[tag] = [P.loc[(f, pr[(f, d)]), ("sub_ec", 23)] if (f, d) in pr else np.nan for f, d in zip(day.farm, day.day)]
    day[tag + "_from"] = [pr.get((f, d), np.nan) for f, d in zip(day.farm, day.day)]

# ---------- 입력 요약 22개 ----------
g = X.groupby(["farm", "day"])
night = X[X.hour <= 5].groupby(["farm", "day"]); noon = X[(X.hour >= 10) & (X.hour <= 15)].groupby(["farm", "day"])
F = pd.DataFrame({
    "in_temp": g.in_temp.mean(), "in_temp_rng": g.in_temp.max() - g.in_temp.min(), "out_temp": g.out_temp.mean(),
    "dT": g.in_temp.mean() - g.out_temp.mean(), "in_hum": g.in_hum.mean(), "in_hum_night": night.in_hum.mean(),
    "in_co2": g.in_co2.mean(), "in_co2_noon": noon.in_co2.mean(), "out_rad": g.out_rad.mean(), "in_rad": g.in_rad.mean(),
    "vent": g.act_vent.mean(), "vent0": g.act_vent.apply(lambda s: (s == 0).mean()), "heat": g.act_heating.mean(),
    "thermal": g.act_thermal.mean(), "shade": g.act_shade.mean(), "circfan": g.act_circfan.mean(),
    "co2sup": g.act_co2.mean(), "fog": g.act_fog.mean(), "side": g.act_side.mean(), "valve": g.act_valve.mean(),
    "wspd": g.out_wspd.mean(), "out_hum": g.out_hum.mean()}).reset_index()
INF = [c for c in F.columns if c not in ("farm", "day")]
day = day.drop(columns=[c for c in INF if c in day.columns]).merge(F, on=["farm", "day"], how="left")


def loo(S, xcol=None, feats=None):
    """하루씩 빼기 예측 오차(하루 수준 잔차 기준). xcol: 정보원 1개, feats: LOO 안에서 고르는 입력 후보."""
    r = S.r.values; p = S.p.values; n = len(S); e = np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        if xcol is None and feats is None:
            e[i] = r[i] - r[tr].mean(); continue
        if feats is not None:
            best = max(feats, key=lambda c: abs(spearmanr(S[c].values[tr], r[tr], nan_policy="omit")[0]) if S[c].values[tr].std() > 0 else 0)
            z = S[best].values
        else:
            z = S[xcol].values - p
        A = np.c_[np.ones(tr.sum()), z[tr]]
        ok = np.isfinite(A).all(1)
        b = np.linalg.lstsq(A[ok], r[tr][ok], rcond=None)[0]
        e[i] = r[i] - (b[0] + b[1] * z[i]) if np.isfinite(z[i]) else r[i] - r[tr].mean()
    return np.sqrt(np.mean(e ** 2))


def report(name, S):
    print("\n=== %s: %d일 (진짜 고EC %d) ===" % (name, len(S), (S.y >= 1).sum()))
    b0 = np.sqrt(np.mean(S.r ** 2)); b1 = loo(S)
    print("B0 보정 없음 %.4f | B1 상수 이동 %.4f (%+.1f%%)  ← 공통 과소만" % (b0, b1, 100 * (b1 / b0 - 1)))
    print("정보원   있음  ρ(x,y)  ρ(x−p,r)  LOO RMSE  B1대비   순열p(크기정보)")
    for c in ("A1", "A12", "PR", "PRD", "CHi", "CHo"):
        m = S[c].notna()
        Sm = S[m].reset_index(drop=True)
        if len(Sm) < 8:
            continue
        b1m = loo(Sm); v = loo(Sm, c)
        null = []
        for _ in range(500):
            Q = Sm.copy(); Q[c] = rng.permutation(Q[c].values)
            null.append(loo(Q, c))
        pv = (np.sum(np.array(null) <= v) + 1) / (len(null) + 1)
        print("%-6s %3d/%d  %+.2f    %+.2f     %.4f   %+6.1f%%   %.3f" % (
            c, m.sum(), len(S), spearmanr(Sm[c], Sm.y)[0], spearmanr(Sm[c] - Sm.p, Sm.r)[0], v, 100 * (v / b1m - 1), pv))
    v = loo(S, feats=INF)
    null = []
    for _ in range(200):
        Q = S.copy(); Q["r"] = rng.permutation(Q.r.values)
        null.append(loo(Q, feats=INF) / loo(Q))
    pv = (np.sum(np.array(null) <= v / b1) + 1) / (len(null) + 1)
    rho = {c: spearmanr(S[c], S.r, nan_policy="omit")[0] for c in INF}
    top = sorted(rho, key=lambda c: -abs(rho[c]))[:4]
    print("IN     입력22중 LOO선택 %.4f  %+6.1f%%  순열p %.3f | 전체표본 상위: %s" % (
        v, 100 * (v / b1 - 1), pv, ", ".join("%s %+.2f" % (c, rho[c]) for c in top)))


if __name__ == "__main__":
    hi = day[day.y >= 1].reset_index(drop=True)
    ph = day[day.p >= 0.9].reset_index(drop=True)
    report("진짜 고EC 31일", hi)
    report("예측 ≥ .9 40일", ph)
    # 사슬 연결이 '진짜 앞날'인지: 오라클판과 입력판 일치
    m = hi.CHo_from.notna() & hi.CHi_from.notna()
    print("\n사슬 앞 기록 일치(입력판 = 오라클판): 고EC %d일 중 %d일" % (m.sum(), (hi.CHo_from[m] == hi.CHi_from[m]).sum()))
    cols = ["farm", "day", "late", "role", "y", "p", "r", "A1", "A12", "PR", "PRD", "CHo", "CHi", "CHo_from", "CHi_from"]
    print(hi.sort_values("y", ascending=False)[cols].round(2).to_string(index=False))
    day.to_csv(os.path.join(R, "..", "results", "ec_err03_sources_days_v2.csv"), index=False, encoding="utf-8-sig")
