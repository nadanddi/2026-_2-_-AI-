# -*- coding: utf-8 -*-
"""[v2: 전날 후보를 두 온실 ID(F13·F47) 전체로 확장 — 전날 날짜 묶음과 외기 쌍둥이인 다른 온실 1차 정답 기록 포함; 진짜 앞날 판정도 두 온실 전체 정답 기록 대상. 입력 자정연속 표준화는 온실 공통]
PD1 — '날짜 확정 2차 기록의 전날 후보' 전제 검산 (진단 전용, 정답은 판정에만) — 2026-10-07 연구실 클로드
2차 정답 기록 b 가 같은 온실 1차 정답 기록과 외기 24h 완전 쌍둥이(z-RMSE ≤ .05)면 b 의 달력 날짜 = 그 1차 기록의 날짜.
1차는 달력 순서이므로 전날 기록 = 쌍둥이 1차 기록이 속한 날짜 묶음(같은 날짜 1~2기록) 바로 앞 날짜 묶음의 기록들.
질문:
 Q1 날짜 확정 비율, 전날 후보 수
 Q2 진짜 앞날(정답 자정 연속 |EC_b(0) − EC_a(23)| < .05 인 같은 온실 정답 기록 a)이 전날 후보 안에 있는 비율
     비교: 전체 정답 기록 중 자정 연속 기록 수(우연 기준)
 Q3 전날 후보의 23시 EC 를 하루 수준으로 썼을 때(오라클 선택 / 후보 평균 / 입력 자정 연속 최선) 일반 날 하루 RMSE vs 모델
     입력 자정 연속 = 실내 온습도·CO2 23→0시 점프 표준화 제곱합(학습 자료 + b 의 0시 입력만 → 규정상 가능한 선택)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd

R = os.path.dirname(os.path.abspath(__file__))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
for T in (X, Y):
    T["farm"], T["day"], T["hour"] = T.row_id.str[:3], T.row_id.str[4:7].astype(int), T.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]; Y = Y[Y.farm.isin(["F13", "F47"])]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; IN = ["in_temp", "in_hum", "in_co2"]
WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W); WZ = (WV - WV.mean()) / WV.std()
EC = Y.pivot_table(index=["farm", "day"], columns="hour", values="sub_ec")
INV = {v: X.pivot_table(index=["farm", "day"], columns="hour", values=v) for v in IN}
sc = {v: np.nanstd(np.diff(INV[v].values, axis=1)) for v in IN}
ym = EC.mean(axis=1)
mod = pd.read_csv(os.path.join(R, "..", "results", "ec_err01_days_DIAG10_v1.csv")).set_index(["farm", "day"]).p
out = []
for f in ("F13", "F47"):
    D1 = sorted(d for ff, d in EC.index if ff == f and d < 179)
    # 1차 날짜 묶음: 연속 기록이 쌍둥이면 같은 날짜
    grp, gi = {}, 0
    for k, d in enumerate(D1):
        if k and np.sqrt(np.nanmean((WZ.loc[(f, D1[k - 1])].values - WZ.loc[(f, d)].values) ** 2)) > .05:
            gi += 1
        grp[d] = gi
    G = {}
    for d, g in grp.items():
        G.setdefault(g, []).append(d)
    A1 = WZ.loc[[(f, d) for d in D1]].values
    for b in sorted(d for ff, d in EC.index if ff == f and d >= 179):
        dist = np.sqrt(np.nanmean((A1 - WZ.loc[(f, b)].values) ** 2, axis=1))
        tw = [D1[i] for i in np.where(dist <= .05)[0]]
        allc = [(ff, a) for ff, a in EC.index if (ff, a) != (f, b)]
        truth = [k for k in allc if abs(EC.loc[(f, b), 0] - EC.loc[k, 23]) < .05]
        row = dict(farm=f, b=b, dated=bool(tw), n_true_any=len(truth), y=ym[(f, b)], p=mod.get((f, b), np.nan))
        if tw:
            g = grp[tw[0]]
            prevd = G.get(g - 1, [])
            cand = [(f, a) for a in prevd]
            if prevd:
                ref = WZ.loc[(f, prevd[0])].values
                o = "F47" if f == "F13" else "F13"
                for a in sorted(d for ff, d in EC.index if ff == o and d < 179):
                    if np.sqrt(np.nanmean((WZ.loc[(o, a)].values - ref) ** 2)) <= .05:
                        cand.append((o, a))
            row.update(n_cand=len(cand), n_other=sum(1 for k in cand if k[0] != f), true_in=any(k in truth for k in cand),
                       true_in_own=any(k in truth for k in cand if k[0] == f), true_in_other=any(k in truth for k in cand if k[0] != f))
            if cand:
                e23 = np.array([EC.loc[k, 23] for k in cand])
                row["orac"] = e23[np.argmin(np.abs(e23 - EC.loc[(f, b), 0]))]
                row["orac_other"] = cand[int(np.argmin(np.abs(e23 - EC.loc[(f, b), 0])))][0] != f
                row["mean"] = e23.mean()
                cost = [sum(((INV[v].loc[(f, b), 0] - INV[v].loc[k, 23]) / sc[v]) ** 2 for v in IN) for k in cand]
                row["inp"] = e23[int(np.nanargmin(cost))]
                row["spread"] = e23.max() - e23.min()
        out.append(row)
O = pd.DataFrame(out)
print("2차 정답 기록 %d, 날짜 확정(1차 쌍둥이) %d" % (len(O), O.dated.sum()))
Q = O[O.dated]
print("다른 온실 후보 수 분포 %s | 앞날이 같은 온실 후보에 %d, 다른 온실 후보에 %d, 오라클이 다른 온실 %d" % (Q.n_other.value_counts().to_dict(), Q.true_in_own.sum(), Q.true_in_other.sum(), Q.orac_other.sum()))
print("전날 후보 수 분포 %s | 진짜 앞날이 후보 안 %d/%d (%.0f%%) | (참고: 정답 기록 전체 중 자정 연속 기록 수 중앙 %d)" % (
    Q.n_cand.value_counts().to_dict(), Q.true_in.sum(), len(Q), 100 * Q.true_in.mean(), O.n_true_any.median()))
print("후보 23시 EC 범위(후보 2개일 때) 중앙 %.3f" % Q[Q.n_cand >= 2].spread.median())
rm = lambda e: np.sqrt(np.nanmean(np.square(e)))
for nm, m in (("전체", Q.y > -1), ("일반(y<1)", Q.y < 1), ("고EC", Q.y >= 1)):
    q = Q[m & Q.p.notna()]
    print("  %-9s n=%2d | 모델 %.3f | 후보 오라클 %.3f | 후보 평균 %.3f | 입력 자정연속 최선 %.3f" % (
        nm, len(q), rm(q.p - q.y), rm(q.orac - q.y), rm(q["mean"] - q.y), rm(q.inp - q.y)))
print("  입력 선택 = 오라클 선택 비율 %.2f (후보 2개 이상)" % (Q[Q.n_cand >= 2].eval("inp == orac").mean()))
O.to_csv(os.path.join(R, "..", "results", "ec_pd1_dated_pred_v2.csv"), index=False, encoding="utf-8-sig")
