# -*- coding: utf-8 -*-
"""FC3 — FC1 비평의 '시각 인과판 이득 소멸'(6.427) 수치가 이상(h=15 .215 > h=8 .078, 하루 전체 .058)해서 직접 재검산 (실행 전 고정) — 2026-10-07 연구실 클로드
규칙: h 시 행마다 대상 b 와 전날 묶음 후보 모두 0..h 시 지문(구동기 7 누적평균·0 비율, 실내 3 누적평균)을 같은 창으로,
      편차 = 지문 − 같은 날짜 묶음 '참조 정답 기록'(가림·평가·자기 제외) 같은 창 평균. 표준화 = 참조 정답 기록(창별).
      후보 = 전날 묶음(1차 순서)의 참조 정답 기록. 선택 = 편차 거리 최소. v = EC23(선택).
      가림 = DIAG10 폴드 기록 + 실제 평가 60. 대상 = 2차 정답 날짜 확정(C1·C2).
점검 1: 창별 하루 단위 오차(h = 5, 8, 10, 12, 15, 18, 20, 23) — h=23 이 FC0(.058~.066) 근처여야 함.
점검 2(주): 시간 행 보정 pred_h = sg_h − pm_h + v_h (h ≥ 5; v_h = h 시 창으로 고른 기록의 EC23), 그 밖 sg_h.
           SG2 DIAG10 시드 3 각각과 비교 — 2차 결정 날 시간 RMSE, 일반 날, 고EC 날. 위약(2묶음 전) 함께.
[관문(고정)] 점검 2에서 시드 3 모두 결정 날 일반 날 시간 RMSE 가 SG2 와 위약보다 낮을 것 → 계획 단계로. 아니면 종료.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]; INDOOR = ["in_temp", "in_hum", "in_co2"]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TR["t"] = 0; TE["t"] = 1
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])].sort_values(["farm", "day", "hour"])
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
EC = Y[Y.farm.isin(["F13", "F47"])].pivot_table(index=["farm", "day"], columns="hour", values="sub_ec"); ym = EC.mean(axis=1); labset = set(EC.index)
isT = X.groupby(["farm", "day"]).t.first()
FW = {}
for h in range(24):
    Xh = X[X.hour <= h]; g = Xh.groupby(["farm", "day"])
    FW[h] = pd.concat([g[ACTS].mean().add_suffix("_m"), g[ACTS].agg(lambda s: (s == 0).mean()).add_suffix("_z"), g[INDOOR].mean().add_suffix("_d")], axis=1)
WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W); WZ = (WV - WV[isT == 0].mean()) / WV[isT == 0].std()
keys = list(WZ.index); A = WZ.values; Dm = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
par = list(range(len(keys)))
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for i in range(len(keys)):
    for j in np.where(Dm[i] <= .05)[0]:
        par[fd(i)] = fd(j)
grp = pd.Series([fd(i) for i in range(len(keys))], index=pd.MultiIndex.from_tuples(keys))
cls = pd.read_csv(os.path.join(R, "..", "results", "ec_dt0_date_class_v1.csv")).set_index(["farm", "day"]).cls
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); f10 = DP[DP.validator == "DIAG10"].groupby(["farm", "day"]).validation_fold.first()
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); S = S[S.validator == "DIAG10"].sort_values(["farm", "day", "hour"])
def prev_group(b, steps):
    p1 = sorted([k for k in grp.index[grp == grp[b]] if k[1] < 179], key=lambda k: k[1])
    if not p1:
        return None
    f0, d0 = p1[0]; seen = []
    for d in sorted((d for f, d in keys if f == f0 and d < 179 and d < d0), reverse=True):
        gg = grp[(f0, d)]
        if gg != grp[b] and (not seen or seen[-1] != gg):
            seen.append(gg)
            if len(seen) == steps:
                return gg
    return None
pick = {}
targets = [b for b in S.groupby(["farm", "day"]).size().index if cls.get(b) in ("C1", "C2")]
for b in targets:
    ref = labset - ({k for k, kk in f10.items() if kk == f10[b]} | {k for k in keys if isT[k] == 1})
    refl = [k for k in labset if k in ref]
    for h in range(5, 24):
        Fh = FW[h]; mu, sd = Fh.loc[refl].mean(), Fh.loc[refl].std().replace(0, 1); Z = ((Fh - mu) / sd).fillna(0)
        def dev(k):
            mem = [m for m in grp.index[grp == grp[k]] if m in ref and m != k]
            return (Z.loc[k].values - Z.loc[mem].values.mean(axis=0)) if mem else None
        db = dev(b)
        for tag, st in (("", 1), ("pl", 2)):
            pg = prev_group(b, st); v = np.nan
            if db is not None and pg is not None:
                cand = [(k, dev(k)) for k in grp.index[grp == pg] if k in ref]
                cand = [(k, d) for k, d in cand if d is not None]
                if cand:
                    k = min(cand, key=lambda kd: np.sqrt(np.mean((kd[1] - db) ** 2)))[0]; v = EC.loc[k, 23]
            pick[(b, h, tag)] = v
rm = lambda x: np.sqrt(np.nanmean(np.square(x)))
print("점검 1: 창별 하루 단위 오차(정답 일반 날) — 같은 날 SG2 하루평균과 비교")
sgd = S.assign(sg=S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1)).groupby(["farm", "day"]).sg.mean()
for h in (5, 8, 10, 12, 15, 18, 20, 23):
    q = [(ym[b], pick[(b, h, "")], sgd[b]) for b in targets if np.isfinite(pick[(b, h, "")]) and ym[b] < 1]
    q = np.array(q)
    print("  h=%2d n=%2d | 지문 선택 %.3f | SG2 %.3f" % (h, len(q), rm(q[:, 1] - q[:, 0]), rm(q[:, 2] - q[:, 0])))
print("점검 2: 시간 행 보정(h ≥ 5, h 시 창 선택) — 결정 날")
T = S[[(f, d) in set(targets) for f, d in zip(S.farm, S.day)]].copy()
T["v"] = [pick.get(((f, d), h, ""), np.nan) if h >= 5 else np.nan for f, d, h in zip(T.farm, T.day, T.hour)]
T["vp"] = [pick.get(((f, d), h, "pl"), np.nan) if h >= 5 else np.nan for f, d, h in zip(T.farm, T.day, T.hour)]
T["dm"] = T.groupby(["farm", "day"]).sub_ec.transform("mean")
ok = True
for s in (23, 808, 9090):
    sg = T["sg_%d" % s]; pm = sg.groupby([T.farm, T.day]).transform(lambda x: x.expanding().mean())
    pr = np.where(T.v.notna(), sg - pm + T.v, sg); pp = np.where(T.vp.notna(), sg - pm + T.vp, sg)
    for nm, m in (("전체", T.dm > -1), ("일반", T.dm < 1)):
        a, b_, c = rm((sg - T.sub_ec)[m]), rm((pr - T.sub_ec)[m]), rm((pp - T.sub_ec)[m])
        print("  SG2 %4d %-4s | SG2 %.4f | 지문 %.4f (%+.1f%%) | 위약 %.4f" % (s, nm, a, b_, 100 * (b_ / a - 1), c))
        if nm == "일반":
            ok &= (b_ < a) and (b_ < c)
print("[FC3 관문] %s" % ("통과 → 계획 단계로" if ok else "종료"))
