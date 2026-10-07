# -*- coding: utf-8 -*-
"""FC2 — 자정 연속 '운영 상태 편차'로 전날 같은 출처 고르기 (전제 검산, 실행 전 고정) — 2026-10-07 연구실 클로드
FC1 비평(6.427): 하루 전체 지문은 미래 시각 입력 → 시각 인과판에서 이득 소멸. 원리로 변수 고정(사후 선택 금지):
 제어기 상태는 자정을 넘어 이어짐 → b 의 0시 상태와 전날 후보의 23시 상태를 비교. 0시 값이라 모든 시각 행에서 인과.
 편차 = 값 − 같은 날짜 묶음 '참조 정답 기록'(가림·평가 제외, 자기 제외) 같은 시각 평균 (날씨 제거). 표준화 = 참조 정답 기록.
 V1: 구동기 7개 (b 0시 편차 vs 후보 23시 편차)   V2: V1 + 실내 3개
 대상: 2차 정답 기록 중 날짜 확정(C1·C2). 가림 = DIAG10 폴드 기록 + 실제 평가 60. 전날 묶음 = 1차 순서 앞 날짜 묶음(FC0 와 같음).
 후보 = 전날 묶음의 참조 정답 기록만(평가·가림 기록 후보 제외). 같은 묶음의 두 대상이 같은 후보를 고르면 둘 다 보류.
 v = EC23(선택). 위약: 2묶음 전에서 같은 방식. 비교: SG2 DIAG10 시드평균 하루평균(같은 날).
[관문(고정)] V1 또는 V2 가 정답 일반 날에서 SG2 보다 낮고, 위약보다도 낮을 것(둘 다). 아니면 종료.
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
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
EC = Y[Y.farm.isin(["F13", "F47"])].pivot_table(index=["farm", "day"], columns="hour", values="sub_ec"); ym = EC.mean(axis=1); labset = set(EC.index)
isT = X.groupby(["farm", "day"]).t.first()
H0 = X[X.hour == 0].set_index(["farm", "day"])[ACTS + INDOOR]; H23 = X[X.hour == 23].set_index(["farm", "day"])[ACTS + INDOOR]
allv = pd.concat([H0, H23]); mu, sd = allv[[k in labset for k in allv.index]].mean(), allv[[k in labset for k in allv.index]].std().replace(0, 1)
H0s, H23s = (H0 - mu) / sd, (H23 - mu) / sd
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
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); S = S[S.validator == "DIAG10"]
S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1); sgd = S.groupby(["farm", "day"]).sg.mean()
def dev(T, k, ref, cols):
    mem = [m for m in grp.index[grp == grp[k]] if m in ref and m != k and m in T.index]
    if k not in T.index or not mem:
        return None
    v = T.loc[k, cols].values - T.loc[mem, cols].values.mean(axis=0)
    return None if np.isnan(v).any() else v
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
rows = []
for b in sgd.index:
    if cls.get(b) not in ("C1", "C2"):
        continue
    hidden = {k for k, kk in f10.items() if kk == f10[b]} | {k for k in keys if isT[k] == 1}
    ref = labset - hidden
    out = dict(farm=b[0], day=b[1], y=ym[b], sg=sgd[b], grp=grp[b])
    for vn, cols in (("V1", ACTS), ("V2", ACTS + INDOOR)):
        db = dev(H0s, b, ref, cols)
        for tag, steps in (("", 1), ("_pl", 2)):
            pg = prev_group(b, steps); val, pick = np.nan, None
            if db is not None and pg is not None:
                cand = [(k, dev(H23s, k, ref, cols)) for k in grp.index[grp == pg] if k in ref]
                cand = [(k, v) for k, v in cand if v is not None]
                if cand:
                    pick = min(cand, key=lambda kv: np.sqrt(np.mean((kv[1] - db) ** 2)))[0]; val = EC.loc[pick, 23]
            out[vn + tag] = val; out[vn + tag + "_pick"] = str(pick)
    rows.append(out)
D = pd.DataFrame(rows)
for vn in ("V1", "V2"):   # 충돌 보류
    dup = D.groupby(["grp", vn + "_pick"]).size(); bad = {k for k, c in dup.items() if c > 1 and k[1] != "None"}
    D.loc[[(g_, p_) in bad for g_, p_ in zip(D.grp, D[vn + "_pick"])], vn] = np.nan
rm = lambda x: np.sqrt(np.nanmean(np.square(x)))
ok_any = False
for vn in ("V1", "V2"):
    for nm, m in (("전체", D.y > -1), ("정답 일반", D.y < 1)):
        q = D[m & D[vn].notna() & D[vn + "_pl"].notna()]
        s_, v_, p_ = rm(q.sg - q.y), rm(q[vn] - q.y), rm(q[vn + "_pl"] - q.y)
        print("%s %-6s n=%2d | SG2 %.3f | 전날 선택 %.3f | 위약(2묶음 전) %.3f" % (vn, nm, len(q), s_, v_, p_))
        if nm == "정답 일반" and v_ < s_ and v_ < p_:
            ok_any = True
print("결정 수(충돌 보류 후): V1 %d, V2 %d / 대상 %d" % (D.V1.notna().sum(), D.V2.notna().sum(), len(D)))
print("[FC2 관문] %s" % ("통과 → 계획 단계로" if ok_any else "종료"))
D.to_csv(os.path.join(R, "..", "results", "ec_fc2_days_v1.csv"), index=False, encoding="utf-8-sig")
