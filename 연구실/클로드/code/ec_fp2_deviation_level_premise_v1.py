# -*- coding: utf-8 -*-
"""FP2 — 편차 지문(규정 근사판)으로 출처를 정하고 같은 출처 달력 이웃 정답으로 2차 하루 수준을 추정하면 SG2 보다 나은가 (전제 검산) — 2026-10-07 연구실 클로드
FP1: 같은 날짜 묶음 편차 지문 k=4 → 같은 날짜 기록 쌍별 다른 군집 98.2%(우연 75%), 앞날 지문으로 다음날 기록 고르기 88%(우연 27%).
규정 근사:
 - 묶음 평균은 '정답 있는 학습 기록, 대상 b 제외'로만 계산(평가 기록·b 자신 입력이 평균에 안 들어감). 대상 b 의 편차 = b 지문 − 그 평균.
 - 군집 중심은 학습 기록(편차 = 학습 기록끼리 평균 기준)으로 KMeans k=4 (seed 0, n_init 20).
 - 지문은 아직 하루 전체(시각 인과판은 다음 단계) → 이번 결과는 상한 성격.
추정: SG2 달력(정답 전체 − b±1 기록)에서 b 의 달력 cb, 같은 군집 정답 기록 중 cb 앞쪽 최근접 1개·뒤쪽 최근접 1개(|Δcal| ≤ 5)의 하루평균을
      거리 가중 보간(한쪽만 있으면 그 값) = E_src. 대조: 같은 창의 아무 군집 최근접 앞뒤 보간 E_any.
비교: SG2 저장 OOF(DIAG10 시드평균) 하루평균. 2차 46일 전체 / 예측 일반 / 정답 일반 / 고EC.
"""
import os, sys, importlib.util
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research")
sys.path.insert(0, RES)
import env  # noqa
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(RES, "ec3_SG2_reference_knn_level_v1.py")); sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
R = os.path.dirname(os.path.abspath(__file__))
ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]; INDOOR = ["in_temp", "in_hum", "in_co2"]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TR["t"] = 0; TE["t"] = 1
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
ym = Y[Y.farm.isin(["F13", "F47"])].groupby(["farm", "day"]).sub_ec.mean(); labset = set(ym.index)
g = X.groupby(["farm", "day"]); isT = g.t.first()
F = pd.concat([g[ACTS].mean().add_suffix("_m"), g[ACTS].agg(lambda s: (s == 0).mean()).add_suffix("_z"), g[INDOOR].mean().add_suffix("_d")], axis=1)
Fs = ((F - F[isT == 0].mean()) / F[isT == 0].std().replace(0, 1)).fillna(0)
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
def dev(k, exclude=()):
    mem = [m for m in grp.index[grp == grp[k]] if m in labset and m != k and m not in exclude]
    if not mem:
        return None
    return Fs.loc[k].values - Fs.loc[mem].values.mean(axis=0)
trainDv = {k: dev(k) for k in labset}
M = np.array([v for v in trainDv.values() if v is not None]); Kk = [k for k, v in trainDv.items() if v is not None]
km = KMeans(4, n_init=20, random_state=0).fit(M)
clab = dict(zip(Kk, km.labels_))
Rr, WVs, hrs, SIG = sg2.prepare_structure()
S = pd.read_csv(os.path.join(RES, "local", "ec3_SG2_all.csv")); S = S[S.validator == "DIAG10"]
S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1); sgd = S.groupby(["farm", "day"]).sg.mean()
rows = []
for b in sgd.index:
    ref = {x for x in labset if not (x[0] == b[0] and abs(x[1] - b[1]) <= 1)}
    dv = dev(b, exclude=())
    cb_cl = int(km.predict(dv[None, :])[0]) if dv is not None else -1
    cal = sg2.ref_calendar(Rr, WVs, ref)
    E = [e for e in Rr[Rr.farm == b[0]].day if (b[0], e) in ref]
    A1 = WVs.loc[[(b[0], e) for e in E]].values; okk = np.sqrt(np.nanmean((A1 - WVs.loc[b].values) ** 2, axis=1)) <= .05
    if okk.any():
        cb = float(np.mean([cal[(b[0], e)] for e, o in zip(E, okk) if o]))
    else:
        ds = sorted(Rr[Rr.farm == b[0]].day); i = ds.index(b[1]); p = [d for d in ds[:i] if (b[0], d) in cal]
        cb = cal[(b[0], p[-1])] + .1 if p else np.nan
    def interp(cands):
        pre = [(cb - cal[k], k) for k in cands if 0 < cb - cal[k] <= 5]; post = [(cal[k] - cb, k) for k in cands if 0 < cal[k] - cb <= 5]
        same = [k for k in cands if cal[k] == cb]
        if same:
            return float(np.median([ym[k] for k in same]))
        if pre and post:
            (d1, k1), (d2, k2) = min(pre), min(post); return float((d2 * ym[k1] + d1 * ym[k2]) / (d1 + d2))
        if pre or post:
            return float(ym[min(pre or post)[1]])
        return np.nan
    cands = [(b[0], e) for e in E if (b[0], e) in cal]
    src = [k for k in cands if clab.get(k) == cb_cl]
    rows.append(dict(farm=b[0], day=b[1], y=ym[b], sg=sgd[b], cl=cb_cl, e_src=interp(src), e_any=interp(cands), nsrc=len(src)))
D = pd.DataFrame(rows)
rm = lambda x: np.sqrt(np.nanmean(np.square(x)))
for nm, m in (("2차 전체", D.y > -1), ("예측 일반(sg<.9)", D.sg < .9), ("정답 일반(y<1)", D.y < 1), ("고EC(y≥1)", D.y >= 1)):
    q = D[m & D.e_src.notna()]
    print("%-14s n=%2d | SG2 %.3f | 같은 출처(편차 군집) 앞뒤 보간 %.3f | 아무 출처 앞뒤 보간 %.3f | 반반(SG2·출처) %.3f" % (
        nm, len(q), rm(q.sg - q.y), rm(q.e_src - q.y), rm(q.e_any - q.y), rm((q.sg + q.e_src) / 2 - q.y)))
print("추정 없는 날 %d" % D.e_src.isna().sum())
D.to_csv(os.path.join(R, "..", "results", "ec_fp2_days_v1.csv"), index=False, encoding="utf-8-sig")
