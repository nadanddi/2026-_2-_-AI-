# -*- coding: utf-8 -*-
"""SN0 — '전날 후보 값으로 끌어당기기(snap)' 전제 검산 (진단) — 2026-10-07 연구실 클로드
발상: 날짜가 확정된 2차 기록 b(C1·C2) 의 전날 날짜 묶음 P 는 1차 구간이라 전부 정답 기록. 같은 출처면 EC0(b) ≈ EC23(앞날).
 기존(FC0~FC3·GX)은 '어느 후보가 앞날인가'를 지문·배제로 골랐고 시각 인과판에서 실패.
 SN0 은 후보 선택을 b 의 시각 인과 예측(pm_h = SG2 0..h 누적평균)에 가장 가까운 후보 EC23 로 한다.
 틀린 후보를 골라도 이동량이 |v − pm_h| ≤ δ 로 묶여 손해가 제한된다.
후보 집합: U0 = P 의 정답 기록 전부, U1 = 배제(b 의 묶음 안 다른 정답 기록이 EC 연속 비용으로 차지한 앞날 제외, 헝가리안).
가림: (i) b 만, (ii) b 와 같은 DIAG10 폴드 기록 전부(후보·배제 모두에서 제외) — (ii)가 현실적.
보정(시각 h): pred_h = sg_h + w·(v_h − pm_h)  if |v_h − pm_h| < δ  else sg_h.
기준 예측: RV2 저장 CUR+SG2 (DIAG10, 시드 47/1414/6464 평균).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TR["t"] = 0; TE["t"] = 1
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
EC = Y[Y.farm.isin(["F13", "F47"])].pivot_table(index=["farm", "day"], columns="hour", values="sub_ec"); ym = EC.mean(axis=1); labset = set(EC.index)
isT = X.groupby(["farm", "day"]).t.first()
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
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); f10 = DP[DP.validator == "DIAG10"].groupby(["farm", "day"]).validation_fold.first().to_dict()
RV2 = pd.read_csv(os.path.join(R, "..", "results", "ec_rv2_preds_v1.csv")); RV2 = RV2[(RV2.validator == "DIAG10") & (RV2.tag == "CUR")]
SG = RV2.groupby("row_id").sg.mean(); SGd = pd.DataFrame({"row_id": SG.index, "sg": SG.values})
SGd["farm"], SGd["day"], SGd["hour"] = SGd.row_id.str[:3], SGd.row_id.str[4:7].astype(int), SGd.row_id.str[8:10].astype(int)
SGH = SGd.pivot_table(index=["farm", "day"], columns="hour", values="sg")
def prev_group(b):
    p1 = sorted([k for k in grp.index[grp == grp[b]] if k[1] < 179 and k in labset], key=lambda k: k[1])
    if not p1:
        return None
    f0, d0 = p1[0]
    for d in sorted((d for f, d in keys if f == f0 and d < 179 and d < d0), reverse=True):
        if grp[(f0, d)] != grp[b]:
            return grp[(f0, d)]
    return None
def cands(b, hidden, excl):
    pg = prev_group(b)
    if pg is None:
        return None, None
    P = [k for k in grp.index[grp == pg] if k in labset and k not in hidden]
    if not P:
        return None, None
    if excl:
        G = [k for k in grp.index[grp == grp[b]] if k in labset and k not in hidden and k != b]
        if G:
            C = np.array([[(EC.loc[c, 0] - EC.loc[a, 23]) ** 2 for a in P] for c in G])
            r, c_ = linear_sum_assignment(C)
            taken = {P[j] for i, j in zip(r, c_) if C[i, j] < .05 ** 2 * 4}
            P = [a for a in P if a not in taken] or P
    # 진짜 앞날(판정용, b 정답 사용): 전체 정답 P 중 EC 연속 최소
    Pall = [k for k in grp.index[grp == pg] if k in labset]
    true_a = min(Pall, key=lambda a: abs(EC.loc[b, 0] - EC.loc[a, 23]))
    return P, true_a
TGT = [b for b in SGH.index if b[1] >= 179 and b in labset and cls.get(b) in ("C1", "C2")]
rm = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
for hide_name in ("b만", "폴드 전체"):
    for excl in (False, True):
        out = {}
        info = []
        for b in TGT:
            hidden = {k for k in keys if isT[k] == 1} | {b}
            if hide_name == "폴드 전체":
                hidden |= {k for k in labset if f10.get(k) == f10.get(b)}
            P, ta = cands(b, hidden, excl)
            if P is None:
                continue
            v23 = np.array([EC.loc[a, 23] for a in P]); y = EC.loc[b].values; sg = SGH.loc[b].values; pm = np.cumsum(sg) / np.arange(1, 25)
            vh = v23[np.argmin(np.abs(v23[None, :] - pm[:, None]), axis=1)]
            info.append(dict(b=b, n=len(P), true_in=ta in P, y=ym[b]))
            for w in (1., .5):
                for dl in (.1, .2, .3, 9.):
                    pr = np.where(np.abs(vh - pm) < dl, sg + w * (vh - pm), sg)
                    out.setdefault((w, dl), []).append((pr - y, sg - y, ym[b]))
            orc = EC.loc[ta, 23] if ta in P else np.nan
            out.setdefault("orc", []).append(((sg - pm + orc) - y if np.isfinite(orc) else sg - y, sg - y, ym[b]))
        I = pd.DataFrame(info)
        print("\n[가림 %s | 배제 %s] 대상 %d | 후보 수 중앙 %.1f | 진짜 앞날이 후보 안 %.2f" % (hide_name, excl, len(I), I.n.median(), I.true_in.mean()))
        for key, v in out.items():
            e = np.array([x[0] for x in v]); s = np.array([x[1] for x in v]); yy = np.array([x[2] for x in v])
            nm = yy < 1
            nb = sum((np.mean(x[0] ** 2) < np.mean(x[1] ** 2) - 1e-12) for x, q in zip(v, nm) if q); nw = sum((np.mean(x[0] ** 2) > np.mean(x[1] ** 2) + 1e-12) for x, q in zip(v, nm) if q)
            print("  %-12s 전체 SG2 %.4f → %.4f | 일반(n=%d) %.4f → %.4f (개선 %d·악화 %d) | 고EC %.4f → %.4f" % (
                str(key), rm(s), rm(e), nm.sum(), rm(s[nm]), rm(e[nm]), nb, nw, rm(s[~nm]), rm(e[~nm])))
