# -*- coding: utf-8 -*-
"""HX9 근권 온도 가족 — 후속 v2 (2026-10-13 집 클로드). v1(rz_rootzone_v1.py) 결과를 본 뒤의 사후 점검(탐색, 판정 아님).
v1 에서 눈에 띈 두 가지를 다시 본다. 사후라서 '확인'이 아니라 '견고성 점검'으로만 쓴다.
 (1) Y3(고EC 사슬 안 같은 동 연속일 EC 변화)과 Δ배지온(Δst, 진단) / Δ예측 배지온(Δpst, 입력 대리) 의 양의 관계
     (v1: Δln EC ~ Δst 기울기 +.054/℃ CI[+.013,+.084], 센서 예측 .02 보다 큼; r(ΔEC, Δpst_min) .45, 39변수 중 최대).
     점검: 사슬 부트스트랩 CI, 사슬 하나 빼기 교차적합 R², 순열 p, 날씨(Δ외기온·Δ외기 일사) 통제 편상관(N2: 날씨 대리인가),
     같은 온실 id 링크 vs 다른 id 링크, 1차/2차, 큰 사슬 ep1 제외, 보통 기록 링크와 비교(고EC 특유인가).
     사전에 정한 작은 집합만: dst_mean, dst_min (진단), dpst_mean, dpst_min, din_min, din_mean (입력 대리; 링크가 진단용이므로 지금은 서술).
 (2) 전체 360 기록에서 온도 수준 변수의 교차적합 이득 +.02 (r .05) — 편상관(sg 통제), 1차/2차, 폴드별 부호.
예측(이 점검 전에 적음): (1) Δst 효과가 Δ외기온을 통제하면 크게 줄면 날씨(증산·관리) 경로이고 '근권 온도' 고유 효과 아님.
 (2) 전체 기록 이득은 계절(겨울 과대/과소 예측) 잔재일 가능성이 크고 2차에서 재현 안 될 것.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import env  # noqa: F401
import warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
RNG = np.random.default_rng(1013)
V = pd.read_csv(os.path.join(HERE, "vars_v1.csv"), index_col="rec")
R = pd.read_csv(os.path.join(HX9, "data", "records.csv"), index_col="rec")
H = pd.read_csv(os.path.join(HX9, "data", "hourly.csv")).sort_values(["rec", "hour"])
REC = list(R.index)
OT = H.pivot(index="rec", columns="hour", values="out_temp").loc[REC].values
RAD = H.pivot(index="rec", columns="hour", values="out_rad").loc[REC].values
W = pd.DataFrame({"out_temp_mean": OT.mean(1), "out_rad_sum": RAD.sum(1)}, index=REC)
V = V.join(W)

def chain_boot(x, y, g, B=4000, fn=None):
    fn = fn or (lambda a, b: np.corrcoef(a, b)[0, 1])
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; out = []
    for _ in range(B):
        s = RNG.choice(ug, len(ug), replace=True); ii = np.concatenate([idx[u] for u in s])
        if np.std(x[ii]) > 0 and np.std(y[ii]) > 0: out.append(fn(x[ii], y[ii]))
    return np.percentile(out, [2.5, 97.5])
def perm_p(x, y, P=10000):
    r0 = abs(np.corrcoef(x, y)[0, 1]); xs = (x - x.mean()) / x.std(); ys = (y - y.mean()) / y.std()
    rp = np.abs(np.array([RNG.permutation(ys) for _ in range(P)]) @ xs / len(xs))
    return (1 + (rp >= r0 - 1e-12).sum()) / (P + 1)
def loco_r2(x, y, g):
    pv = np.zeros(len(y)); p0 = np.zeros(len(y))
    for u in np.unique(g):
        te = g == u; tr = ~te
        b = np.polyfit(x[tr], y[tr], 1); pv[te] = np.polyval(b, x[te]); p0[te] = y[tr].mean()
    sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pv) ** 2) / sst, 1 - np.sum((y - p0) ** 2) / sst, int(((y - pv) ** 2 < (y - p0) ** 2).sum())
def partial(x, y, Z):
    Z = np.c_[np.ones(len(x)), Z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]; ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1], rx, ry

# ---------------- (1) Y3 ----------------
L = R[R.prev_rec.notna()].copy()
L["dEC"] = L.ec_mean - L.prev_ec_mean
L["dlnEC"] = np.log(L.ec_mean) - np.log(L.prev_ec_mean)
L["dlnEC25"] = np.log(V.loc[L.index, "ec25_mean"].values) - np.log(V.loc[L.prev_rec, "ec25_mean"].values)
for v in ["st_mean", "st_min", "pst_mean", "pst_min", "in_min", "in_mean", "out_temp_mean", "out_rad_sum"]:
    L["d_" + v] = V.loc[L.index, v].values - V.loc[L.prev_rec, v].values
L["hh"] = (L.hi10 == 1) & (L.prev_hi10 == 1)
Y3 = L[L.hh].copy(); g3 = Y3.ep_id.values.astype(int)
print(f"(1) hi10->hi10 clean links n={len(Y3)} (chains {len(np.unique(g3))}; same-farm {int(Y3.prev_same_farm.sum())}, pass2 {int(Y3.pass2.sum())})")
print(f"    weather change SD: d_out_temp {Y3.d_out_temp_mean.std():.2f}C, d_out_rad_sum {Y3.d_out_rad_sum.std():.0f}; "
      f"corr(d_st_mean, d_out_temp) {np.corrcoef(Y3.d_st_mean, Y3.d_out_temp_mean)[0,1]:.2f}, corr(d_st_mean, d_out_rad) {np.corrcoef(Y3.d_st_mean, Y3.d_out_rad_sum)[0,1]:.2f}")
rows = []
SET = ["d_st_mean", "d_st_min", "d_pst_mean", "d_pst_min", "d_in_min", "d_in_mean", "d_out_temp_mean", "d_out_rad_sum"]
for yv in ["dEC", "dlnEC", "dlnEC25"]:
    for v in SET:
        x = Y3[v].values; y = Y3[yv].values
        r = np.corrcoef(x, y)[0, 1]; ci = chain_boot(x, y, g3); p = perm_p(x, y)
        r2v, r20, nimp = loco_r2(x, y, g3)
        pw, _, _ = partial(x, y, Y3[["d_out_temp_mean", "d_out_rad_sum"]].values) if not v.startswith("d_out") else (np.nan, 0, 0)
        sl = stats.linregress(x, y).slope
        m1 = Y3.ep_id.values != 1; r_noep1 = np.corrcoef(x[m1], y[m1])[0, 1]
        sf = Y3.prev_same_farm.values == 1; r_sf = np.corrcoef(x[sf], y[sf])[0, 1]; r_xf = np.corrcoef(x[~sf], y[~sf])[0, 1] if (~sf).sum() > 3 else np.nan
        p2 = Y3.pass2.values == 1; r_p1 = np.corrcoef(x[~p2], y[~p2])[0, 1]
        rows.append(dict(y=yv, var=v, r=r, ci_lo=ci[0], ci_hi=ci[1], perm_p=p, slope=sl, loco_r2=r2v, loco_r2_mean=r20, n_imp=nimp,
                         pr_ctrl_weather=pw, r_noep1=r_noep1, r_samefarm=r_sf, n_sf=int(sf.sum()), r_crossfarm=r_xf, r_pass1=r_p1, n_p2=int(p2.sum())))
T = pd.DataFrame(rows)
print(T.round(3).to_string(index=False))
# slope comparison: hi10 links vs normal links (is the excess over sensor .02 specific to high episodes?)
NL = L[(L.hi10 == 0) & (L.prev_hi10 == 0)]
for lab, S in (("hi10->hi10", Y3), ("normal->normal", NL)):
    for v in ["d_st_mean", "d_pst_mean", "d_out_temp_mean"]:
        s = stats.linregress(S[v], S.dlnEC)
        print(f"    slope dlnEC ~ {v:16s} {lab:15s} n={len(S):3d}: {s.slope:+.4f} (r {s.rvalue:+.2f})")
# multiple regression: dlnEC ~ d_st_mean + d_out_temp + d_out_rad on hi10 links (chain bootstrap of d_st coefficient)
X = np.c_[np.ones(len(Y3)), Y3[["d_st_mean", "d_out_temp_mean", "d_out_rad_sum"]].values]
b = np.linalg.lstsq(X, Y3.dlnEC.values, rcond=None)[0]
ug = np.unique(g3); idx = {u: np.where(g3 == u)[0] for u in ug}; bs = []
for _ in range(4000):
    s = RNG.choice(ug, len(ug), replace=True); ii = np.concatenate([idx[u] for u in s])
    bs.append(np.linalg.lstsq(X[ii], Y3.dlnEC.values[ii], rcond=None)[0])
bs = np.array(bs)
print(f"    dlnEC ~ d_st + d_out_temp + d_out_rad (hi10 links): d_st {b[1]:+.4f}/C CI[{np.percentile(bs[:,1],2.5):+.4f},{np.percentile(bs[:,1],97.5):+.4f}], "
      f"d_out_temp {b[2]:+.4f} CI[{np.percentile(bs[:,2],2.5):+.4f},{np.percentile(bs[:,2],97.5):+.4f}], d_out_rad {b[3]*1000:+.4f}/1000 CI[{np.percentile(bs[:,3],2.5)*1000:+.4f},{np.percentile(bs[:,3],97.5)*1000:+.4f}]")
T.to_csv(os.path.join(HERE, "y3_followup_v2.csv"), index=False, encoding="utf-8-sig")

# ---------------- (2) all-record temperature-level gain ----------------
print("\n(2) all 360 records: Y2 (y - sg) vs temperature level, controlling sg")
A = R[R.has_sg == 1].copy()
lc = V.loc[A.index, "linkcomp"].values; lcc = pd.factorize(lc)[0]
for v in ["st_mean", "pst_mean", "in_mean", "st_cold14", "pst_cold14", "out_temp_mean"]:
    x = V.loc[A.index, v].values; y = A.resid_mean.values; sg = A.sg_mean.values
    pr_, rx, ry = partial(x, y, sg[:, None])
    ci = chain_boot(rx, ry, lcc, B=2000)
    p2 = A.pass2.values == 1
    pr1 = partial(x[~p2], y[~p2], sg[~p2, None])[0]; pr2 = partial(x[p2], y[p2], sg[p2, None])[0]
    hi = A.hi10.values == 1
    prn = partial(x[~hi], y[~hi], sg[~hi, None])[0]
    # fold signs (DIAG10 fold)
    fs = []
    for f in sorted(A.fold.dropna().unique()):
        m = A.fold.values == f
        if m.sum() > 8: fs.append(np.sign(partial(x[m], y[m], sg[m, None])[0]))
    print(f"  {v:14s} partial r|sg {pr_:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] pass1 {pr1:+.3f} pass2 {pr2:+.3f} (n2 {int(p2.sum())}) "
          f"normal-only {prn:+.3f}; DIAG10 fold signs + {int(np.sum(np.array(fs) > 0))}/{len(fs)}")
