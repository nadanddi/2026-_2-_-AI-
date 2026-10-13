# -*- coding: utf-8 -*-
"""HX9 / 가족: 근권(배지) 온도와 EC 온도 효과 — v1 (2026-10-13 집 클로드, 하위 에이전트)
사용자 요청: '고EC의 크기 예측에 인과관계가 조금이라도 있는 데이터 전부 확인. 파생 변수도 직접 만들어 확인'.
이 파일은 탐색·서술(채택 판정 아님). 기존 파일 수정 없음. 산출은 hx9/rootzone/ 아래만.

========================================================================================================
[실행 전에 고정한 예측 — 이 블록은 결과를 보기 전에 썼고, 결과를 본 뒤 고치지 않는다]
기제(일반 지식 표시): (G1) EC 센서·이온 이동도는 온도에 따라 약 2%/℃ 변한다(무보정 EC의 교과서 값).
  이 자료에서는 이미 기록 안 기울기 γ ≈ .020/℃ 확인(카탈로그 6.27/6.130/6.657 RZ4).
  (G2) 일반 지식: 근권이 차가우면 물·양분 흡수가 줄어든다(방향은 물이 덜 빠지면 EC↓, 이온이 덜 빠지면 EC↑ — 양쪽 다 가능).
  (G3) 난방은 공기를 데우고 배지는 1~2시간 늦게 따라 데워진다(6.610/6.655 확인).
이 기제가 '고EC 크기'를 움직인다면 보일 흔적(지문):
  P1 센서 효과: 고EC 기록 사이 ln(하루 평균 EC)와 배지온도 하루 평균의 기울기 ≈ +.02/℃.
     고EC 기록의 배지온도 하루 평균 SD ≈ 1.5℃ → 하루 평균 EC 차이 ±3% (EC 1.6에서 ±.05) 뿐.
     모델 잔차 SD ≈ .34 → 센서 효과로 설명 가능한 잔차 분산은 많아야 ~2~5% (|r| ≲ .15~.2) 로 예측.
     'ln 잔차 ~ (실제 배지온 − 예측 배지온)의 ln 보정계수 차'의 기울기는 ≈1 이어야 한다.
     같은 동 연속일(Δ): Δln EC ~ Δ배지온 기울기 ≈ .02, 온도 보정 EC25 의 Δ 는 Δ배지온과 무관(≈0)이어야 한다.
  P2 근권 냉해/흡수 저하: 고EC 기록 안에서 차가운 배지(낮은 최저, 많은 냉량 ℃·h)일수록 EC25 가 높고 잔차가 +.
     부호는 배지온과 Y2 사이 음(−). 계절(외기온) 통제 뒤에도 남아야 한다. 연속일에서 추워진 날 EC 상승.
  P3 난방: 난방 시간·밤 난방이 많을수록 배지−공기 차↑, 근권 온도↑ → (P2 경로면) EC25↓, 잔차 −.
  P4 동 지문: 고EC 동이 다른 베드(배지 보온) 유형이면 배지−실내 차 / 예측 대비 배지온 잔차가 고EC 와 보통에서 다르고,
     고EC 안에서 크기와 함께 움직인다.
  P5 입력 대리값: 진단 변수(정답 배지온)에 신호가 있고 입력으로 만든 예측 배지온 대리값에 없으면, 그 정보는 평가 행에서 못 쓴다.
보이면 안 되는 것(기제 반증):
  N1 센서 효과로 보기엔 너무 큰 기울기(예: |dlnEC/dT| > .06/℃)를 '센서 효과'로 부르는 것 — 그건 관리·계절 공변동이다.
  N2 효과가 외기온(계절) 통제 뒤 사라짐 → 계절 대리일 뿐.
  N3 효과가 한 사슬(특히 F47 큰 사슬 ep1)이나 F47_132 한 기록에만 의존.
  N4 EC25 로 바꿨을 때 고EC 기록 사이 변동이 오히려 커지거나 설명력이 같다면 온도는 크기 원인이 아님.
전체 예측(사전): 사용 가능 변수는 none~weak, 진단 변수 중 배지온 잔차·결합(γ)은 weak 가능성. 다변량 상한 R²cv ≤ 0.
========================================================================================================
쓰는 자료(읽기만): hx9/data/records.csv, hx9/data/hourly.csv (prep_table_v1 산출).
평가 행 사용 가능성 정의: usable = 그 기록 0..h 입력 + 학습 정답/입력만으로 계산 가능 (하루 집계는 h=23 상한;
  0..h 누적판을 h=0,6,12,18,23 에서 따로 시험). diagnostic = 대상 자신의 정답(sub_temp/sub_ec) 또는 정답 기반 연결 필요.
통계: 고EC(hi10) 기록 35개(모델 OOF 있는 31개), 사슬 = ep_id 8개. 사슬 부트스트랩 CI, 사슬 하나 빼기(LOCO) 교차적합,
  기록 순열 p, 변수 수만큼 BH 보정. 전체 기록(360)은 clean 링크 연결 성분(81개)을 묶음으로 사용.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # .../research
import env  # noqa: F401
import warnings, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
HERE = os.path.dirname(os.path.abspath(__file__))
HX9 = os.path.dirname(HERE)
RNG = np.random.default_rng(20261013)

R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
H = pd.read_csv(os.path.join(HX9, "data", "hourly.csv"))
H = H.sort_values(["rec", "hour"]).reset_index(drop=True)
REC = list(R.index)
assert H.rec.nunique() == 400 and len(H) == 9600
def piv(c):
    return H.pivot(index="rec", columns="hour", values=c).loc[REC].values.astype(float)
EC = piv("sub_ec"); ST = piv("sub_temp"); SG = piv("sg")
IN_raw = piv("in_temp"); HUM = piv("in_hum"); OT = piv("out_temp"); RAD = piv("out_rad"); WS = piv("out_wspd")
HEAT = piv("act_heating"); THERM = piv("act_thermal"); SHADE = piv("act_shade"); VENT = piv("act_vent"); FAN = piv("act_circfan")
def ffill_causal(A):
    """within-record forward fill only (no future values); leading NaN stays NaN."""
    B = A.copy()
    for h in range(1, 24):
        m = np.isnan(B[:, h]); B[m, h] = B[m, h - 1]
    return B
IN = ffill_causal(IN_raw); HUMf = ffill_causal(HUM)
NREC = len(REC)
FARM1 = (R.farm.values == "F47").astype(float)

# ---------------- chains / groups ----------------
par = {r: r for r in REC}
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for b, a in zip(R.index, R.prev_rec):
    if isinstance(a, str): par[fd(b)] = fd(a)
R["linkcomp"] = [fd(r) for r in REC]
LC_CODE = pd.factorize(R.linkcomp)[0]
print(f"records {NREC}; clean-link components {R.linkcomp.nunique()} (max size {R.linkcomp.value_counts().max()})")

# ---------------- (A) predicted substrate temperature from inputs (usable proxy) ----------------
def lag(A, k):
    B = np.empty_like(A)
    for h in range(24): B[:, h] = A[:, max(h - k, 0)]   # within record only; before hour 0 -> hour-0 value (causal)
    return B
def cummean(A):
    out = np.full_like(A, np.nan)
    for h in range(24): out[:, h] = np.nanmean(A[:, :h + 1], 1)
    return out
feat_list = {"hour": np.tile(np.arange(24.0), (NREC, 1)), "farm": np.tile(FARM1[:, None], (1, 24)),
             "in": IN, "in_cm": cummean(IN), "hum": HUMf, "ot": OT, "rad": RAD, "rad_cs": np.cumsum(RAD, 1), "ws": WS,
             "heat": HEAT, "therm": THERM, "shade": SHADE, "vent": VENT, "fan": FAN}
for k in range(1, 7): feat_list[f"in_l{k}"] = lag(IN, k)
for k in range(1, 4): feat_list[f"ot_l{k}"] = lag(OT, k)
for k in range(1, 5): feat_list[f"rad_l{k}"] = lag(RAD, k)
for k in (1, 2): feat_list[f"heat_l{k}"] = lag(HEAT, k); feat_list[f"therm_l{k}"] = lag(THERM, k)
FN = list(feat_list)
XH = np.stack([feat_list[f].reshape(-1) for f in FN], 1)
yH = ST.reshape(-1)
gH = np.repeat(LC_CODE, 24)
PST = np.full(NREC * 24, np.nan)
for tr, te in GroupKFold(10).split(XH, yH, gH):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)
    m.fit(XH[tr], yH[tr]); PST[te] = m.predict(XH[te])
PST = PST.reshape(NREC, 24)
hi = R.hi10.values == 1
rm = lambda a, b: float(np.sqrt(np.nanmean((a - b) ** 2)))
print(f"[A] predicted sub_temp (cross-fit by link component, 10 folds): RMSE all {rm(PST, ST):.3f}C, hi10 {rm(PST[hi], ST[hi]):.3f}C; "
      f"day-mean RMSE all {rm(PST.mean(1), ST.mean(1)):.3f}, hi10 {rm(PST[hi].mean(1), ST[hi].mean(1)):.3f}; "
      f"day-mean bias hi10 {np.mean(ST[hi].mean(1) - PST[hi].mean(1)):+.3f}, normal {np.mean(ST[~hi].mean(1) - PST[~hi].mean(1)):+.3f}")

# ---------------- (B) variables ----------------
NIGHT = np.array([h <= 5 or h >= 20 for h in range(24)])
def wmask(h, night=False):
    m = np.arange(24) <= h
    return m & NIGHT if night else m
def nm(A, m): return np.nanmean(A[:, m], 1) if m.any() else np.full(NREC, np.nan)
lnfac = lambda T: np.log(1 + 0.02 * (T - 25.0))
def usable_vars(h):
    """0..h cumulative versions (h=23 -> day aggregate)."""
    m = wmask(h); mn = wmask(h, True)
    d = {}
    d["in_mean"] = nm(IN, m); d["in_min"] = np.nanmin(np.where(m, IN, np.nan), 1); d["in_h0"] = IN[:, 0]
    d["in_night"] = nm(IN, mn)
    for base in (10, 12, 14): d[f"in_cold{base}"] = np.nansum(np.clip(base - IN[:, m], 0, None), 1)
    d["in_below10"] = np.nansum(IN[:, m] < 10, 1).astype(float)
    d["in_amp"] = np.nanmax(np.where(m, IN, np.nan), 1) - np.nanmin(np.where(m, IN, np.nan), 1)
    d["heat_hours"] = (HEAT[:, m] > 0).sum(1).astype(float); d["heat_mean"] = HEAT[:, m].mean(1)
    d["heat_night"] = nm(HEAT, mn)
    d["inout_night"] = nm(IN - OT, mn)
    d["pst_mean"] = PST[:, m].mean(1); d["pst_min"] = PST[:, m].min(1); d["pst_h0"] = PST[:, 0]
    d["pst_in"] = nm(PST - IN, m)
    for base in (12, 14, 16): d[f"pst_cold{base}"] = np.clip(base - PST[:, m], 0, None).sum(1)
    d["pst_rise"] = PST[:, m].max(1) - PST[:, 0]
    d["lnfac_pred"] = lnfac(PST[:, m]).mean(1)
    return d
U23 = usable_vars(23)
USABLE = list(U23)
D = {}
D["st_mean"] = ST.mean(1); D["st_min"] = ST.min(1); D["st_h0"] = ST[:, 0]; D["st_amp"] = ST.max(1) - ST.min(1)
D["st_in_mean"] = np.nanmean(ST - IN, 1); D["st_in_night"] = np.nanmean((ST - IN)[:, NIGHT], 1); D["st_in_h0"] = ST[:, 0] - IN[:, 0]
for base in (12, 14, 16): D[f"st_cold{base}"] = np.clip(base - ST, 0, None).sum(1)
D["st_below12"] = (ST < 12).sum(1).astype(float)
D["st_resid_mean"] = (ST - PST).mean(1); D["st_resid_h0"] = ST[:, 0] - PST[:, 0]; D["st_resid_night"] = (ST - PST)[:, NIGHT].mean(1)
D["lnfac_gap"] = lnfac(ST).mean(1) - lnfac(PST).mean(1)   # expected ln-residual from temperature misprediction under pure sensor model
lnEC = np.log(EC)
gam = np.full(NREC, np.nan); cor = np.full(NREC, np.nan)
for i in range(NREC):
    if np.std(ST[i]) > 0.2:
        sl = stats.linregress(ST[i], lnEC[i]); gam[i] = sl.slope; cor[i] = sl.rvalue
D["gamma_rec"] = gam; D["corr_rec"] = cor
DIAG = list(D)
V = pd.DataFrame({**U23, **D}, index=REC)
VARS = USABLE + DIAG
print(f"[B] variables: usable {len(USABLE)}, diagnostic {len(DIAG)}, total {len(VARS)}")

# targets & helper columns
EC25 = EC / (1 + 0.02 * (ST - 25.0))
SG25 = SG / (1 + 0.02 * (PST - 25.0))
R["ec25_mean"] = EC25.mean(1); R["sg25_mean"] = SG25.mean(1)
R["Y2"] = R.resid_mean
R["Y2log"] = np.log(R.ec_mean) - np.log(R.sg_mean)
R["Y2c"] = np.log(R.ec25_mean) - np.log(R.sg25_mean)
OUTV = V.copy()
for c in ["farm", "pass2", "hi10", "hi12", "ep_id", "linkcomp", "ec_mean", "sg_mean", "Y2", "Y2log", "ec25_mean", "sg25_mean", "Y2c", "prev_rec"]:
    OUTV[c] = R[c]
OUTV.index.name = "rec"
OUTV.to_csv(os.path.join(HERE, "vars_v1.csv"), encoding="utf-8-sig")

# ---------------- (C) per-variable statistics ----------------
hiS = R[(R.hi10 == 1)].copy()                 # Y1 set (35)
hiR = R[(R.hi10 == 1) & (R.has_sg == 1)].copy()   # Y2 set (31)
alR = R[R.has_sg == 1].copy()                 # all with OOF (360)
CH = hiR.ep_id.values.astype(int)
def pr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) == 0: return np.nan, np.nan, int(m.sum())
    return stats.pearsonr(x[m], y[m])[0], stats.spearmanr(x[m], y[m])[0], int(m.sum())
def boot_ci(x, y, groups, B=2000):
    m = np.isfinite(x) & np.isfinite(y); x, y, g = x[m], y[m], groups[m]
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; rs = []
    for _ in range(B):
        s = RNG.choice(ug, len(ug), replace=True); ii = np.concatenate([idx[u] for u in s])
        if np.std(x[ii]) > 0 and np.std(y[ii]) > 0: rs.append(np.corrcoef(x[ii], y[ii])[0, 1])
    return np.nanpercentile(rs, [2.5, 97.5]) if rs else (np.nan, np.nan)
def perm_p(x, y, P=5000):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    r0 = abs(np.corrcoef(x, y)[0, 1]); xs = (x - x.mean()) / x.std(); ys = (y - y.mean()) / y.std()
    perms = np.array([RNG.permutation(ys) for _ in range(P)])
    rp = np.abs(perms @ xs / len(xs))
    return (1 + (rp >= r0 - 1e-12).sum()) / (P + 1)
def ols_fit_pred(Xtr, ytr, Xte):
    A = np.c_[np.ones(len(Xtr)), Xtr]; b = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return np.c_[np.ones(len(Xte)), Xte] @ b
def cv_gain(y, base, var, groups):
    """leave-one-group-out OLS: y ~ base vs y ~ base + var (same rows). returns R2cv_base, R2cv_var, gain, n_improve, n."""
    m = np.isfinite(y) & np.isfinite(var) & np.isfinite(base)
    y, base, var, g = y[m], base[m], var[m], groups[m]
    pb = np.zeros(len(y)); pv = np.zeros(len(y))
    for u in np.unique(g):
        te = g == u; tr = ~te
        pb[te] = ols_fit_pred(base[tr, None], y[tr], base[te, None])
        pv[te] = ols_fit_pred(np.c_[base[tr], var[tr]], y[tr], np.c_[base[te], var[te]])
    sst = np.sum((y - y.mean()) ** 2)
    r2b = 1 - np.sum((y - pb) ** 2) / sst; r2v = 1 - np.sum((y - pv) ** 2) / sst
    return r2b, r2v, r2v - r2b, int(((y - pv) ** 2 < (y - pb) ** 2).sum()), int(m.sum())
def partial_r(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 6: return np.nan
    Z = np.c_[np.ones(m.sum()), z[m]]
    rx = x[m] - Z @ np.linalg.lstsq(Z, x[m], rcond=None)[0]; ry = y[m] - Z @ np.linalg.lstsq(Z, y[m], rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]

# Y3: day-to-day change along clean links, both ends hi10
y3 = R[(R.hi10 == 1) & (R.prev_hi10 == 1)].copy()
y3["dEC"] = y3.ec_mean - y3.prev_ec_mean
y3["dlnEC"] = np.log(y3.ec_mean) - np.log(y3.prev_ec_mean)
y3["dlnEC25"] = np.log(y3.ec25_mean.values) - np.log(R.loc[y3.prev_rec, "ec25_mean"].values)
print(f"[C] sets: Y1 hi10 n={len(hiS)}, Y2 hi10 with OOF n={len(hiR)} (chains {len(np.unique(CH))}), all with OOF n={len(alR)}, "
      f"Y3 hi10->hi10 links n={len(y3)}")

OTM = OT.mean(1)
rows = []
for v in VARS:
    x1 = V.loc[hiS.index, v].values; x2 = V.loc[hiR.index, v].values; xa = V.loc[alR.index, v].values
    r1, s1, n1 = pr(x1, hiS.ec_mean.values)
    r1l, _, _ = pr(x1, np.log(hiS.ec25_mean.values))
    r2, s2, n2 = pr(x2, hiR.Y2.values)
    r2l, _, _ = pr(x2, hiR.Y2log.values)
    ci = boot_ci(x2, hiR.Y2.values, CH)
    pp = perm_p(x2, hiR.Y2.values) if n2 >= 5 and np.nanstd(x2) > 0 else np.nan
    b0, b1, gain, nimp, ncv = cv_gain(hiR.Y2.values, hiR.sg_mean.values, x2, CH)
    pc = partial_r(x2, hiR.Y2.values, OTM[[REC.index(r) for r in hiR.index]])   # control: outdoor temp day mean (season)
    ra, sa, na = pr(xa, alR.Y2.values)
    cia = boot_ci(xa, alR.Y2.values, LC_CODE[[REC.index(r) for r in alR.index]], B=1000)
    a0, a1, again, animp, ancv = cv_gain(alR.Y2.values, alR.sg_mean.values, xa, LC_CODE[[REC.index(r) for r in alR.index]])
    # pass consistency
    p1 = hiR.pass2.values == 0
    rp1, _, np1 = pr(x2[p1], hiR.Y2.values[p1]); rp2, _, np2 = pr(x2[~p1], hiR.Y2.values[~p1])
    pa1 = alR.pass2.values == 0
    rap1, _, _ = pr(xa[pa1], alR.Y2.values[pa1]); rap2, _, _ = pr(xa[~pa1], alR.Y2.values[~pa1])
    # Y3
    xb = V.loc[y3.index, v].values; xp = V.loc[y3.prev_rec, v].values
    r3d, _, n3 = pr(xb - xp, y3.dEC.values); r3l, _, _ = pr(xb, y3.dEC.values)
    # leave-big-chain-out sensitivity (N3)
    keep = CH != 1
    r2x, _, _ = pr(x2[keep], hiR.Y2.values[keep])
    keep2 = hiR.index != "F47_132"
    r2y, _, _ = pr(x2[keep2], hiR.Y2.values[keep2])
    rows.append(dict(var=v, kind="usable" if v in USABLE else "diagnostic", n1=n1, r_Y1=r1, rho_Y1=s1, r_lnEC25=r1l,
                     n2=n2, r_Y2=r2, rho_Y2=s2, r_Y2log=r2l, ci_lo=ci[0], ci_hi=ci[1], perm_p=pp,
                     r2cv_base=b0, r2cv_var=b1, cv_gain=gain, n_improve=nimp, n_cv=ncv, pr_ctrl_out=pc,
                     r_Y2_noep1=r2x, r_Y2_noF47_132=r2y,
                     r_all=ra, rho_all=sa, n_all=na, ci_all_lo=cia[0], ci_all_hi=cia[1], cv_gain_all=again, n_imp_all=animp, n_cv_all=ancv,
                     r_hi_p1=rp1, n_hi_p1=np1, r_hi_p2=rp2, n_hi_p2=np2, r_all_p1=rap1, r_all_p2=rap2,
                     n3=n3, r_Y3_delta=r3d, r_Y3_level=r3l))
T = pd.DataFrame(rows)
# BH over the family (Y2 hi10 permutation p)
p = T.perm_p.values; o = np.argsort(p); q = np.empty_like(p); mtot = np.isfinite(p).sum()
cm = np.minimum.accumulate((p[o] * mtot / np.arange(1, len(p) + 1))[::-1])[::-1]; q[o] = np.minimum(cm, 1)
T["bh_q"] = q
def verdict(r):
    ci_ex = (r.ci_lo > 0) or (r.ci_hi < 0)
    if abs(r.r_Y2) >= .35 and ci_ex and r.cv_gain > 0: return "signal"
    if (abs(r.r_Y2) >= .2 and (ci_ex or r.cv_gain > 0)) or (ci_ex and r.cv_gain > 0): return "weak"
    return "none"
T["verdict"] = T.apply(verdict, axis=1)
T.to_csv(os.path.join(HERE, "stats_v1.csv"), index=False, encoding="utf-8-sig")
print("\n[C] per-variable results (Y2 = day-mean residual y - sg among hi10 with OOF; CI = chain bootstrap; cv = leave-one-chain-out)")
cols = ["var", "kind", "n1", "r_Y1", "r_lnEC25", "n2", "r_Y2", "rho_Y2", "ci_lo", "ci_hi", "perm_p", "bh_q", "cv_gain", "n_improve",
        "pr_ctrl_out", "r_Y2_noep1", "r_Y2_noF47_132", "r_hi_p1", "r_hi_p2", "verdict"]
print(T[cols].round(3).to_string(index=False))
print("\n[C2] all records (n 360) and Y3 (day-to-day change hi10->hi10, descriptive: link uses labels)")
cols2 = ["var", "r_all", "rho_all", "ci_all_lo", "ci_all_hi", "cv_gain_all", "n_imp_all", "r_all_p1", "r_all_p2", "n3", "r_Y3_delta", "r_Y3_level"]
print(T[cols2].round(3).to_string(index=False))
print(f"Bonferroni threshold .05/{mtot} = {.05/mtot:.4f}; min perm p {np.nanmin(p):.4f}; min BH q {np.nanmin(q):.3f}")
print(f"baseline R2cv (Y2 ~ sg, LOCO, hi10) = {T.r2cv_base.iloc[0]:.3f}")

# ---------------- (D) usable variables at earlier hours (0..h cumulative) vs hourly residual ----------------
print("\n[D] usable cumulative 0..h versions vs hour-h residual (ec_h - sg_h) among hi10 with OOF (n 31) / all (n 360)")
ih = [REC.index(r) for r in hiR.index]; ia = [REC.index(r) for r in alR.index]
Drows = []
for h in (0, 6, 12, 18, 23):
    Uh = usable_vars(h)
    resh = EC[:, h] - SG[:, h]
    for v in USABLE:
        rh, _, _ = pr(Uh[v][ih], resh[ih]); ra_, _, _ = pr(Uh[v][ia], resh[ia])
        Drows.append(dict(h=h, var=v, r_hi=rh, r_all=ra_))
DT = pd.DataFrame(Drows)
DTw = DT.pivot(index="var", columns="h", values="r_hi").loc[USABLE]
DTa = DT.pivot(index="var", columns="h", values="r_all").loc[USABLE]
print("r with hourly residual, hi10:"); print(DTw.round(3).to_string())
print("r with hourly residual, all:"); print(DTa.round(3).to_string())
DT.to_csv(os.path.join(HERE, "hourly_cum_v1.csv"), index=False, encoding="utf-8-sig")

# ---------------- (E) sensor-effect fingerprints (P1) ----------------
print("\n[E] sensor-effect fingerprints")
def slope_ci(x, y, groups, B=2000):
    m = np.isfinite(x) & np.isfinite(y); x, y, g = x[m], y[m], groups[m]
    s0 = stats.linregress(x, y).slope; ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; ss = []
    for _ in range(B):
        s = RNG.choice(ug, len(ug), replace=True); ii = np.concatenate([idx[u] for u in s])
        if np.std(x[ii]) > 0: ss.append(stats.linregress(x[ii], y[ii]).slope)
    return s0, np.percentile(ss, [2.5, 97.5]), int(m.sum())
s, ci, n = slope_ci(hiS.st_mean.values, np.log(hiS.ec_mean.values), hiS.ep_id.values.astype(int))
print(f"E1 between hi10 records: d ln(EC day mean)/d(sub_temp day mean) = {s:+.4f}/C  CI[{ci[0]:+.4f},{ci[1]:+.4f}] n={n} (sensor prediction +.02)")
s, ci, n = slope_ci(D["lnfac_gap"][[REC.index(r) for r in hiR.index]], hiR.Y2log.values, CH)
print(f"E2 hi10: ln residual ~ lnfac(true T) - lnfac(pred T): slope {s:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] n={n} (sensor prediction 1.0)")
s, ci, n = slope_ci(D["lnfac_gap"][ia], alR.Y2log.values, LC_CODE[ia], B=1000)
print(f"E2b all: same slope {s:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] n={n}")
dst = V.loc[y3.index, "st_mean"].values - V.loc[y3.prev_rec, "st_mean"].values
g3 = y3.ep_id.values.astype(int)
s, ci, n = slope_ci(dst, y3.dlnEC.values, g3)
print(f"E3 hi10 links: d(dlnEC)/d(dST) = {s:+.4f}/C CI[{ci[0]:+.4f},{ci[1]:+.4f}] n={n} (sensor prediction +.02); "
      f"SD dST {np.std(dst):.2f}C, SD dlnEC {np.std(y3.dlnEC):.3f}")
s, ci, n = slope_ci(dst, y3.dlnEC25.values, g3)
print(f"E3b same with ln EC25: {s:+.4f}/C CI[{ci[0]:+.4f},{ci[1]:+.4f}] (sensor-only prediction 0)")
# all clean links (any level) for contrast
al3 = R[R.prev_rec.notna()].copy()
dst_a = V.loc[al3.index, "st_mean"].values - V.loc[al3.prev_rec, "st_mean"].values
dln_a = np.log(al3.ec_mean.values) - np.log(al3.prev_ec_mean.values)
s, ci, n = slope_ci(dst_a, dln_a, LC_CODE[[REC.index(r) for r in al3.index]], B=1000)
print(f"E3c all clean links: d(dlnEC)/d(dST) = {s:+.4f}/C CI[{ci[0]:+.4f},{ci[1]:+.4f}] n={n}")
# within-record hourly coupling: pooled, record-demeaned
def within(A, rows_):
    B = A[rows_]; return (B - np.nanmean(B, 1, keepdims=True)).reshape(-1)
for lab, rows_, grp in (("hi10", ih, CH), ("normal", [i for i in ia if R.hi10.values[i] == 0], None)):
    xw = within(ST, rows_); yw = within(lnEC, rows_)
    sl = stats.linregress(xw, yw)
    xr = within(ST - PST, rows_); yr = within(lnEC - np.log(SG), rows_)
    sr = stats.linregress(xr[np.isfinite(yr)], yr[np.isfinite(yr)])
    print(f"E4 within-record hourly ({lab}, {len(rows_)} recs): dlnEC/dST {sl.slope:+.4f} (r {sl.rvalue:.2f}); "
          f"ln residual vs (ST - predicted ST): {sr.slope:+.4f} (r {sr.rvalue:.2f})")
# median per-record gamma by group
print(f"E5 per-record gamma median: hi10 {np.nanmedian(gam[hi]):+.4f}, normal {np.nanmedian(gam[~hi]):+.4f}; corr median hi10 {np.nanmedian(cor[hi]):.2f}, normal {np.nanmedian(cor[~hi]):.2f}")

# ---------------- (F) compensated level: is EC25 easier to explain among hi10? ----------------
print("\n[F] compensated level (EC25 = EC/(1+.02(T-25)), true sub_temp -> diagnostic)")
for lab, S in (("hi10", hiS), ("all", R)):
    a = np.log(S.ec_mean.values); b = np.log(S.ec25_mean.values)
    print(f"F1 {lab} n={len(S)}: SD ln EC {a.std():.4f}, SD ln EC25 {b.std():.4f}, ratio var {b.var()/a.var():.3f}, corr {np.corrcoef(a,b)[0,1]:.3f}")
a = np.log(hiR.ec_mean.values); b = np.log(hiR.ec25_mean.values)
print(f"F2 hi10 with OOF: corr(ln sg, ln EC) {np.corrcoef(np.log(hiR.sg_mean), a)[0,1]:.3f} vs corr(ln sg25(pred T), ln EC25(true T)) "
      f"{np.corrcoef(np.log(hiR.sg25_mean), b)[0,1]:.3f}; SD Y2log {hiR.Y2log.std():.4f} vs SD Y2c {hiR.Y2c.std():.4f}")
INP = [c for c in R.columns if c.endswith("_d") or c.endswith("_n")]
def loco_ridge_r2(Xm, y, g):
    pred = np.zeros(len(y))
    for u in np.unique(g):
        te = g == u
        mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))
        mdl.fit(Xm[~te], y[~te]); pred[te] = mdl.predict(Xm[te])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
Xi = hiS[INP].values; g1 = hiS.ep_id.values.astype(int)
print(f"F3 hi10 n={len(hiS)}: LOCO ridge on {len(INP)} day/night input means -> R2cv ln EC {loco_ridge_r2(Xi, np.log(hiS.ec_mean.values), g1):+.3f}, "
      f"ln EC25 {loco_ridge_r2(Xi, np.log(hiS.ec25_mean.values), g1):+.3f}")

# ---------------- (G) house fingerprint (P4): root-zone thermal behaviour high vs normal in the same calendar window ----------------
print("\n[G] P4 house fingerprint: hi10 vs normal records in the same DM11 window (descriptive; AUC of variable for hi10)")
win = R[(R.dm11_day >= R.loc[hiS.index, "dm11_day"].min()) & (R.dm11_day <= R.loc[hiS.index, "dm11_day"].max())]
print(f"window DM11 {win.dm11_day.min():.0f}-{win.dm11_day.max():.0f}: n {len(win)}, hi10 {int(win.hi10.sum())}")
from sklearn.metrics import roc_auc_score
grows = []
for v in VARS:
    x = V.loc[win.index, v].values; m = np.isfinite(x)
    auc = roc_auc_score(win.hi10.values[m], x[m]) if len(np.unique(win.hi10.values[m])) == 2 else np.nan
    grows.append(dict(var=v, auc_hi=auc, mean_hi=np.nanmean(x[win.hi10.values == 1]), mean_norm=np.nanmean(x[win.hi10.values == 0])))
G = pd.DataFrame(grows)
print(G.round(3).to_string(index=False))

# ---------------- (H) multivariate non-causal upper bound ----------------
print("\n[H] multivariate upper bound: family variables (+ sg_mean) -> Y2, leave-one-chain-out (hi10) / leave-one-component-out folds (all)")
def mv_cv(Xm, y, base, g, kind, seeds=(0, 1, 2)):
    pred = np.zeros(len(y)); pb = np.zeros(len(y))
    for u in np.unique(g):
        te = g == u; tr = ~te
        pb[te] = ols_fit_pred(base[tr, None], y[tr], base[te, None])
        XX = np.c_[base, Xm]
        if kind == "ridge":
            mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))
            mdl.fit(XX[tr], y[tr]); pred[te] = mdl.predict(XX[te])
        else:
            ps = []
            for s_ in seeds:
                mdl = make_pipeline(SimpleImputer(strategy="median"),
                                    GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=s_))
                mdl.fit(XX[tr], y[tr]); ps.append(mdl.predict(XX[te]))
            pred[te] = np.mean(ps, 0)
    sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pred) ** 2) / sst, 1 - np.sum((y - pb) ** 2) / sst, int(((y - pred) ** 2 < (y - pb) ** 2).sum()), pred, pb
mv_rows = []
yh = hiR.Y2.values; bh = hiR.sg_mean.values
ga = LC_CODE[ia]; gfold = np.zeros(len(ia), int)
for k, (tr, te) in enumerate(GroupKFold(10).split(ia, groups=ga)): gfold[te] = k
for setname, vs in (("all39", VARS), ("usable", USABLE), ("diagnostic", DIAG)):
    Xh = V.loc[hiR.index, vs].values
    for kind in ("ridge", "gbm"):
        r2, r2b, nimp, pred, pb = mv_cv(Xh, yh, bh, CH, kind)
        P = 200 if kind == "ridge" else 60
        null = []
        for _ in range(P):
            yp = RNG.permutation(yh); null.append(mv_cv(Xh, yp, bh, CH, kind, seeds=(0,))[0])
        null = np.array(null)
        p1m = hiR.pass2.values == 1
        mv_rows.append(dict(set=setname, kind=kind, scope="hi10", n=len(yh), r2cv=r2, r2cv_base=r2b, gain=r2 - r2b, n_improve=nimp,
                            null_r2_95=np.percentile(null, 95), p_null=(1 + (null >= r2).sum()) / (P + 1),
                            pass2_improve=int(((yh - pred) ** 2 < (yh - pb) ** 2)[p1m].sum()), pass2_n=int(p1m.sum())))
        Xa = V.loc[alR.index, vs].values
        r2a, r2ba, nimpa, preda, pba = mv_cv(Xa, alR.Y2.values, alR.sg_mean.values, gfold, kind, seeds=(0,))
        hm = alR.hi10.values == 1
        mv_rows.append(dict(set=setname, kind=kind, scope="all", n=len(ia), r2cv=r2a, r2cv_base=r2ba, gain=r2a - r2ba, n_improve=nimpa,
                            hi_improve=int(((alR.Y2.values - preda) ** 2 < (alR.Y2.values - pba) ** 2)[hm].sum()), hi_n=int(hm.sum())))
MV = pd.DataFrame(mv_rows)
print(MV.round(3).to_string(index=False))
MV.to_csv(os.path.join(HERE, "multivar_v1.csv"), index=False, encoding="utf-8-sig")
print("\nsaved:", os.path.join(HERE, "vars_v1.csv"), "| stats_v1.csv | hourly_cum_v1.csv | multivar_v1.csv")
