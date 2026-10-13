# -*- coding: utf-8 -*-
"""HX9 / history family v1 (2026-10-13 집 클로드).
사용자 요청: '고EC 크기 예측에 인과관계가 조금이라도 있는 데이터 전부 확인'. 이 스크립트 = '여러 날 누적 이력(달력 경유)' 가족.

기전 가설 (일반 농업 지식 — 이 자료에서 확인한 것이 아님):
  배지 EC 는 '급액으로 들어온 염 - 배액/흡수로 나간 염' 이 여러 날 적분된 결과다. 일사가 적은 날이 이어지면
  일사 비례 급액량이 줄어 배액(세척)이 줄고 염이 쌓인다 / 반대로 증산 수요(일사·VPD)가 큰 날이 이어지면
  물만 빠져 농축된다. 어느 쪽이든 '이전 2~7일 누적 기상·밀폐 일수'가 고EC '크기'를 정해야 한다.

사전 예측한 흔적 (계산 전에 고정, 2026-10-13):
  P1. 고EC 기록 안에서 하루 평균 EC(Y1)가 이전 k일 누적 일사·ET0·VPD·밀폐 일수와 |r| >= .35 로 같은 방향.
      적분 기전이면 k=5~7 창이 k=1 보다 강해야 한다.
  P2. 현재 모델 잔차(Y2 = y - sg)도 같은 변수와 상관 (모델은 당일 입력만 보므로 여러 날 정보가 남아 있어야 함).
      사슬 부트스트랩 CI 가 0 을 빼고 LOCO 교차 R2 이득 > 0.
  P3. 에피소드 안 하루→다음날 변화(Y3 = EC(b) - EC(전날))가 전날(k=1) 기상 강제력과 상관 (적분기라면 '변화 ∝ 강제력').
  P4. 날짜 수준 변수(외기 이력)는 같은 날 모든 동에 똑같으므로, 기전이 '날씨 이력'이라면 같은 날짜의 고EC 기록끼리
      크기가 비슷해야 하고, 같은 날짜의 다른 동도 높아야 한다.
  P5. 계절(DM11 날짜)을 통제해도 P1/P2 상관이 남아야 한다.
  P6. 1차(num<179)와 2차(num>=179)에서 부호가 같아야 한다.
보이면 안 되는 것 (보이면 기전 기각 또는 약화):
  N1. 같은 날짜에 고EC 동과 보통 동이 함께 흔함 (날씨 이력으로는 누가 높은지 못 가름 → 날씨는 '조건' 이지 '크기' 아님).
  N2. 같은 날짜의 고EC 기록끼리 크기 차이가 날짜 간 차이만큼 큼 (날짜 수준 변수의 설명 상한이 낮음).
  N3. 상관이 날짜를 몇 일 옮긴 위약(placebo) 변수에서도 똑같이 나옴 (자기상관·계절 추세의 산물).
  N4. 상관이 계절 통제 후 사라짐.

변수 사용 가능성 표시:
  U    = 평가 행 h 에서 계산 가능: 학습 전용 DMT3e 달력 + 평가 기록 자기 0..h 실외를 학습 쌍둥이와 맞춰 날짜를 찾고,
         그 이전 날짜의 '학습 기록' 실외만 씀 (h=0 쌍둥이 일치 375/400).
  U23  = 당일 24시간 입력 필요 (h 까지의 부분합은 U).
  D    = 진단 전용: DM11 달력(평가 입력으로 구간 연결) 또는 clean 사슬(정답 연속성으로 고른 전날) 사용.
         사슬 변수는 '입력만으로 같은 동 전날을 찾는 연결기'로 바꾸면 U 가 될 수 있음(연결 정확도 ~78% 선행 결과).

산출: hx9/history/vars_v1.csv (rec 키), hx9/history/stats_v1.csv, hx9/history/hist_v1.log (stdout)
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
HX9 = os.path.dirname(HERE); RES = os.path.dirname(HX9)
sys.path.insert(0, os.path.join(RES, "linker_v4"))
from feat4 import X as XALL, FARM, NUM, N  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

RNG = np.random.default_rng(20261013)
NB = 2000
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
H = pd.read_csv(os.path.join(HX9, "data", "hourly.csv"))
C11 = pd.read_csv(os.path.join(RES, "ec0h", "rise", "dm11_calendar_v1.csv"))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"

# ------------------------------------------------------------------ hourly physics helpers (FAO-56, general knowledge)
def es(T): return 0.6108 * np.exp(17.27 * T / (T + 237.3))
def daily_weather(out):  # out [24,4] = out_temp, out_hum, out_rad, out_wspd
    T, RH, Rs, u = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    vpd = es(T) * (1 - RH / 100.0)
    Rn = 0.77 * Rs * 0.0036                                     # MJ m-2 h-1, net shortwave only
    D = 4098 * es(T) / (T + 237.3) ** 2; g = 0.0665
    et0 = (0.408 * D * Rn + g * 37 / (T + 273) * u * vpd) / (D + g * (1 + 0.34 * u))
    return dict(rad=np.nansum(Rs) * 0.0036, et0=np.nansum(np.clip(et0, 0, None)), vpd=np.nanmean(vpd[9:17]),
                tmean=np.nanmean(T), tmin=np.nanmin(T), hum=np.nanmean(RH))

# ------------------------------------------------------------------ date tables
# (a) DMT3e training-only calendar: (seg, day) -> weather from TRAINING records on that date (identical across houses; checked 118/118)
HO = {r: g.sort_values("hour")[["out_temp", "out_hum", "out_rad", "out_wspd"]].values for r, g in H.groupby("rec")}
dw = {r: daily_weather(HO[r]) for r in R.index}
DW = pd.DataFrame.from_dict(dw, orient="index"); DW.index.name = "rec"
WV = list(DW.columns)
T_dmt = DW.join(R[["dmt_seg", "dmt_day"]]).groupby(["dmt_seg", "dmt_day"])[WV].mean()
# (b) DM11 calendar (descriptive): day -> weather from ALL records (train+eval) on that day
dw11 = []
for r, d in zip(C11.rec, C11.day):
    i = [k for k in range(N) if rid(k) == r][0]
    dw11.append(dict(day=d, **daily_weather(XALL[i])))
T_11 = pd.DataFrame(dw11).groupby("day")[WV].mean()
T_11 = T_11.reindex(np.arange(1, 127))
P75 = np.nanpercentile(T_dmt.rad.values, 75); P40 = np.nanpercentile(T_dmt.rad.values, 40)

def date_vars(get):
    """get(k) -> weather dict-like (Series) of the date k days before the record (k>=1), or None if unavailable; get(0) = same date."""
    def win(col, k, need):
        v = [get(j)[col] for j in range(1, k + 1) if get(j) is not None]
        return np.mean(v) if len(v) >= need else np.nan
    o = {}
    for k, need in ((1, 1), (3, 2), (5, 3), (7, 4)):
        o[f"rad_p{k}"] = win("rad", k, need)
    for k, need in ((1, 1), (3, 2), (7, 4)):
        o[f"et0_p{k}"] = win("et0", k, need); o[f"vpd_p{k}"] = win("vpd", k, need); o[f"tout_p{k}"] = win("tmean", k, need)
    o["tmin_p3"] = win("tmin", 3, 2); o["hum_p3"] = win("hum", 3, 2)
    o["rad_p14"] = win("rad", 14, 7)
    tm = [(j, get(j)["tmean"]) for j in range(1, 6) if get(j) is not None]
    o["ttrend5"] = (-np.polyfit([a for a, _ in tm], [b for _, b in tm], 1)[0]) if len(tm) >= 3 else np.nan  # deg/day, + = warming
    t25 = [get(j)["tmean"] for j in range(2, 6) if get(j) is not None]
    o["tchg1"] = (get(1)["tmean"] - np.mean(t25)) if (get(1) is not None and len(t25) >= 2) else np.nan
    r25 = [get(j)["rad"] for j in range(2, 6) if get(j) is not None]
    o["radchg1"] = (get(1)["rad"] - np.mean(r25)) if (get(1) is not None and len(r25) >= 2) else np.nan
    # days since last high-radiation day (rad >= P75) within 10 days; censored if not found
    avail = 0; ds = np.nan
    for j in range(1, 11):
        g = get(j)
        if g is None: break
        avail += 1
        if g["rad"] >= P75: ds = j; break
    if not np.isfinite(ds) and avail >= 5: ds = avail + 1
    o["dsince_hirad"] = ds
    run = 0; brk = False
    for j in range(1, 11):
        g = get(j)
        if g is None: break
        if g["rad"] < P40: run += 1
        else: brk = True; break
    o["lowrad_run"] = run if (brk or run >= 3) else np.nan
    g0 = get(0)
    o["rad_d0"] = g0["rad"] if g0 is not None else np.nan; o["et0_d0"] = g0["et0"] if g0 is not None else np.nan
    return o

VD = {}
for r in R.index:
    s, d = R.at[r, "dmt_seg"], R.at[r, "dmt_day"]
    def g_dmt(k, s=s, d=d):
        key = (s, d - k)
        return T_dmt.loc[key] if key in T_dmt.index else None
    d11 = R.at[r, "dm11_day"]
    def g_11(k, d11=d11):
        if not np.isfinite(d11): return None
        dd = int(d11 - k)
        if dd < 1 or dd > 126 or not np.isfinite(T_11.loc[dd, "rad"]): return None
        return T_11.loc[dd]
    a = date_vars(g_dmt); b = date_vars(g_11)
    VD[r] = {**a, **{k + "_dm11": v for k, v in b.items()}}
V = pd.DataFrame.from_dict(VD, orient="index"); V.index.name = "rec"
V["season_dm11"] = R.dm11_day

# ------------------------------------------------------------------ house-specific history along the clean predecessor chain (D)
Hh = H.copy()
Hh["in_vpd"] = es(Hh.in_temp) * (1 - Hh.in_hum / 100.0)
Hh["trans"] = Hh.out_rad * 0.0036 * (1 - Hh.act_shade.fillna(0) / 100.0) * Hh.in_vpd   # MJ kPa proxy (no in_rad available)
hd = Hh.groupby("rec").agg(fan=("act_circfan", "mean"), v0=("act_vent", lambda s: (s == 0).mean()), heat=("act_heating", "mean"),
                           vent=("act_vent", "mean"), co2=("act_co2", "mean"), intemp=("in_temp", "mean"), trans=("trans", "sum"))
hd["invpd"] = Hh[(Hh.hour >= 9) & (Hh.hour <= 16)].groupby("rec").in_vpd.mean()
hd["sealed"] = ((hd.fan < 10) & (hd.v0 > 0.85)).astype(float)
PREV = R.prev_rec.dropna().to_dict()
CH = {}
for r in R.index:
    ch = []; x = r
    while x in PREV and len(ch) < 7:
        x = PREV[x]; ch.append(x)
    CH[r] = ch
def chain_vars(r):
    ch = CH[r]; o = {"chain_n": len(ch)}
    if not ch:
        return {**o, **{k: np.nan for k in ("sealed_run", "sealed_p3", "sealed_p7", "heat_p3", "vent_p3", "invpd_p3", "trans_p1",
                                              "trans_p3", "co2_p3", "intemp_p3")}}
    sl = [hd.at[x, "sealed"] for x in ch]
    run = 0
    for v in sl:
        if v == 1: run += 1
        else: break
    o["sealed_run"] = run if (run < len(sl) or len(sl) >= 3) else np.nan
    c3 = ch[:3]
    o["sealed_p3"] = np.mean(sl[:3]) if len(c3) >= 2 else np.nan
    o["sealed_p7"] = np.mean(sl) if len(ch) >= 4 else np.nan
    for col, nm in (("heat", "heat_p3"), ("vent", "vent_p3"), ("invpd", "invpd_p3"), ("trans", "trans_p3"), ("co2", "co2_p3"), ("intemp", "intemp_p3")):
        o[nm] = hd.loc[c3, col].mean() if len(c3) >= 2 else np.nan
    o["trans_p1"] = hd.at[ch[0], "trans"]
    return o
V = V.join(pd.DataFrame.from_dict({r: chain_vars(r) for r in R.index}, orient="index"))

PRIMARY_U = ["rad_p1", "rad_p3", "rad_p5", "rad_p7", "rad_p14", "et0_p1", "et0_p3", "et0_p7", "vpd_p1", "vpd_p3", "vpd_p7",
             "tout_p1", "tout_p3", "tout_p7", "tmin_p3", "hum_p3", "ttrend5", "tchg1", "radchg1", "dsince_hirad", "lowrad_run"]
U23 = ["rad_d0", "et0_d0"]
DM11V = [c + "_dm11" for c in PRIMARY_U]
SEASON = ["season_dm11"]
CHAIN = ["chain_n", "sealed_run", "sealed_p3", "sealed_p7", "heat_p3", "vent_p3", "invpd_p3", "trans_p1", "trans_p3", "co2_p3", "intemp_p3"]
USABLE = {**{c: "usable" for c in PRIMARY_U}, **{c: "usable(h=23; partial sums usable)" for c in U23},
          **{c: "diagnostic only (DM11 calendar)" for c in DM11V + SEASON},
          **{c: "diagnostic only (clean chain = label continuity; usable if an input-only linker replaces it)" for c in CHAIN}}
ALLV = PRIMARY_U + U23 + SEASON + CHAIN + DM11V
V = V[ALLV]
V.to_csv(os.path.join(HERE, "vars_v1.csv"), encoding="utf-8")

# ------------------------------------------------------------------ targets
D = R.join(V)
D["Y3"] = D.ec_mean - D.prev_ec_mean
hiA = D[D.hi10 == 1]                                  # Y1 (n 35)
hiS = D[(D.hi10 == 1) & (D.has_sg == 1)]              # Y2h (n 31)
allS = D[D.has_sg == 1].copy()                        # Y2a (n 360)
allS["clu"] = [f"d{int(x)}" if np.isfinite(x) else r for r, x in zip(allS.index, allS.dm11_day)]
hi3 = D[(D.hi10 == 1) & D.prev_ec_mean.notna()]      # Y3 (n 31)
prd = D[(D.has_sg == 1) & (D.sg_mean >= 0.9)]          # Y2p predicted-high subset
print(f"n: Y1 hi10 {len(hiA)} | Y2h hi10&OOF {len(hiS)} | Y2a all OOF {len(allS)} | Y3 hi10&clean-prev {len(hi3)} | Y2p sg>=0.9 {len(prd)} "
      f"(of which normal {int((prd.hi10==0).sum())}) | chains(ep_id) in Y2h {hiS.ep_id.nunique()}")

def pr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) == 0: return np.nan, np.nan, np.nan, int(m.sum())
    r, p = stats.pearsonr(x[m], y[m]); rho = stats.spearmanr(x[m], y[m])[0]
    return r, p, rho, int(m.sum())

def boot_ci(df, v, y, grp):
    sub = df[[v, y, grp]].dropna()
    if len(sub) < 6: return np.nan, np.nan
    gs = sub[grp].unique(); idx = {g: np.where(sub[grp].values == g)[0] for g in gs}
    xv, yv = sub[v].values, sub[y].values; out = []
    for _ in range(NB):
        pick = RNG.choice(gs, len(gs), replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        if len(ii) < 5 or np.std(xv[ii]) == 0 or np.std(yv[ii]) == 0: continue
        out.append(np.corrcoef(xv[ii], yv[ii])[0, 1])
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if len(out) > 100 else (np.nan, np.nan)

def ols_pred(Xtr, ytr, Xte):
    A = np.c_[np.ones(len(Xtr)), Xtr]; B = np.c_[np.ones(len(Xte)), Xte]
    w = np.linalg.lstsq(A, ytr, rcond=None)[0]; return B @ w

def cv_gain(df, v, y, base_cols, grp, kfold=None):
    sub = df[[v, y, grp] + base_cols].dropna()
    if len(sub) < 8 or sub[grp].nunique() < 3: return np.nan, np.nan, 0
    yv = sub[y].values; g = sub[grp].values
    pb = np.zeros(len(sub)); pf = np.zeros(len(sub))
    if kfold:
        splits = list(GroupKFold(n_splits=min(kfold, len(np.unique(g)))).split(sub, yv, g))
    else:
        splits = [(np.where(g != k)[0], np.where(g == k)[0]) for k in np.unique(g)]
    for tr, te in splits:
        xb = sub[base_cols].values; xf = sub[base_cols + [v]].values
        mu, sd = xf[tr].mean(0), xf[tr].std(0) + 1e-9
        pb[te] = ols_pred((xb[tr] - mu[:len(base_cols)]) / sd[:len(base_cols)], yv[tr], (xb[te] - mu[:len(base_cols)]) / sd[:len(base_cols)])
        pf[te] = ols_pred((xf[tr] - mu) / sd, yv[tr], (xf[te] - mu) / sd)
    sst = np.sum((yv - yv.mean()) ** 2)
    r2b = 1 - np.sum((yv - pb) ** 2) / sst; r2f = 1 - np.sum((yv - pf) ** 2) / sst
    return r2f - r2b, r2b, int(np.sum(np.abs(yv - pf) < np.abs(yv - pb)))

def partial_r(df, v, y, ctl):
    sub = df[[v, y, ctl]].dropna()
    if len(sub) < 6: return np.nan
    def res(a):
        A = np.c_[np.ones(len(sub)), sub[ctl].values]; return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return np.corrcoef(res(sub[v].values), res(sub[y].values))[0, 1]

rows = []
for v in ALLV:
    r1, p1, s1, n1 = pr(hiA[v].values, hiA.ec_mean.values)
    r2, p2, s2, n2 = pr(hiS[v].values, hiS.resid_mean.values)
    ra, pa, sa, na = pr(allS[v].values, allS.resid_mean.values)
    r3, p3, s3, n3 = pr(hi3[v].values, hi3.Y3.values)
    rp, pp, sp, npp = pr(prd[v].values, prd.resid_mean.values)
    lo1, hi1 = boot_ci(hiA, v, "ec_mean", "ep_id")
    lo2, hi2 = boot_ci(hiS, v, "resid_mean", "ep_id")
    loa, hia = boot_ci(allS, v, "resid_mean", "clu")
    lo3, hi3c = boot_ci(hi3, v, "Y3", "ep_id")
    g2, b2, imp2 = cv_gain(hiS, v, "resid_mean", ["sg_mean"], "ep_id")
    ga, ba, impa = cv_gain(allS, v, "resid_mean", ["sg_mean"], "clu", kfold=10)
    g3, b3, imp3 = cv_gain(hi3, v, "Y3", ["prev_ec_mean"], "ep_id")
    s1p = hiS[hiS.pass2 == 0]; s2p = hiS[hiS.pass2 == 1]
    a1p = allS[allS.pass2 == 0]; a2p = allS[allS.pass2 == 1]
    rows.append(dict(variable=v, usable=USABLE[v],
        n_Y1=n1, r_Y1=r1, rho_Y1=s1, ci_Y1=f"[{lo1:+.2f},{hi1:+.2f}]",
        n_Y2h=n2, r_Y2h=r2, rho_Y2h=s2, p_Y2h=p2, ci_Y2h=f"[{lo2:+.2f},{hi2:+.2f}]", ci_lo_Y2h=lo2, ci_hi_Y2h=hi2,
        cvgain_Y2h=g2, cvbase_Y2h=b2, nimp_Y2h=imp2, rpart_season_Y2h=partial_r(hiS, v, "resid_mean", "dm11_day") if v != "season_dm11" else np.nan,
        r_Y2h_pass1=pr(s1p[v].values, s1p.resid_mean.values)[0], r_Y2h_pass2=pr(s2p[v].values, s2p.resid_mean.values)[0],
        n_Y2a=na, r_Y2a=ra, p_Y2a=pa, ci_Y2a=f"[{loa:+.2f},{hia:+.2f}]", cvgain_Y2a=ga, nimp_Y2a=impa,
        r_Y2a_pass1=pr(a1p[v].values, a1p.resid_mean.values)[0], r_Y2a_pass2=pr(a2p[v].values, a2p.resid_mean.values)[0],
        r_Y2a_F13=pr(allS[allS.farm == "F13"][v].values, allS[allS.farm == "F13"].resid_mean.values)[0],
        r_Y2a_F47=pr(allS[allS.farm == "F47"][v].values, allS[allS.farm == "F47"].resid_mean.values)[0],
        n_Y3=n3, r_Y3=r3, ci_Y3=f"[{lo3:+.2f},{hi3c:+.2f}]", cvgain_Y3=g3, nimp_Y3=imp3,
        n_Y2p=npp, r_Y2p=rp))
S = pd.DataFrame(rows)
def verdict(x):
    if not np.isfinite(x.r_Y2h): return "none"
    ci_ex = np.isfinite(x.ci_lo_Y2h) and (x.ci_lo_Y2h > 0 or x.ci_hi_Y2h < 0)
    if abs(x.r_Y2h) >= .35 and ci_ex and x.cvgain_Y2h > 0: return "signal"
    if (abs(x.r_Y2h) >= .25 and (ci_ex or x.cvgain_Y2h > 0)) or (ci_ex and x.cvgain_Y2h > 0): return "weak"
    return "none"
S["verdict"] = S.apply(verdict, axis=1)
S.to_csv(os.path.join(HERE, "stats_v1.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
print("\n=== per-variable (Y1 = hi10 day-mean EC; Y2h = resid among hi10; Y2a = resid all; Y3 = day-to-day change in hi10) ===")
print(S[["variable", "n_Y1", "r_Y1", "ci_Y1", "n_Y2h", "r_Y2h", "rho_Y2h", "ci_Y2h", "cvgain_Y2h", "nimp_Y2h", "rpart_season_Y2h",
         "r_Y2h_pass1", "r_Y2h_pass2", "verdict"]].round(3).to_string(index=False))
print()
print(S[["variable", "n_Y2a", "r_Y2a", "ci_Y2a", "cvgain_Y2a", "r_Y2a_pass1", "r_Y2a_pass2", "r_Y2a_F13", "r_Y2a_F47",
         "n_Y3", "r_Y3", "ci_Y3", "cvgain_Y3", "nimp_Y3", "n_Y2p", "r_Y2p"]].round(3).to_string(index=False))
nt = len(ALLV)
print(f"\nvariables tested: {nt} ({len(PRIMARY_U)} U date-level DMT3e, {len(U23)} U23 current-day, {len(SEASON)} season, {len(CHAIN)} chain D, "
      f"{len(DM11V)} DM11 robustness copies) x 4 targets. Bonferroni alpha .05/{nt} = {0.05/nt:.4f}")
print("min p Y2h:", S.loc[S.p_Y2h.idxmin(), ["variable", "p_Y2h"]].tolist(), "| min p Y2a:", S.loc[S.p_Y2a.idxmin(), ["variable", "p_Y2a"]].tolist())
print("verdict counts:", S.verdict.value_counts().to_dict())

# ------------------------------------------------------------------ P4 / N1 / N2: same-date structure among high records
print("\n=== same-date fingerprint (DM11 dates; descriptive) ===")
hh = hiA.copy()
nn = sum(1 for r in hh.index if R.at[r, "twin_n"] > 0)
mixed = sum(1 for r in hh.index if R.at[r, "twin_n"] > R.at[r, "twin_hi10_n"])
print(f"hi10 records with >=1 same-date training twin: {nn}/{len(hh)}; of these with >=1 NORMAL same-date twin: {mixed}")
grp = hh.groupby("dm11_day").ec_mean
multi = grp.filter(lambda s: len(s) >= 2)
tot_var = hh.ec_mean.var(ddof=0)
within = hh.loc[multi.index].groupby("dm11_day").ec_mean.apply(lambda s: s.var(ddof=0)).mean() if len(multi) else np.nan
print(f"hi10 Y1 variance total {tot_var:.4f}; mean within-date variance on dates with >=2 hi10 ({multi.index.size} recs, "
      f"{hh.loc[multi.index].dm11_day.nunique()} dates) {within:.4f} -> date-level ceiling ~ {1-within/tot_var:.2f} of variance")
for d_, s_ in hh.loc[multi.index].groupby("dm11_day"):
    print(f"   DM11 {int(d_)}: " + ", ".join(f"{r} {e:.2f} (ep {int(R.at[r,'ep_id'])}, resid {R.at[r,'resid_mean']:+.2f})" for r, e in s_.ec_mean.items()))
hs2 = hiS.groupby("dm11_day").resid_mean
m2 = hs2.filter(lambda s: len(s) >= 2)
w2 = hiS.loc[m2.index].groupby("dm11_day").resid_mean.apply(lambda s: s.var(ddof=0)).mean()
print(f"Y2h total var {hiS.resid_mean.var(ddof=0):.4f}; within-date {w2:.4f} -> ceiling ~ {1-w2/hiS.resid_mean.var(ddof=0):.2f}")
# all records: date ICC of residual
aa = allS[allS.dm11_day.notna()]
ma = aa.groupby("dm11_day").resid_mean.filter(lambda s: len(s) >= 2)
wa = aa.loc[ma.index].groupby("dm11_day").resid_mean.apply(lambda s: s.var(ddof=0)).mean()
print(f"Y2a total var {aa.resid_mean.var(ddof=0):.4f}; within-date {wa:.4f} -> date-level ceiling ~ {1-wa/aa.resid_mean.var(ddof=0):.2f}")

# ------------------------------------------------------------------ N3: date-shift placebo on DM11 copies (keeps autocorrelation)
print("\n=== date-shift placebo (DM11 calendar copies; weather of date+s days assigned to the record) ===")
T11v = {}
for d_ in range(1, 127):
    def g(k, d_=d_):
        dd = d_ - k
        if dd < 1 or dd > 126 or not np.isfinite(T_11.loc[dd, "rad"]): return None
        return T_11.loc[dd]
    T11v[d_] = date_vars(g)
T11v = pd.DataFrame.from_dict(T11v, orient="index")
SH = [s for s in range(-30, 31) if abs(s) >= 5]
for yname, df, ycol in (("Y1", hiA, "ec_mean"), ("Y2h", hiS, "resid_mean"), ("Y3", hi3, "Y3"), ("Y2a", allS, "resid_mean")):
    dd = df.dm11_day.values; yy = df[ycol].values
    obs = {}; nulls = {v: [] for v in PRIMARY_U}; nullmax = []
    for v in PRIMARY_U:
        xx = np.array([T11v.at[int(d), v] if np.isfinite(d) else np.nan for d in dd]); obs[v] = pr(xx, yy)[0]
    for s in SH:
        mx = 0
        for v in PRIMARY_U:
            xx = np.array([T11v.at[int(d + s), v] if (np.isfinite(d) and 1 <= d + s <= 126) else np.nan for d in dd])
            r_ = pr(xx, yy)[0]
            if np.isfinite(r_): nulls[v].append(abs(r_)); mx = max(mx, abs(r_))
        nullmax.append(mx)
    om = max(abs(x) for x in obs.values() if np.isfinite(x))
    best = max(obs, key=lambda k: abs(obs[k]) if np.isfinite(obs[k]) else -1)
    pv = {v: np.mean(np.array(nulls[v]) >= abs(obs[v])) for v in PRIMARY_U if np.isfinite(obs[v])}
    print(f"{yname}: strongest DM11 date variable {best} r {obs[best]:+.3f}; family max|r| {om:.3f} vs placebo max|r| median "
          f"{np.median(nullmax):.3f}, 95% {np.percentile(nullmax,95):.3f}, P(placebo max >= obs) {np.mean(np.array(nullmax) >= om):.3f}")
    print("    per-variable placebo p (<.1 shown):", {k: round(x, 3) for k, x in pv.items() if x < .1})

# ------------------------------------------------------------------ multivariate non-causal upper bound
print("\n=== multivariate upper bound (all family variables -> residual) ===")
FAM_U = PRIMARY_U + U23
FAM_ALL = PRIMARY_U + U23 + SEASON + CHAIN
def mv(df, ycol, cols, grp, kfold=None, label=""):
    sub = df.dropna(subset=[ycol, "sg_mean"]); yv = sub[ycol].values; g = sub[grp].values
    Xf = sub[["sg_mean"] + cols].values.astype(float)
    if kfold: splits = list(GroupKFold(n_splits=kfold).split(sub, yv, g))
    else: splits = [(np.where(g != k)[0], np.where(g == k)[0]) for k in np.unique(g)]
    pb = np.zeros(len(sub)); pr_ = np.zeros(len(sub)); pg = np.zeros(len(sub)); alph = []
    for tr, te in splits:
        med = np.nanmedian(Xf[tr], 0); med = np.where(np.isfinite(med), med, 0)
        Xi = np.where(np.isfinite(Xf), Xf, med); mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-9; Z = (Xi - mu) / sd
        pb[te] = ols_pred(Z[tr][:, :1], yv[tr], Z[te][:, :1])
        # nested alpha by inner group split
        best, ba = None, 1e18
        gtr = g[tr]; ug = np.unique(gtr)
        inner = [(np.where(gtr != k)[0], np.where(gtr == k)[0]) for k in ug] if not kfold else list(GroupKFold(n_splits=5).split(tr, yv[tr], gtr))
        for a in (1, 3, 10, 30, 100, 300, 1000):
            e = 0
            for itr, ite in inner:
                m = Ridge(alpha=a).fit(Z[tr][itr], yv[tr][itr]); e += np.sum((yv[tr][ite] - m.predict(Z[tr][ite])) ** 2)
            if e < ba: ba, best = e, a
        alph.append(best)
        pr_[te] = Ridge(alpha=best).fit(Z[tr], yv[tr]).predict(Z[te])
        gb = HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=max(3, len(tr) // 15), random_state=0)
        pg[te] = gb.fit(Xf[tr], yv[tr]).predict(Xf[te])
    sst = np.sum((yv - yv.mean()) ** 2)
    r2 = lambda p: 1 - np.sum((yv - p) ** 2) / sst
    hi = sub.hi10.values == 1
    print(f"{label}: n {len(sub)} groups {len(np.unique(g))} | R2cv base(sg) {r2(pb):+.3f} | ridge {r2(pr_):+.3f} (alphas {sorted(set(alph))}) "
          f"| GBM {r2(pg):+.3f} | hi10 improved vs base: ridge {int(np.sum(np.abs(yv-pr_)[hi] < np.abs(yv-pb)[hi]))}/{int(hi.sum())}, "
          f"GBM {int(np.sum(np.abs(yv-pg)[hi] < np.abs(yv-pb)[hi]))}/{int(hi.sum())} | hi10 MSE base {np.mean((yv-pb)[hi]**2):.4f} "
          f"ridge {np.mean((yv-pr_)[hi]**2):.4f} GBM {np.mean((yv-pg)[hi]**2):.4f}")
    return r2(pb), r2(pr_), r2(pg)
UB = {}
UB["Y2h_U"] = mv(hiS, "resid_mean", FAM_U, "ep_id", label="Y2h hi10, U variables only, LOCO chains")
UB["Y2h_ALL"] = mv(hiS, "resid_mean", FAM_ALL, "ep_id", label="Y2h hi10, U + season + chain(D), LOCO chains")
UB["Y2a_U"] = mv(allS, "resid_mean", FAM_U, "clu", kfold=10, label="Y2a all, U variables only, GroupKFold10 by date")
UB["Y2a_ALL"] = mv(allS, "resid_mean", FAM_ALL, "clu", kfold=10, label="Y2a all, U + season + chain(D), GroupKFold10 by date")
# Y1 level among highs: ec ~ sg + family
hiS2 = hiS.copy(); hiS2["Y1"] = hiS2.ec_mean
UB["Y1_ALL"] = mv(hiS2, "Y1", FAM_ALL, "ep_id", label="Y1 hi10 level, sg + all family, LOCO chains")

# ------------------------------------------------------------------ onset description (occurrence, not magnitude)
print("\n=== episode onset vs ordinary days (descriptive) ===")
on = D[(D.hi10 == 1) & (D.prev_hi10 == 0)]
ordn = D[(D.hi10 == 0) & (D.prev_hi10 == 0)]
for v in ["rad_p3", "rad_p7", "et0_p3", "tout_p3", "ttrend5", "dsince_hirad", "lowrad_run", "rad_p3_dm11", "dsince_hirad_dm11",
          "lowrad_run_dm11", "sealed_run", "sealed_p3"]:
    a, b = on[v].dropna(), ordn[v].dropna()
    if len(a) >= 2 and len(b) >= 5:
        print(f"  {v:20s} onset n {len(a)} mean {a.mean():7.3f} | ordinary n {len(b)} mean {b.mean():7.3f} | MWU p {stats.mannwhitneyu(a, b).pvalue:.3f}")
print("saved:", os.path.join(HERE, "vars_v1.csv"), "|", os.path.join(HERE, "stats_v1.csv"))
