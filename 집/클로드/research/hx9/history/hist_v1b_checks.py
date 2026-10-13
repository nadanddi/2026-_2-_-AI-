# -*- coding: utf-8 -*-
"""HX9 / history family — v1b follow-up checks (2026-10-13 집 클로드). hist_v1.py 결과를 읽고 '비판' 단계 검사를 더한다.
hist_v1.py 와 그 산출물은 고치지 않는다.

v1 에서 남은 의문과 이 스크립트의 검사 (모두 계산 전에 적어 둠):
  C1. v1 의 같은 날짜 분산 비교는 ddof=0 이라 2~3개 묶음에서 '날짜 안 분산'을 과소평가 → 날짜 수준 상한이 부풀었을 수 있음.
      → ddof=1 (일원분산분석 ICC) 과 같은 날짜 쌍 잔차 상관으로 다시 잼. 예측: ICC 가 0 근처면 날짜 수준 변수(외기 이력)는
        잔차를 설명할 여지가 거의 없음.
  C2. v1 Y2a 다변량 GBM 의 hi10 MSE 개선(0.160 → 0.135)은 GroupKFold(날짜) 라서 겹치는 여러 날 창이 '날짜 식별표'로
      새는 인공물일 수 있음 (6.175 MW1 실패 방식). → DM11 14일 연속 블록 단위로 빼는 교차검증 + 날짜 이동 위약(s 일 옮긴
      날씨 이력으로 같은 학습) 분포와 비교. 예측(인공물이면): 블록 CV 에서 이득 소멸, 위약도 비슷한 이득.
  C3. 약한 후보(vpd_p3, et0_p3, trans_p1, lowrad_run)의 사슬 하나 빼기 안정성 (부호가 한 사슬에 달렸는지).
  C4. trans_p1 (전날 같은 동 증산 대리 = Σ 일사×(1-차광)×실내VPD) 의 구성요소 분해 (사후 서술, 선택에 쓰지 않음).
  C5. 평가 기록 60개에 대해 DMT3e 달력 변수가 실제로 계산되는 비율 (U 표시의 실효 범위).
  C6. 1차/2차 부호 일치율 (Y2a, 변수 전체): 우연이면 50%.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
HX9 = os.path.dirname(HERE); RES = os.path.dirname(HX9)
sys.path.insert(0, os.path.join(RES, "linker_v4"))
from feat4 import X as XALL, FARM, NUM, N, TRAIN  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
import importlib.util  # noqa: E402

R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
H = pd.read_csv(os.path.join(HX9, "data", "hourly.csv"))
V = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec")
S = pd.read_csv(os.path.join(HERE, "stats_v1.csv"))
D = R.join(V); D["Y3"] = D.ec_mean - D.prev_ec_mean
hiS = D[(D.hi10 == 1) & (D.has_sg == 1)]; hi3 = D[(D.hi10 == 1) & D.prev_ec_mean.notna()]; allS = D[D.has_sg == 1].copy()
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"

# re-use the daily weather / date_vars definitions from hist_v1 without re-running its analysis
src = open(os.path.join(HERE, "hist_v1.py"), encoding="utf-8").read()
cut = src.index("VD = {}")
ns = {"__file__": os.path.join(HERE, "hist_v1.py"), "__name__": "hist_v1_defs"}
exec(compile(src[:cut], "hist_v1_defs", "exec"), ns)   # defines es, daily_weather, T_dmt, T_11, P75, P40, date_vars
T_dmt, T_11, date_vars, daily_weather, es = ns["T_dmt"], ns["T_11"], ns["date_vars"], ns["daily_weather"], ns["es"]
PRIMARY_U = ["rad_p1", "rad_p3", "rad_p5", "rad_p7", "rad_p14", "et0_p1", "et0_p3", "et0_p7", "vpd_p1", "vpd_p3", "vpd_p7",
             "tout_p1", "tout_p3", "tout_p7", "tmin_p3", "hum_p3", "ttrend5", "tchg1", "radchg1", "dsince_hirad", "lowrad_run"]

# ------------------------------------------------------------------ C1 unbiased same-date ICC
print("=== C1 same-date ICC (DM11 dates, ddof=1 one-way ANOVA) ===")
def icc(df, y):
    d = df[df.dm11_day.notna()]
    g = d.groupby("dm11_day")[y]
    k = g.size(); d = d[d.dm11_day.isin(k[k >= 1].index)]
    grand = d[y].mean(); ng = g.ngroups; n = len(d)
    ssb = (g.size() * (g.mean() - grand) ** 2).sum(); ssw = ((d[y] - d.dm11_day.map(g.mean())) ** 2).sum()
    msb = ssb / (ng - 1); msw = ssw / (n - ng)
    n0 = (n - (g.size() ** 2).sum() / n) / (ng - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw), ng, n, int((g.size() >= 2).sum())
def pair_r(df, y):
    d = df[df.dm11_day.notna()]; a, b = [], []
    for _, s in d.groupby("dm11_day")[y]:
        v = s.values
        for i in range(len(v)):
            for j in range(len(v)):
                if i != j: a.append(v[i]); b.append(v[j])
    return (np.corrcoef(a, b)[0, 1], len(a) // 2) if len(a) > 4 else (np.nan, 0)
for nm, df, y in (("Y1 hi10 ec_mean", D[D.hi10 == 1], "ec_mean"), ("Y2h hi10 resid", hiS, "resid_mean"), ("Y2a all resid", allS, "resid_mean"),
                  ("all ec_mean", D, "ec_mean"), ("Y2a normal resid", allS[allS.hi10 == 0], "resid_mean")):
    ic, ng, n, nmult = icc(df, y); r, npairs = pair_r(df, y)
    print(f"  {nm:20s} ICC {ic:+.3f} (n {n}, dates {ng}, dates with >=2 {nmult}) | same-date pair r {r:+.3f} ({npairs} pairs)")

# ------------------------------------------------------------------ DM11 per-day table of date variables (for shifting)
T11v = {}
for d_ in range(1, 127):
    def g(k, d_=d_):
        dd = d_ - k
        if dd < 1 or dd > 126 or not np.isfinite(T_11.loc[dd, "rad"]): return None
        return T_11.loc[dd]
    T11v[d_] = date_vars(g)
T11v = pd.DataFrame.from_dict(T11v, orient="index")
DV = PRIMARY_U + ["rad_d0", "et0_d0"]

def shifted(df, s):
    dd = df.dm11_day.values
    return np.array([[T11v.at[int(d + s), v] if (np.isfinite(d) and 1 <= d + s <= 126) else np.nan for v in DV] for d in dd])

# ------------------------------------------------------------------ C2 block CV + date-shift placebo for multivariate Y2a
print("\n=== C2 multivariate Y2a with 14-day DM11 block CV, actual vs date-shift placebo ===")
A = allS.copy()
A["blk"] = [f"b{int(d // 14)}" if np.isfinite(d) else f"r{r}" for r, d in zip(A.index, A.dm11_day)]
yv = A.resid_mean.values; sg = A.sg_mean.values; hi = A.hi10.values == 1
blks = A.blk.values; ub = np.unique(blks)
splits = [(np.where(blks != b)[0], np.where(blks == b)[0]) for b in ub]
def run(Xd):
    Xf = np.c_[sg, Xd]
    pb = np.zeros(len(A)); pr = np.zeros(len(A)); pg = np.zeros(len(A))
    for tr, te in splits:
        Aa = np.c_[np.ones(len(tr)), sg[tr]]; w = np.linalg.lstsq(Aa, yv[tr], rcond=None)[0]; pb[te] = np.c_[np.ones(len(te)), sg[te]] @ w
        med = np.nanmedian(Xf[tr], 0); med = np.where(np.isfinite(med), med, 0); Xi = np.where(np.isfinite(Xf), Xf, med)
        mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-9; Z = (Xi - mu) / sd
        pr[te] = Ridge(alpha=100).fit(Z[tr], yv[tr]).predict(Z[te])
        pg[te] = HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=20, random_state=0).fit(Xf[tr], yv[tr]).predict(Xf[te])
    sst = np.sum((yv - yv.mean()) ** 2); r2 = lambda p: 1 - np.sum((yv - p) ** 2) / sst
    return dict(base=r2(pb), ridge=r2(pr), gbm=r2(pg), hi_base=np.mean((yv - pb)[hi] ** 2), hi_ridge=np.mean((yv - pr)[hi] ** 2),
                hi_gbm=np.mean((yv - pg)[hi] ** 2), hi_imp_gbm=int(np.sum(np.abs(yv - pg)[hi] < np.abs(yv - pb)[hi])))
act = run(shifted(A, 0))
print(f"  actual (s=0): blocks {len(ub)} | R2cv base {act['base']:+.3f} ridge {act['ridge']:+.3f} GBM {act['gbm']:+.3f} | hi10 MSE base "
      f"{act['hi_base']:.4f} ridge {act['hi_ridge']:.4f} GBM {act['hi_gbm']:.4f} (GBM improves {act['hi_imp_gbm']}/{int(hi.sum())})")
pl = []
for s in [-28, -21, -17, -14, -10, -7, 7, 10, 14, 17, 21, 28]:
    o = run(shifted(A, s)); o["s"] = s; pl.append(o)
P = pd.DataFrame(pl)
print("  placebo shifts:\n" + P[["s", "base", "ridge", "gbm", "hi_base", "hi_ridge", "hi_gbm", "hi_imp_gbm"]].round(4).to_string(index=False))
print(f"  P(placebo GBM R2 >= actual) {np.mean(P.gbm >= act['gbm']):.2f}; P(placebo ridge R2 >= actual) {np.mean(P.ridge >= act['ridge']):.2f}; "
      f"P(placebo hi10 GBM MSE <= actual) {np.mean(P.hi_gbm <= act['hi_gbm']):.2f}")
# same for Y2h with LOCO chains (DM11 copies, full 31 rows)
print("\n  Y2h (hi10) LOCO chains, DM11 copies: actual vs placebo (ridge alpha 300)")
Bh = hiS.copy(); yh = Bh.resid_mean.values; sgh = Bh.sg_mean.values; gh = Bh.ep_id.values
def runh(Xd):
    Xf = np.c_[sgh, Xd]; pb = np.zeros(len(Bh)); pr = np.zeros(len(Bh))
    for k in np.unique(gh):
        tr, te = np.where(gh != k)[0], np.where(gh == k)[0]
        Aa = np.c_[np.ones(len(tr)), sgh[tr]]; w = np.linalg.lstsq(Aa, yh[tr], rcond=None)[0]; pb[te] = np.c_[np.ones(len(te)), sgh[te]] @ w
        med = np.nanmedian(Xf[tr], 0); med = np.where(np.isfinite(med), med, 0); Xi = np.where(np.isfinite(Xf), Xf, med)
        mu, sd = Xi[tr].mean(0), Xi[tr].std(0) + 1e-9; Z = (Xi - mu) / sd
        pr[te] = Ridge(alpha=300).fit(Z[tr], yh[tr]).predict(Z[te])
    sst = np.sum((yh - yh.mean()) ** 2); return 1 - np.sum((yh - pb) ** 2) / sst, 1 - np.sum((yh - pr) ** 2) / sst
b0, r0 = runh(shifted(Bh, 0))
plh = [runh(shifted(Bh, s))[1] for s in range(-30, 31) if abs(s) >= 5]
print(f"  actual base {b0:+.3f} ridge {r0:+.3f} | placebo ridge median {np.median(plh):+.3f}, 90% {np.percentile(plh, 90):+.3f}, "
      f"P(placebo >= actual) {np.mean(np.array(plh) >= r0):.2f}")

# ------------------------------------------------------------------ C3 drop-one-chain stability
print("\n=== C3 drop-one-chain r (weak candidates) ===")
for v, df, y in (("vpd_p3", hiS, "resid_mean"), ("et0_p3", hiS, "resid_mean"), ("vpd_p3_dm11", hiS, "resid_mean"),
                 ("lowrad_run_dm11", hiS, "resid_mean"), ("trans_p1", hiS, "resid_mean"), ("trans_p1", hi3, "Y3"),
                 ("ttrend5_dm11", hi3, "Y3"), ("dsince_hirad", D[D.hi10 == 1], "ec_mean")):
    sub = df[[v, y, "ep_id"]].dropna(); out = []
    for k in sorted(sub.ep_id.unique()):
        s2 = sub[sub.ep_id != k]
        out.append(f"{int(k)}:{np.corrcoef(s2[v], s2[y])[0,1]:+.2f}(n{len(s2)})")
    cnt = sub.ep_id.value_counts().sort_index().to_dict()
    print(f"  {v:16s} vs {y:10s} all r {np.corrcoef(sub[v], sub[y])[0,1]:+.2f} n {len(sub)} chains {cnt} | drop: " + " ".join(out))

# ------------------------------------------------------------------ C4 trans_p1 decomposition (post hoc, descriptive)
print("\n=== C4 trans_p1 components (prev-day record on clean chain; descriptive, not used for selection) ===")
Hh = H.copy(); Hh["in_vpd"] = es(Hh.in_temp) * (1 - Hh.in_hum / 100.0)
Hh["light"] = Hh.out_rad * 0.0036 * (1 - Hh.act_shade.fillna(0) / 100.0)
Hh["rv"] = Hh.out_rad * 0.0036 * Hh.in_vpd
day = Hh[(Hh.hour >= 9) & (Hh.hour <= 16)]
comp = pd.DataFrame({"light_sum": Hh.groupby("rec").light.sum(), "rad_x_invpd": Hh.groupby("rec").rv.sum(),
                     "invpd_day": day.groupby("rec").in_vpd.mean(), "shade_day": day.groupby("rec").act_shade.mean(),
                     "rad_sum": Hh.groupby("rec").out_rad.sum() * 0.0036, "intemp_day": day.groupby("rec").in_temp.mean(),
                     "vent_day": day.groupby("rec").act_vent.mean()})
for c in comp.columns:
    D[c + "_p1"] = [comp.at[p, c] if isinstance(p, str) else np.nan for p in D.prev_rec]
hiS = D[(D.hi10 == 1) & (D.has_sg == 1)]; hi3 = D[(D.hi10 == 1) & D.prev_ec_mean.notna()]
for c in ["trans_p1"] + [c + "_p1" for c in comp.columns]:
    a = hi3[[c, "Y3"]].dropna(); b = hiS[[c, "resid_mean"]].dropna()
    print(f"  {c:18s} r_Y3 {np.corrcoef(a[c], a.Y3)[0,1]:+.3f} (n {len(a)}) | r_Y2h {np.corrcoef(b[c], b.resid_mean)[0,1]:+.3f} (n {len(b)})")

# ------------------------------------------------------------------ C5 evaluation coverage of DMT3e variables
print("\n=== C5 coverage of DMT3e date variables for the 60 evaluation records ===")
TRI = np.where(TRAIN)[0]; EVI = np.where(~TRAIN)[0]
seg_day = {r: (R.at[r, "dmt_seg"], R.at[r, "dmt_day"]) for r in R.index}
cov = dict(n=0, h0_match=0, p1=0, p3=0, p7=0)
for i in EVI:
    cov["n"] += 1
    eq = np.all(np.abs(XALL[TRI, 0, :] - XALL[i, 0, :]) < 1e-6, axis=1)
    m = [rid(j) for j in TRI[eq]]
    if not m: continue
    cov["h0_match"] += 1
    sds = {seg_day[x] for x in m}
    s, d = sorted(sds)[0]
    have = [(s, d - k) in T_dmt.index for k in range(1, 8)]
    cov["p1"] += int(have[0]); cov["p3"] += int(sum(have[:3]) >= 2); cov["p7"] += int(sum(have) >= 4)
print("  ", cov, "(h0_match = eval record whose hour-0 outdoor equals >=1 training record; p1/p3/p7 = window computable)")
for nm, df in (("train hi10", D[D.hi10 == 1]), ("train all", D)):
    print(f"   {nm}: rad_p1 available {int(df.rad_p1.notna().sum())}/{len(df)}, rad_p3 {int(df.rad_p3.notna().sum())}, rad_p7 {int(df.rad_p7.notna().sum())}; "
          f"pass2 hi10 rad_p1 available {int(df[df.pass2 == 1].rad_p1.notna().sum())}/{int((df.pass2 == 1).sum())}")

# ------------------------------------------------------------------ C6 pass-1 vs pass-2 sign agreement
print("\n=== C6 pass-1 vs pass-2 sign agreement of r(var, residual) over all records ===")
ok = S[["r_Y2a_pass1", "r_Y2a_pass2"]].dropna()
agree = np.mean(np.sign(ok.r_Y2a_pass1) == np.sign(ok.r_Y2a_pass2))
print(f"  variables {len(ok)}; same sign {agree:.2f}; binomial p (vs .5) {stats.binomtest(int(round(agree*len(ok))), len(ok)).pvalue:.3f}; "
      f"corr of r across passes {np.corrcoef(ok.r_Y2a_pass1, ok.r_Y2a_pass2)[0,1]:+.2f}")
ok = S[["r_Y2a_F13", "r_Y2a_F47"]].dropna()
print(f"  F13 vs F47: same sign {np.mean(np.sign(ok.r_Y2a_F13) == np.sign(ok.r_Y2a_F47)):.2f}; corr {np.corrcoef(ok.r_Y2a_F13, ok.r_Y2a_F47)[0,1]:+.2f}")
ph = hiS[hiS.pass2 == 1]
print(f"  pass-2 hi10 (n {len(ph)}): mean resid {ph.resid_mean.mean():+.3f} vs pass-1 hi10 {hiS[hiS.pass2==0].resid_mean.mean():+.3f}; "
      f"their DMT3e rad_p1 available {int(ph.rad_p1.notna().sum())}/{len(ph)} (all at DMT3e segment starts)")
for v in ["vpd_p3_dm11", "et0_p3_dm11", "lowrad_run_dm11", "ttrend5_dm11", "trans_p1"]:
    s1 = hiS[hiS.pass2 == 0][[v, "resid_mean"]].dropna(); s2 = ph[[v, "resid_mean"]].dropna()
    if len(s1) > 4 and len(s2) >= 3:
        w = np.polyfit(s1[v], s1.resid_mean, 1); pred2 = np.polyval(w, s2[v])
        print(f"   {v:16s} pass-1 fit applied to pass-2 hi10: mean predicted resid {pred2.mean():+.3f} vs actual {s2.resid_mean.mean():+.3f}; "
              f"r within pass-2 {np.corrcoef(s2[v], s2.resid_mean)[0,1]:+.2f} (n {len(s2)})")
