# -*- coding: utf-8 -*-
"""VW2 (verify_water) — 다변량 상한 재점검 + 빠뜨린 변수 (2026-10-13 집 클로드, 적대적 검증자).
 (1) 상한 재점검: wb2 의 ridge/GBM 47개 동시 투입은 n=31 에서 과적합으로 음수가 '당연'할 수 있다.
     더 공정한 상한 = 사슬 하나 빼기 안에서 '변수 고르기까지' 학습 폴드에서만 하는 중첩 선택(상위 1·2·3개, 잔차 ~ sg + x).
     (a) 47개 전부(진단 포함, 비인과 상한)  (b) 사용 가능(U/U5/Uid)만.  ep_mean_et_sofar 는 결측=정답 표지라 제외.
 (2) 빠뜨린 변수 7개(사용 가능, 학습 입력만, MASK 준수: 같은 ID 지연은 학습 기록만): fog_sum, fog_hours, co2_draw_closed, co2_dep_day,
     circfan_sum, lag1id_et_total, lagmean_id_et(num-2, num-4). 같은 판정 규칙 + 7개 가족 순열 p.
     일반 원예 지식(우리 자료로 확인 안 됨): 포그는 VPD 를 낮춰 증산을 줄임; 밀폐 낮 CO2 감소는 광합성·기공 개도(증산과 같이 움직임) 대리;
     순환팬은 엽면 경계층을 얇게 해 증산을 늘림.
 (3) 하루 안 기전 진단(정답 사용 → 진단 전용): 고EC 기록에서 시간당 ΔEC 와 시간당 증산 대리의 기록 안 상관(07~18시),
     낮 EC 상승(16시-06시)과 낮 증산 합의 상관. B형(농축)이면 양, A형(세척 부족, 하루 단위)이면 하루 안에서는 무관하거나 음.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
rng = np.random.default_rng(41013)

R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HX9, "water", "vars_v1.csv")).set_index("rec").loc[R.index]
CD = pd.read_csv(os.path.join(HX9, "water", "vars_v1_columns.csv")); VARS = list(CD.variable); USAB = dict(zip(CD.variable, CD.usability))
R = R.join(F)
H = pd.read_csv(os.path.join(HX9, "data", "hourly.csv")).sort_values(["rec", "hour"])

# ---------------- new variables ----------------
def es(T): return 0.6108 * np.exp(17.27 * T / (T + 237.3))
HRS = np.arange(24); DAY = (HRS >= 9) & (HRS <= 16)
V = ["out_rad", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_fog", "act_co2", "act_circfan", "sub_ec"]
new, hourly_mech = {}, []
for rec, g in H.groupby("rec"):
    g = g.set_index("hour").reindex(range(24))
    a = {v: g[v].interpolate(limit_direction="both").values.astype(float) for v in V}
    co2, vent, fog, aco2 = a["in_co2"], a["act_vent"], a["act_fog"], a["act_co2"]
    dco2 = np.r_[np.nan, -np.diff(co2)]
    mm = DAY & (vent < 5) & (aco2 <= 0)
    rad = np.clip(a["out_rad"] - 5, 0, None)
    radIn = rad * 0.65 * (1 - 0.6 * a["act_shade"] / 100) * (1 - 0.4 * a["act_thermal"] / 100)
    vpd = es(a["in_temp"]) * (1 - np.clip(a["in_hum"], 0, 100) / 100)
    E = np.where(rad > 10, 0.30 * radIn + 20 * vpd, 10 * vpd)
    new[rec] = dict(fog_sum=np.nansum(fog) / 100, fog_hours=float((fog > 0).sum()),
                    co2_draw_closed=np.nanmean(dco2[mm]) if mm.sum() >= 2 else np.nan,
                    co2_dep_day=np.nanmean(co2[:6]) - np.nanmin(co2[9:17]),
                    circfan_sum=np.nansum(a["act_circfan"]) / 100,
                    ec_rise_day=a["sub_ec"][16] - a["sub_ec"][6])
    w = (HRS >= 7) & (HRS <= 18)
    dec = np.r_[np.nan, np.diff(a["sub_ec"])]
    hourly_mech.append(dict(rec=rec, r_hourly=np.corrcoef(dec[w], E[w])[0, 1] if np.std(E[w]) > 0 and np.std(dec[w]) > 0 else np.nan))
N = pd.DataFrame(new).T.loc[R.index]
lagk = lambda r, k: f"{r[:3]}_{int(r[4:]) - k:03d}"
N["lag1id_et_total"] = [R.at[lagk(r, 1), "et_total"] if lagk(r, 1) in R.index else np.nan for r in R.index]
N["lagmean_id_et"] = [np.nanmean([R.at[lagk(r, k), "et_total"] for k in (2, 4) if lagk(r, k) in R.index]) if any(lagk(r, k) in R.index for k in (2, 4)) else np.nan
                      for r in R.index]
NEWV = ["fog_sum", "fog_hours", "co2_draw_closed", "co2_dep_day", "circfan_sum", "lag1id_et_total", "lagmean_id_et"]
R = R.join(N)
M = pd.DataFrame(hourly_mech).set_index("rec"); R = R.join(M)

HI2 = R[(R.hi10 == 1) & (R.has_sg == 1)].copy(); AL2 = R[R.has_sg == 1].copy()
y = HI2.resid_mean.values; sg = HI2.sg_mean.values; ep = HI2.ep_id.values.astype(int)
par = {r: r for r in R.index}
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for b, a_ in R.prev_rec.items():
    if isinstance(a_, str): par[fd(b)] = fd(a_)
R["chain_all"] = [fd(r) for r in R.index]; AL2 = R[R.has_sg == 1].copy()


def lstsq_pred(Xtr, ytr, Xte):
    c = np.linalg.lstsq(np.c_[np.ones(len(Xtr)), Xtr], ytr, rcond=None)[0]
    return np.c_[np.ones(len(Xte)), Xte] @ c


# ---------------- (1) nested-selection LOCO upper bound ----------------
def nested(df, feats, k, groups="ep_id", yv=None):
    yy = df.resid_mean.values if yv is None else yv; s = df.sg_mean.values; g = df[groups].values
    X = df[feats].values.astype(float); pb = np.zeros(len(yy)); pm = np.zeros(len(yy)); picks = []
    for u in np.unique(g):
        te = g == u; tr = ~te
        med = np.nanmedian(X[tr], 0); med = np.where(np.isfinite(med), med, 0)
        Xtr = np.where(np.isfinite(X[tr]), X[tr], med); Xte = np.where(np.isfinite(X[te]), X[te], med)
        rtr = np.array([abs(np.corrcoef(Xtr[:, j], yy[tr])[0, 1]) if np.std(Xtr[:, j]) > 1e-12 else 0 for j in range(X.shape[1])])
        top = np.argsort(-np.nan_to_num(rtr))[:k]; picks.append(tuple(feats[j] for j in top))
        pb[te] = lstsq_pred(s[tr, None], yy[tr], s[te, None])
        pm[te] = lstsq_pred(np.c_[s[tr], Xtr[:, top]], yy[tr], np.c_[s[te], Xte[:, top]])
    sst = np.sum((yy - yy.mean()) ** 2)
    return (np.sum((yy - pb) ** 2) - np.sum((yy - pm) ** 2)) / sst, picks

ALLNL = [v for v in VARS if v != "ep_mean_et_sofar"]
USEV = [v for v in VARS if USAB[v] in ("U", "U5", "Uid")]
print("=== (1) nested-selection leave-one-episode gain over a+b*sg (hi10 Y2, n=%d) ===" % len(HI2))
for nm, feats in (("all46(non-causal)", ALLNL), ("usable", USEV), ("usable+new7", USEV + NEWV)):
    for k in (1, 2, 3):
        gn, picks = nested(HI2, feats, k)
        nul = [nested(HI2, feats, k, yv=None if False else rng.permutation(y))[0] for _ in range(60)]
        pc = pd.Series([p[0] for p in picks]).value_counts().head(3).to_dict()
        print(f"  {nm:18s} top{k}: gain {gn:+.3f} | null(permuted y, 60) mean {np.mean(nul):+.3f}, 95% {np.percentile(nul,95):+.3f} | first pick per fold {pc}")

# ---------------- (2) missed variables ----------------
def logo_gain(yv, base, x, g):
    m = np.isfinite(yv) & np.isfinite(base) & np.isfinite(x)
    yv, base, g, xx = yv[m], base[m], np.asarray(g)[m], x[m]
    pb, pa = np.zeros_like(yv), np.zeros_like(yv)
    for u in np.unique(g):
        te = g == u; tr = ~te
        pb[te] = lstsq_pred(base[tr, None], yv[tr], base[te, None])
        pa[te] = lstsq_pred(np.c_[base[tr], xx[tr]], yv[tr], np.c_[base[te], xx[te]])
    sst = np.sum((yv - yv.mean()) ** 2)
    return (np.sum((yv - pb) ** 2) - np.sum((yv - pa) ** 2)) / sst


def clus_ci(x, yv, g, nb=2000):
    m = np.isfinite(x) & np.isfinite(yv); x, yv, g = x[m], yv[m], np.asarray(g)[m]
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; out = []
    for _ in range(nb):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug), replace=True)])
        if np.std(x[s]) > 1e-12 and np.std(yv[s]) > 1e-12: out.append(np.corrcoef(x[s], yv[s])[0, 1])
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


PR = R[(R.hi10 == 1) & R.prev_rec.notna()].copy()
PR = PR[[R.at[p, "hi10"] == 1 and R.at[p, "ep_id"] == R.at[b, "ep_id"] for b, p in PR.prev_rec.items()]]
def pcor(d, x, a, z):
    d = d[[x, a] + z].dropna()
    if len(d) < 6 or d[x].std() < 1e-12: return np.nan
    Z = np.c_[np.ones(len(d)), d[z].values]
    ex = d[x].values - Z @ np.linalg.lstsq(Z, d[x].values, rcond=None)[0]; ea = d[a].values - Z @ np.linalg.lstsq(Z, d[a].values, rcond=None)[0]
    return np.corrcoef(ex, ea)[0, 1]

rows = []
for v in NEWV:
    x = HI2[v].values.astype(float); m = np.isfinite(x)
    r = np.corrcoef(x[m], y[m])[0, 1] if np.std(x[m]) > 0 else np.nan
    lo, hi = clus_ci(x, y, ep); gn = logo_gain(y, sg, x, ep)
    ex = lo > 0 or hi < 0
    vd = "signal" if abs(r) >= .35 and ex and gn > 0 else ("weak" if (abs(r) >= .25 and ex) or (gn > 0 and abs(r) >= .20) else "none")
    d = HI2[[v, "resid_mean", "ep_id"]].dropna(); d = d[d.groupby("ep_id")[v].transform("size") >= 2]
    rw = np.corrcoef(d[v] - d.groupby("ep_id")[v].transform("mean"), d.resid_mean - d.groupby("ep_id").resid_mean.transform("mean"))[0, 1]
    xa = AL2[v].values.astype(float); ya = AL2.resid_mean.values; ma = np.isfinite(xa)
    ra = np.corrcoef(xa[ma], ya[ma])[0, 1]; la, ha = clus_ci(xa, ya, AL2.chain_all.values, nb=1000)
    rows.append(dict(variable=v, n=int(m.sum()), r_Y2=r, ci=f"[{lo:+.2f},{hi:+.2f}]", cv_gain=gn, verdict=vd, r_within_ep=rw,
                     r_level=np.corrcoef(x[m], HI2.ec_mean.values[m])[0, 1], r_Y2all=ra, ci_all=f"[{la:+.2f},{ha:+.2f}]",
                     pr_Y3=pcor(PR, v, "ec_mean", ["prev_ec_mean"]),
                     auc_hi=stats.mannwhitneyu(R.loc[R.hi10 == 1, v].dropna(), R.loc[R.hi10 == 0, v].dropna()).statistic /
                            (R.loc[R.hi10 == 1, v].notna().sum() * R.loc[R.hi10 == 0, v].notna().sum())))
T = pd.DataFrame(rows).set_index("variable")
Xm = HI2[NEWV].values.astype(float)
def allr(yv):
    return np.array([np.corrcoef(Xm[np.isfinite(Xm[:, j]), j], yv[np.isfinite(Xm[:, j])])[0, 1] if np.std(Xm[np.isfinite(Xm[:, j]), j]) > 0 else np.nan
                     for j in range(len(NEWV))])
mx = np.array([np.nanmax(np.abs(allr(rng.permutation(y)))) for _ in range(2000)])
T["p_fw7"] = [np.mean(mx >= abs(r)) for r in T.r_Y2]
print("\n=== (2) missed variables (usable; 7 tested; same pre-fixed rule) ===")
print(T.round(3).to_string())

# ---------------- (3) within-day mechanism (diagnostic only) ----------------
print("\n=== (3) within-day mechanism (uses target labels -> diagnostic only) ===")
for nm, s in (("hi10", R[R.hi10 == 1]), ("normal", R[R.hi10 == 0])):
    rh = s.r_hourly.dropna()
    print(f"  {nm:6s}: mean within-record r(hourly dEC, hourly ET proxy, 07-18h) {rh.mean():+.3f} (sd {rh.std():.3f}, n {len(rh)}, "
          f"share>0 {np.mean(rh > 0):.2f}); r(ec_rise_day 16h-06h, et_day) {s[['ec_rise_day','et_day']].corr().iloc[0,1]:+.3f}; "
          f"mean ec_rise_day {s.ec_rise_day.mean():+.3f}")
