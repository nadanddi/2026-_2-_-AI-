# -*- coding: utf-8 -*-
"""HX9 / water  WB2 — 물 수지·증산 가족: 고EC 크기와의 관계 시험 (2026-10-13 집 클로드, 하위 작업).
입력: hx9/data/records.csv, hourly.csv (공용표), hx9/water/vars_v1.csv, vars_running_v1.csv (wb1).

=====================  실행 전에 적은 예측 (지문)  =====================
가설 W: 물 수지(증산·환기 수증기 배출·결로) 가 고EC 동의 EC '크기'를 정한다.
 A형(세척 부족): 증산↓(흐림·고습·밀폐·결로) → 배액↓ → 염 누적 → EC↑.   B형(농축): 증산↑, 급액 지연 → EC↑.
 W 가 크기를 몰면 보여야 할 것:
  P1  고EC(hi10) 안에서 Y1(하루 평균 EC) 과 증산 대리(et_total, rad_in_sum, vpd_rad, vpd_day) 상관 |r| ≥ .35, 부호는 A형이면 음, B형이면 양.
      밀폐·결로·밤 습도(sealed_hours, cond_hours, hum_night) 는 A형이면 양.
  P2  누적이 원인이면 하루 변화 Y3(clean 링크 같은 에피소드 안 ΔEC = 오늘 - 전날) 가 '그날' 증산 대리와 같은 부호로 상관
      (A형: 증산 적은 날 EC 가 오름 → r(et, Y3) < 0).  Y1 은 여러 날 누적값(et_3link, ep_mean_et_sofar, 7일 일사) 쪽이 같은날보다 강해야.
  P3  현재 모델은 같은 날 입력만 보므로 같은날 변수는 잔차 Y2 와 약하고, 여러 날 누적·전날 변수는 Y2 와 |r| ≥ .35 이어야.
  P4  F13·F47, 1차·2차에서 같은 부호. 계절(DM11 날짜) 통제 후에도 남아야.
 보이면 안 되는 것 (보이면 W 는 크기 원인 아님):
  N1  부호가 온실·차수에서 뒤집힘.   N2  고EC 대 보통 구분(체제)에서만 강하고 고EC 안에서는 |r| < .2.
  N3  한 에피소드 빼면 사라짐(사슬 하나 빼기 교차적합 이득 ≤ 0).   N4  Y3 와 무관한데 Y1 만 상관(= 계절·체제 표지).
  N5  같은 날 실외만으로 된 변수(rad_out_sum 등)가 같은 날 다른 동의 EC 차이를 설명한다고 보이는 것 — 같은 날짜 동끼리 실외가 같으므로 불가능,
      보이면 날짜(계절) 효과.
 내 사전 기대(정직하게): 과거 HC5(이전 3/7일 일사·VPD |ρ| ≤ .18), OP0(이전 운영 이력 ρ ≤ .25), PP1(AHx 등 약함), 6.778 증분(R²cv 음수)
  로 보아 대부분 none, 잘해야 weak.
판정 규칙 (실행 전 고정):
  signal : 고EC 잔차 Y2 와 |r| ≥ .35, 에피소드(사슬) 부트스트랩 95% CI 가 0 제외, 사슬 하나 빼기 R² 이득 > 0  (셋 다).
  weak   : signal 아님 & (고EC Y2 |r| ≥ .25 이고 CI 0 제외) 또는 (고EC Y2 CV 이득 > 0 이고 |r| ≥ .20)
           또는 (전체 기록 Y2 의 사슬 CI 0 제외 이고 전체 CV 이득 > 0).
  none   : 그 외.   다중검정: 변수 수 × 목표 수를 적고, 고EC Y2 는 기록 순열 최대 |r| 가족 p 를 함께 보고.
지난 시험과 다른 점: PP1(hx8) 은 S_pred·S_true 표본에서 21개 물리 대리의 잔차 설명력만; HC5/OP0 는 같은 동 이전 날짜 일사·VPD·운영 평균.
  여기는 (1) 세 목표(Y1 수준, Y2 잔차, Y3 링크 위 하루 변화)를 같은 변수에 동시에, (2) 증산 합성 대리(ET), 결로·이슬점 여유, 환기 수증기 배출,
  아침 AH 상승, 0~5시판, 같은 ID -2 판(사용 가능 지연), DM11·DMT 달력 여러 날 일사·흐림 지수, 에피소드 누적 ET 를 한 번에,
  (3) 사슬 부트스트랩 CI·사슬 하나 빼기 교차적합·가족 순열 p, (4) 0..h 인과판 시각별 행 잔차 검사, (5) 1차/2차·온실 부호 일관성.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from scipy import stats
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
rng = np.random.default_rng(20261013)
NB = 2000

R = pd.read_csv(os.path.join(DATA, "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec").loc[R.index]
CD = pd.read_csv(os.path.join(HERE, "vars_v1_columns.csv"))
VARS = list(CD.variable); USAB = dict(zip(CD.variable, CD.usability))
assert not set(F.columns) & set(R.columns)
R = R.join(F)
print(f"variables tested: {len(VARS)}  (targets: Y1 hi10 level, Y2 hi10 residual, Y2 all-record residual, Y3 within-episode day change)")

# ---------------- clusters ----------------
# all-record chains = connected components of clean links
par = {r: r for r in R.index}
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for b, a in R.prev_rec.items():
    if isinstance(a, str): par[fd(b)] = fd(a)
R["chain_all"] = [fd(r) for r in R.index]
print(f"all-record clean-link chains: {R.chain_all.nunique()} (largest {R.chain_all.value_counts().max()})")

HI = R[R.hi10 == 1].copy()                     # Y1 sample
HI2 = HI[HI.has_sg == 1].copy()                # Y2 high sample
AL2 = R[R.has_sg == 1].copy()                  # Y2 all-record sample
PR = HI[HI.prev_rec.notna()].copy()
PR = PR[[R.at[p, "hi10"] == 1 and R.at[p, "ep_id"] == R.at[b, "ep_id"] for b, p in PR.prev_rec.items()]]
PR["Y3"] = PR.ec_mean - PR.prev_ec_mean
print(f"samples: Y1 hi10 n={len(HI)} (episodes {HI.ep_id.nunique()}), Y2 hi10 with OOF n={len(HI2)} (episodes {HI2.ep_id.nunique()}, pass2 {int(HI2.pass2.sum())}), "
      f"Y2 all n={len(AL2)} (pass2 {int(AL2.pass2.sum())}), Y3 within-episode pairs n={len(PR)} (episodes {PR.ep_id.nunique()})")
print(f"Y3 mean {PR.Y3.mean():+.3f} sd {PR.Y3.std():.3f};  Y2 hi10 mean {HI2.resid_mean.mean():+.3f} sd {HI2.resid_mean.std():.3f}")


def rr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12: return np.nan, np.nan, int(m.sum())
    return stats.pearsonr(x[m], y[m])[0], stats.spearmanr(x[m], y[m])[0], int(m.sum())


def clus_ci(x, y, g, nb=NB):
    m = np.isfinite(x) & np.isfinite(y); x, y, g = x[m], y[m], np.asarray(g)[m]
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; out = []
    for _ in range(nb):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug), replace=True)])
        if np.std(x[s]) < 1e-12 or np.std(y[s]) < 1e-12: continue
        out.append(np.corrcoef(x[s], y[s])[0, 1])
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if len(out) > 100 else (np.nan, np.nan)


def partial_r(x, y, z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 6: return np.nan
    Z = np.c_[np.ones(m.sum()), z[m]]
    ex = x[m] - Z @ np.linalg.lstsq(Z, x[m], rcond=None)[0]; ey = y[m] - Z @ np.linalg.lstsq(Z, y[m], rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1] if np.std(ex) > 1e-12 else np.nan


def logo_gain(y, base, x, g):
    """leave-one-group-out: baseline y ~ 1 + base, aug y ~ 1 + base + x. returns (gain = (SSEb-SSEa)/SST, n_improved, n, R2cv_base)."""
    m = np.isfinite(y) & np.isfinite(base) & (np.isfinite(x) if x is not None else True)
    y, base, g = y[m], base[m], np.asarray(g)[m]; xx = x[m] if x is not None else None
    pb, pa = np.zeros_like(y), np.zeros_like(y)
    for u in np.unique(g):
        te = g == u; tr = ~te
        Xb = np.c_[np.ones(tr.sum()), base[tr]]; cb = np.linalg.lstsq(Xb, y[tr], rcond=None)[0]
        pb[te] = np.c_[np.ones(te.sum()), base[te]] @ cb
        if xx is not None:
            Xa = np.c_[Xb, xx[tr]]; ca = np.linalg.lstsq(Xa, y[tr], rcond=None)[0]
            pa[te] = np.c_[np.ones(te.sum()), base[te], xx[te]] @ ca
    sst = np.sum((y - y.mean()) ** 2)
    r2b = 1 - np.sum((y - pb) ** 2) / sst
    if xx is None: return np.nan, 0, len(y), r2b
    gain = (np.sum((y - pb) ** 2) - np.sum((y - pa) ** 2)) / sst
    return gain, int(np.sum(np.abs(y - pa) < np.abs(y - pb))), len(y), r2b


rows = []
for v in VARS:
    d = dict(variable=v, usability=USAB[v])
    # Y1 level among hi10
    x = HI[v].values.astype(float); y = HI.ec_mean.values
    d["r_Y1"], d["rho_Y1"], d["n_Y1"] = rr(x, y)
    d["pr_Y1_season"] = partial_r(x, y, HI.dm11_day.values.astype(float))
    lo, hi = clus_ci(x, y, HI.ep_id.values); d["ci_Y1"] = f"[{lo:+.2f},{hi:+.2f}]"
    # Y2 residual among hi10
    x = HI2[v].values.astype(float); y = HI2.resid_mean.values
    d["r_Y2"], d["rho_Y2"], d["n_Y2"] = rr(x, y)
    lo, hi = clus_ci(x, y, HI2.ep_id.values); d["ci_lo"], d["ci_hi"] = lo, hi
    d["pr_Y2_season"] = partial_r(x, y, HI2.dm11_day.values.astype(float))
    g_, imp_, n_, _ = logo_gain(y, HI2.sg_mean.values, x, HI2.ep_id.values)
    d["cv_gain_Y2"], d["n_impr_Y2"] = g_, imp_
    for f_ in ("F13", "F47"):
        s = HI2[HI2.farm == f_]; d[f"r_Y2_{f_}"] = rr(s[v].values.astype(float), s.resid_mean.values)[0]
    s = HI2[HI2.pass2 == 0]; d["r_Y2_p1"] = rr(s[v].values.astype(float), s.resid_mean.values)[0]
    s = HI2[HI2.pass2 == 1]
    xs, ys = s[v].values.astype(float), s.resid_mean.values; mm = np.isfinite(xs)
    d["r_Y2_p2(n5)"] = np.corrcoef(xs[mm], ys[mm])[0, 1] if mm.sum() >= 3 and np.std(xs[mm]) > 0 else np.nan
    # Y2 all records
    x = AL2[v].values.astype(float); y = AL2.resid_mean.values
    d["r_Y2all"], d["rho_Y2all"], d["n_Y2all"] = rr(x, y)
    lo, hi = clus_ci(x, y, AL2.chain_all.values, nb=1000); d["ci_all"] = f"[{lo:+.2f},{hi:+.2f}]"; d["ci_all_lo"], d["ci_all_hi"] = lo, hi
    g_, imp_, n_, _ = logo_gain(y, AL2.sg_mean.values, x, AL2.chain_all.values)
    d["cv_gain_all"] = g_
    s = AL2[AL2.pass2 == 0]; d["r_all_p1"] = rr(s[v].values.astype(float), s.resid_mean.values)[0]
    s = AL2[AL2.pass2 == 1]; d["r_all_p2"] = rr(s[v].values.astype(float), s.resid_mean.values)[0]
    # Y3 day change within episodes
    x = PR[v].values.astype(float); y = PR.Y3.values
    d["r_Y3"], d["rho_Y3"], d["n_Y3"] = rr(x, y)
    lo, hi = clus_ci(x, y, PR.ep_id.values); d["ci_Y3"] = f"[{lo:+.2f},{hi:+.2f}]"
    # regime: high vs normal AUC (descriptive; for N2)
    x = R[v].values.astype(float); lab = R.hi10.values; m = np.isfinite(x)
    d["auc_hi"] = stats.mannwhitneyu(x[m & (lab == 1)], x[m & (lab == 0)]).statistic / ((m & (lab == 1)).sum() * (m & (lab == 0)).sum())
    rows.append(d)
T = pd.DataFrame(rows).set_index("variable")

# family-wise permutation (record level, hi10 Y2) — max |r| over variables
Xm = HI2[VARS].values.astype(float); y = HI2.resid_mean.values
def allr(yv):
    out = np.full(len(VARS), np.nan)
    for j in range(len(VARS)):
        m = np.isfinite(Xm[:, j])
        if m.sum() >= 5 and np.std(Xm[m, j]) > 0: out[j] = np.corrcoef(Xm[m, j], yv[m])[0, 1]
    return out
obs = np.abs(allr(y)); mx = []
for _ in range(NB):
    mx.append(np.nanmax(np.abs(allr(rng.permutation(y)))))
mx = np.array(mx)
T["p_fw_Y2"] = [float(np.mean(mx >= o)) if np.isfinite(o) else np.nan for o in obs]
print(f"family-wise null (record permutation, hi10 Y2): 95% of max|r| = {np.percentile(mx, 95):.3f}")

# verdict (pre-fixed)
def verdict(d):
    ci_ex = np.isfinite(d.ci_lo) and (d.ci_lo > 0 or d.ci_hi < 0)
    ci_all_ex = np.isfinite(d.ci_all_lo) and (d.ci_all_lo > 0 or d.ci_all_hi < 0)
    if abs(d.r_Y2) >= .35 and ci_ex and d.cv_gain_Y2 > 0: return "signal"
    if (abs(d.r_Y2) >= .25 and ci_ex) or (d.cv_gain_Y2 > 0 and abs(d.r_Y2) >= .20) or (ci_all_ex and d.cv_gain_all > 0): return "weak"
    return "none"
T["verdict"] = T.apply(verdict, axis=1)
T["ci_Y2"] = [f"[{a:+.2f},{b:+.2f}]" for a, b in zip(T.ci_lo, T.ci_hi)]
T.reset_index().to_csv(os.path.join(HERE, "wb2_results_v1.csv"), index=False, encoding="utf-8-sig")

show = ["usability", "n_Y1", "r_Y1", "rho_Y1", "pr_Y1_season", "ci_Y1", "n_Y2", "r_Y2", "rho_Y2", "ci_Y2", "pr_Y2_season", "cv_gain_Y2", "n_impr_Y2",
        "p_fw_Y2", "verdict"]
print("\n=== per variable: Y1 (hi10 level) and Y2 (hi10 residual) ===")
print(T[show].round(3).sort_values("r_Y2", key=lambda s: -s.abs()).to_string())
show2 = ["r_Y2_F13", "r_Y2_F47", "r_Y2_p1", "r_Y2_p2(n5)", "n_Y2all", "r_Y2all", "ci_all", "cv_gain_all", "r_all_p1", "r_all_p2", "n_Y3", "r_Y3", "rho_Y3", "ci_Y3", "auc_hi"]
print("\n=== consistency, all-record residual, Y3 day change, regime AUC ===")
print(T[show2].round(3).sort_values("r_Y2all", key=lambda s: -s.abs()).to_string())
print("\nverdict counts:", T.verdict.value_counts().to_dict())

# ---------------- fingerprint checks P1/P2/P4 ----------------
ETV = ["et_total", "et_day", "rad_in_sum", "vpd_rad", "vpd_day", "rad_out_sum"]
print("\n=== fingerprints: transpiration proxies (A-type predicts r_Y1<0 and r_Y3<0; B-type the reverse) ===")
print(T.loc[ETV, ["r_Y1", "pr_Y1_season", "r_Y3", "r_Y2", "r_Y2all", "r_Y2_F13", "r_Y2_F47", "auc_hi"]].round(3).to_string())
MULTI = ["et_3link", "ep_mean_et_sofar", "rad_out_3d", "rad_out_7d", "dull_7d", "prev_et_total", "lag2id_et_total"]
print("=== accumulation (P2/P3): multi-day vs same-day ===")
print(T.loc[MULTI, ["usability", "r_Y1", "pr_Y1_season", "r_Y3", "r_Y2", "ci_Y2", "cv_gain_Y2", "r_Y2all"]].round(3).to_string())
sg = np.sign(T.r_Y2_F13) * np.sign(T.r_Y2_F47)
print(f"farm sign agreement on hi10 Y2: {int((sg > 0).sum())}/{int(np.isfinite(sg).sum())};  "
      f"pass1 vs pass2 sign agreement on all-record Y2: {int((np.sign(T.r_all_p1) * np.sign(T.r_all_p2) > 0).sum())}/{len(T)}")

# ---------------- hourly causal check (0..h running versions vs row residual at h) ----------------
H = pd.read_csv(os.path.join(DATA, "hourly.csv")); H["res"] = H.sub_ec - H.sg
RUN = pd.read_csv(os.path.join(HERE, "vars_running_v1.csv"))
SAMEV = [v for v in RUN.columns if v not in ("rec", "h")]
hi_set = set(HI2.index); al_set = set(AL2.index)
hc = []
for h in (5, 11, 17, 23):
    q = RUN[RUN.h == h].merge(H[H.hour == h][["rec", "res"]], on="rec")
    qh = q[q.rec.isin(hi_set)]; qa = q[q.rec.isin(al_set)]
    for v in SAMEV:
        a = rr(qh[v].values.astype(float), qh.res.values)[0]; b = rr(qa[v].values.astype(float), qa.res.values)[0]
        hc.append(dict(variable=v, h=h, r_hi=a, r_all=b))
HC = pd.DataFrame(hc)
P = HC.pivot(index="variable", columns="h", values="r_hi").add_prefix("hi_h")
P2 = HC.pivot(index="variable", columns="h", values="r_all").add_prefix("all_h")
HCt = P.join(P2)
HCt.reset_index().to_csv(os.path.join(HERE, "wb2_hourly_causal_v1.csv"), index=False, encoding="utf-8-sig")
print("\n=== causal running (0..h, ffill) feature vs row residual y_h - sg_h (hi10 n=%d; all n=%d) ===" % (len(hi_set), len(al_set)))
print(HCt.round(2).sort_values("hi_h23", key=lambda s: -s.abs()).to_string())

# ---------------- multivariate non-causal upper bound ----------------
def mv_logo(df, ycol, feats, base_col, groups, model):
    y = df[ycol].values; g = df[groups].values
    X = df[feats].values.astype(float); b = df[base_col].values.astype(float) if base_col else None
    pb, pm = np.zeros(len(y)), np.zeros(len(y))
    for u in np.unique(g):
        te = g == u; tr = ~te
        med = np.nanmedian(X[tr], 0); med = np.where(np.isfinite(med), med, 0)
        Xtr = np.where(np.isfinite(X[tr]), X[tr], med); Xte = np.where(np.isfinite(X[te]), X[te], med)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        if b is not None:
            Xtr = np.c_[Xtr, b[tr]]; Xte = np.c_[Xte, b[te]]
            lb = LinearRegression().fit(b[tr, None], y[tr]); pb[te] = lb.predict(b[te, None])
        else:
            pb[te] = y[tr].mean()
        if model == "ridge":
            m = RidgeCV(alphas=np.logspace(-1, 4, 26)).fit(Xtr, y[tr])
        else:
            m = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=3,
                                          random_state=7).fit(Xtr, y[tr])
        pm[te] = m.predict(Xte)
    sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pb) ** 2) / sst, 1 - np.sum((y - pm) ** 2) / sst, int(np.sum(np.abs(y - pm) < np.abs(y - pb))), len(y)

ALLV = VARS
USEV = [v for v in VARS if USAB[v] in ("U", "U5", "Uid")]
SAME_U = [v for v in VARS if USAB[v] in ("U", "U5")]
print("\n=== multivariate leave-one-chain-out (non-causal upper bound for the family) ===")
mvres = {}
for nm, feats in (("all47", ALLV), ("usable", USEV), ("same_day_U", SAME_U)):
    for model in ("ridge", "gbm"):
        r2b, r2m, imp, n = mv_logo(HI2, "resid_mean", feats, "sg_mean", "ep_id", model)
        mvres[(nm, model, "Y2hi")] = (r2b, r2m, imp, n)
        print(f"Y2 hi10 | {nm:10s} {model:5s}: R2cv baseline(a+b*sg) {r2b:+.3f} -> model {r2m:+.3f}  (gain {r2m - r2b:+.3f}); improved {imp}/{n}")
for nm, feats in (("all47", ALLV), ("usable", USEV)):
    for model in ("ridge", "gbm"):
        r2b, r2m, imp, n = mv_logo(HI, "ec_mean", feats, None, "ep_id", model)
        mvres[(nm, model, "Y1hi")] = (r2b, r2m, imp, n)
        print(f"Y1 hi10 level (no sg) | {nm:10s} {model:5s}: R2cv mean-only {r2b:+.3f} -> model {r2m:+.3f}; improved {imp}/{n}")
for nm, feats in (("all47", ALLV), ("usable", USEV)):
    for model in ("ridge", "gbm"):
        r2b, r2m, imp, n = mv_logo(AL2, "resid_mean", feats, "sg_mean", "chain_all", model)
        mvres[(nm, model, "Y2all")] = (r2b, r2m, imp, n)
        print(f"Y2 all records | {nm:10s} {model:5s}: R2cv baseline {r2b:+.3f} -> model {r2m:+.3f} (gain {r2m - r2b:+.3f}); improved {imp}/{n}")
PRm = PR.copy()
for model in ("ridge", "gbm"):
    r2b, r2m, imp, n = mv_logo(PRm, "Y3", ALLV, None, "ep_id", model)
    mvres[("all47", model, "Y3")] = (r2b, r2m, imp, n)
    print(f"Y3 within-episode day change | all47 {model:5s}: R2cv mean-only {r2b:+.3f} -> model {r2m:+.3f}; improved {imp}/{n}")

# GBM-on-noise null for the Y2 hi10 multivariate bound: permute y, same pipeline (50 reps)
nul = []
for k in range(50):
    D2 = HI2.copy(); D2["resid_mean"] = rng.permutation(D2.resid_mean.values)
    r2b, r2m, _, _ = mv_logo(D2, "resid_mean", ALLV, "sg_mean", "ep_id", "ridge"); nul.append(r2m - r2b)
print(f"null (permuted Y2, ridge all47) gain: mean {np.mean(nul):+.3f}, 95% {np.percentile(nul, 95):+.3f}")
pd.DataFrame([dict(set=k[0], model=k[1], target=k[2], r2_base=v[0], r2_model=v[1], improved=v[2], n=v[3]) for k, v in mvres.items()]).to_csv(
    os.path.join(HERE, "wb2_multivariate_v1.csv"), index=False, encoding="utf-8-sig")
print("saved:", os.path.join(HERE, "wb2_results_v1.csv"), os.path.join(HERE, "wb2_hourly_causal_v1.csv"), os.path.join(HERE, "wb2_multivariate_v1.csv"))
