# -*- coding: utf-8 -*-
"""VW1 (verify_water) — 물 수지·증산 가족 보고를 반박해 보는 검증 (2026-10-13 집 클로드, 적대적 검증자).
대상 보고: hx9/water/wb1~wb3. 판정 weak 2개(dah_closed_day, hum_drop_morning) + 다변량 상한 + 가장 그럴듯하다던 단서 lag2id_et_total.
반박 시도 (실행 전에 정함):
  A  재현: r_Y2, CV 이득이 wb2 와 같은지.
  B  사슬(에피소드) 하나 빼기: r 부호·크기, CV 이득, 판정이 몇 번 유지되는지.
  C  에피소드 안(평균 뺀) 상관 vs 에피소드 평균끼리(n=8) 상관 — 동(사슬) 정체성 혼동 검사.
  D  모델 예측과 이중 계산: r(var, sg), r(var, ec), 편상관 r(var, ec | sg). hi10 선택(정답 조건)이 sg-잔차 음상관을 만드는지.
  E  강건성: 기록 하나 빼기 CV 이득, 영향 큰 기록 2개 빼기.
  F  다중검정 보정: 사전 판정 규칙(hi10 가지 1·2)을 귀무 자료에 그대로 돌려 '우연히 weak 이상이 몇 개 나오나'.
       (i) 기록 순열(X 행만 섞음, (잔차, sg, 에피소드) 묶음 유지)  (ii) 에피소드 블록 재배치(X 의 사슬 구조 유지).
  G  평가 행 사용 가능성: 평가 기록(test_X)에서 변수가 정의되는 비율과 시각.
  H  lag2id_et_total: num-2 가 실제 clean 전날인 링크 vs 아닌 링크로 나눠 Y3 편상관 (전날 증산 기전이면 '전날 맞음'에서 더 강해야).
판정 기본값: 불확실하면 artifact / inconclusive.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
HERE = os.path.dirname(os.path.abspath(__file__))
HX9 = os.path.dirname(HERE)
rng = np.random.default_rng(31013)

R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HX9, "water", "vars_v1.csv")).set_index("rec").loc[R.index]
CD = pd.read_csv(os.path.join(HX9, "water", "vars_v1_columns.csv")); VARS = list(CD.variable); USAB = dict(zip(CD.variable, CD.usability))
R = R.join(F)
HI2 = R[(R.hi10 == 1) & (R.has_sg == 1)].copy().sort_values(["ep_id", "ep_pos"])
print(f"HI2 n={len(HI2)} episodes={HI2.ep_id.nunique()} sizes={HI2.ep_id.value_counts().sort_index().to_dict()}")
y = HI2.resid_mean.values; sg = HI2.sg_mean.values; ec = HI2.ec_mean.values; ep = HI2.ep_id.values.astype(int)
print(f"r(sg, resid) within hi10 = {np.corrcoef(sg, y)[0,1]:+.3f}  (selection on label: ec>=1.0 truncates; any var tied to sg gets opposite-sign resid r)")


def logo_gain(yv, base, x, g):
    m = np.isfinite(yv) & np.isfinite(base) & np.isfinite(x)
    yv, base, g, xx = yv[m], base[m], np.asarray(g)[m], x[m]
    pb, pa = np.zeros_like(yv), np.zeros_like(yv)
    for u in np.unique(g):
        te = g == u; tr = ~te
        Xb = np.c_[np.ones(tr.sum()), base[tr]]; cb = np.linalg.lstsq(Xb, yv[tr], rcond=None)[0]
        pb[te] = np.c_[np.ones(te.sum()), base[te]] @ cb
        Xa = np.c_[Xb, xx[tr]]; ca = np.linalg.lstsq(Xa, yv[tr], rcond=None)[0]
        pa[te] = np.c_[np.ones(te.sum()), base[te], xx[te]] @ ca
    sst = np.sum((yv - yv.mean()) ** 2)
    return (np.sum((yv - pb) ** 2) - np.sum((yv - pa) ** 2)) / sst


def boot_W(g, B):
    ug = np.unique(g); pos = {u: k for k, u in enumerate(ug)}; gi = np.array([pos[u] for u in g])
    C = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(B)])
    return C[:, gi].astype(float)


def wcorr(W, x, yv):
    sw = W.sum(1); ok0 = sw > 0
    mx = (W @ x) / np.where(ok0, sw, 1); my = (W @ yv) / np.where(ok0, sw, 1)
    dx = x[None, :] - mx[:, None]; dy = yv[None, :] - my[:, None]
    cov = (W * dx * dy).sum(1); vx = (W * dx * dx).sum(1); vy = (W * dy * dy).sum(1)
    ok = ok0 & (vx > 1e-12) & (vy > 1e-12)
    return cov[ok] / np.sqrt(vx[ok] * vy[ok])


def stats_one(x, yv, base, g, W):
    m = np.isfinite(x)
    if m.sum() < 5 or np.std(x[m]) < 1e-12: return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(x[m], yv[m])[0, 1]
    bs = wcorr(W[:, m], x[m], yv[m])
    lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if len(bs) > 100 else (np.nan, np.nan)
    return r, lo, hi, logo_gain(yv, base, x, g)


def verdict(r, lo, hi, gain):
    if not np.isfinite(r): return "none"
    ex = np.isfinite(lo) and (lo > 0 or hi < 0)
    if abs(r) >= .35 and ex and gain > 0: return "signal"
    if (abs(r) >= .25 and ex) or (gain > 0 and abs(r) >= .20): return "weak"
    return "none"


XM = HI2[VARS].values.astype(float)
FOC = ["dah_closed_day", "hum_drop_morning", "lag2id_et_total", "heating_sum", "ahx_day"]

# ---------------- A. reproduce ----------------
W0 = boot_W(ep, 2000)
obs = {v: stats_one(XM[:, j], y, sg, ep, W0) for j, v in enumerate(VARS)}
obs_v = {v: verdict(*obs[v]) for v in VARS}
print("\n=== A. reproduction (hi10 Y2) ===")
for v in FOC:
    r, lo, hi, gn = obs[v]; print(f"  {v:18s} r {r:+.3f} CI [{lo:+.2f},{hi:+.2f}] LOCO gain {gn:+.3f} -> {obs_v[v]}")
n_obs = sum(obs_v[v] in ("weak", "signal") for v in VARS)
print(f"  observed weak/signal count (hi10 branches) = {n_obs}: {[v for v in VARS if obs_v[v] != 'none']}")

# ---------------- B. drop one episode ----------------
print("\n=== B. drop-one-episode (r, LOCO gain, verdict on the rest) ===")
for v in FOC[:3]:
    x = HI2[v].values.astype(float); out = []
    for e in np.unique(ep):
        k = ep != e
        Wk = boot_W(ep[k], 1000)
        r, lo, hi, gn = stats_one(x[k], y[k], sg[k], ep[k], Wk)
        out.append((e, int((ep == e).sum()), r, lo, hi, gn, verdict(r, lo, hi, gn)))
    D = pd.DataFrame(out, columns=["drop_ep", "n_drop", "r", "lo", "hi", "gain", "verdict"])
    print(f"  {v}:"); print(D.round(3).to_string(index=False))
    print(f"    verdict kept weak/signal in {int(D.verdict.isin(['weak','signal']).sum())}/{len(D)} drops; gain>0 in {int((D.gain>0).sum())}/{len(D)}")

# ---------------- C. within vs between episode ----------------
print("\n=== C. within-episode (demeaned) vs between-episode (episode means) ===")
for v in FOC:
    d = HI2[[v, "resid_mean", "ep_id", "ec_mean"]].dropna()
    cnt = d.groupby("ep_id")[v].transform("size"); d = d[cnt >= 2]
    xw = d[v] - d.groupby("ep_id")[v].transform("mean"); yw = d.resid_mean - d.groupby("ep_id").resid_mean.transform("mean")
    rw = np.corrcoef(xw, yw)[0, 1]
    Wc = boot_W(d.ep_id.values, 2000); bs = wcorr(Wc, xw.values, yw.values)
    em = HI2.groupby("ep_id")[[v, "resid_mean"]].mean().dropna()
    rb = np.corrcoef(em[v], em.resid_mean)[0, 1]
    print(f"  {v:18s} within r {rw:+.3f} (n {len(d)}, CI [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]) | "
          f"between (episode means, n {len(em)}) r {rb:+.3f}")

# ---------------- D. double counting with model / selection ----------------
print("\n=== D. model double-counting: r(var,sg), r(var,ec), partial r(var, ec | sg) within hi10 ===")
def pc(x, a, z):
    m = np.isfinite(x) & np.isfinite(a) & np.isfinite(z)
    Z = np.c_[np.ones(m.sum()), z[m]]
    ex = x[m] - Z @ np.linalg.lstsq(Z, x[m], rcond=None)[0]; ea = a[m] - Z @ np.linalg.lstsq(Z, a[m], rcond=None)[0]
    return np.corrcoef(ex, ea)[0, 1]
for v in FOC:
    x = HI2[v].values.astype(float); m = np.isfinite(x)
    print(f"  {v:18s} r(var,sg) {np.corrcoef(x[m], sg[m])[0,1]:+.3f}  r(var,ec) {np.corrcoef(x[m], ec[m])[0,1]:+.3f}  "
          f"pr(var,ec|sg) {pc(x, ec, sg):+.3f}  r(var,resid) {np.corrcoef(x[m], y[m])[0,1]:+.3f}")

# ---------------- E. robustness ----------------
print("\n=== E. leave-one-record-out gain; drop 2 most influential records ===")
for v in FOC[:3]:
    x = HI2[v].values.astype(float); m = np.isfinite(x)
    g_loro = logo_gain(y, sg, x, np.arange(len(y)))
    # influence: change in r when each record removed
    idx = np.where(m)[0]; base_r = np.corrcoef(x[m], y[m])[0, 1]; infl = []
    for i in idx:
        k = m.copy(); k[i] = False; infl.append(np.corrcoef(x[k], y[k])[0, 1] - base_r)
    infl = np.array(infl); top = idx[np.argsort(-np.abs(infl))[:2]]
    k = m.copy(); k[top] = False
    r2 = np.corrcoef(x[k], y[k])[0, 1]; g2 = logo_gain(y[k], sg[k], x[k], ep[k])
    print(f"  {v:18s} LORO gain {g_loro:+.3f} | drop {list(HI2.index[top])} -> r {r2:+.3f}, LOCO gain {g2:+.3f} | "
          f"spearman {stats.spearmanr(x[m], y[m])[0]:+.3f} kendall {stats.kendalltau(x[m], y[m])[0]:+.3f}")

# ---------------- F. null calibration of the verdict rule ----------------
print("\n=== F. null: how many of the 47 variables get weak/signal by chance under the pre-fixed rule (hi10 branches) ===")
eps = np.unique(ep); members = {e: np.where(ep == e)[0] for e in eps}
def count_null(Xn, B=300):
    W = boot_W(ep, B); c = 0; c2 = 0
    for j in range(Xn.shape[1]):
        r, lo, hi, gn = stats_one(Xn[:, j], y, sg, ep, W); v = verdict(r, lo, hi, gn)
        if v != "none": c += 1
        if np.isfinite(r) and gn > 0 and abs(r) >= .20: c2 += 1
    return c, c2
NREP = 400
rec_null, blk_null = [], []
for k in range(NREP):
    rec_null.append(count_null(XM[rng.permutation(len(y))]))
    perm = rng.permutation(eps); Xb = np.empty_like(XM)
    for e, s in zip(eps, perm):
        tgt, src = members[e], members[s]; off = rng.integers(len(src))
        for q, i in enumerate(tgt): Xb[i] = XM[src[(q + off) % len(src)]]
    blk_null.append(count_null(Xb))
for nm, arr in (("record permutation", np.array(rec_null)), ("episode-block", np.array(blk_null))):
    print(f"  {nm:18s}: weak/signal count mean {arr[:,0].mean():.2f}, median {np.median(arr[:,0]):.0f}, P(count>={n_obs}) = {np.mean(arr[:,0] >= n_obs):.2f}; "
          f"branch-2 (gain>0 & |r|>=.2) mean {arr[:,1].mean():.2f}")

# ---------------- G. eval-row availability ----------------
print("\n=== G. eval-row availability (test_X inputs, descriptive) ===")
TX = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TX = TX[TX.row_id.str[:3].isin(["F13", "F47"])].copy()
TX["rec"] = TX.row_id.str[:7]; TX["hour"] = TX.row_id.str[8:].astype(int)
def es(T): return 0.6108 * np.exp(17.27 * T / (T + 237.3))
def ahum(T, RH): return 2165.0 * (np.clip(RH, 1, 100) / 100.0) * es(T) / (T + 273.15)
HRS = np.arange(24); DAY = (HRS >= 9) & (HRS <= 16)
def dah_hdrop(g):
    g = g.set_index("hour").reindex(range(24)).ffill().bfill()
    T, RH, vent, fog = g.in_temp.values, g.in_hum.values, g.act_vent.values, g.act_fog.values
    AH = ahum(T, RH); dAH = np.r_[np.nan, np.diff(AH)]
    mm = DAY & (vent < 5) & (fog <= 0)
    first2 = np.where(mm)[0][1] if mm.sum() >= 2 else np.nan
    return pd.Series(dict(dah=np.nanmean(dAH[mm]) if mm.sum() >= 2 else np.nan, dah_first_hour=first2,
                          hdrop=RH[6] - np.nanmin(RH[7:14])))
EV = TX.groupby("rec").apply(dah_hdrop)
print(f"  eval records {len(EV)}: dah_closed_day defined {int(EV.dah.notna().sum())} ({EV.dah.notna().mean():.0%}); "
      f"training all {F.dah_closed_day.notna().mean():.0%}, training hi10 {R[R.hi10==1].dah_closed_day.notna().mean():.0%}")
print(f"  eval: hour at which dah_closed_day first becomes computable (2nd qualifying hour): {EV.dah_first_hour.value_counts().sort_index().to_dict()}")
print("  -> for an eval row at hour h < that hour the variable does not exist; hum_drop_morning partial from h=7, complete at h=13.")

# ---------------- H. lag2id_et_total: is num-2 really the previous day? ----------------
print("\n=== H. lag2id_et_total vs Y3, split by whether num-2 is the clean previous day ===")
lag2 = {r: f"{r[:3]}_{int(r[4:]) - 2:03d}" for r in R.index}
R["lag2_is_prev"] = [int(isinstance(p, str) and p == lag2[r]) for r, p in R.prev_rec.items()]
PR = R[(R.hi10 == 1) & R.prev_rec.notna()].copy()
PR = PR[[R.at[p, "hi10"] == 1 and R.at[p, "ep_id"] == R.at[b, "ep_id"] for b, p in PR.prev_rec.items()]]
AP = R[R.prev_rec.notna()].copy()
def pcor_df(d, x, a, z):
    d = d[[x, a] + z].dropna()
    if len(d) < 5 or d[x].std() < 1e-12: return np.nan, len(d)
    Z = np.c_[np.ones(len(d)), d[z].values]
    ex = d[x].values - Z @ np.linalg.lstsq(Z, d[x].values, rcond=None)[0]; ea = d[a].values - Z @ np.linalg.lstsq(Z, d[a].values, rcond=None)[0]
    return np.corrcoef(ex, ea)[0, 1], len(d)
for nm, d in (("hi10 within-episode pairs", PR), ("all clean pairs", AP)):
    for flag in (1, 0):
        s = d[d.lag2_is_prev == flag]
        pr_, n_ = pcor_df(s, "lag2id_et_total", "ec_mean", ["prev_ec_mean"])
        print(f"  {nm:26s} num-2 {'IS' if flag else 'NOT'} prev day: partial r(lag2id_et, EC_b | EC_a) {pr_:+.3f} (n {n_})")
print("  drop-one-episode (hi10 pairs) partial r:")
for e in np.unique(PR.ep_id):
    pr_, n_ = pcor_df(PR[PR.ep_id != e], "lag2id_et_total", "ec_mean", ["prev_ec_mean"])
    print(f"    drop ep {int(e)} (n_drop {int((PR.ep_id==e).sum())}): {pr_:+.3f} (n {n_})")
