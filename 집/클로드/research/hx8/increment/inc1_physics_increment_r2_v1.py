# -*- coding: utf-8 -*-
"""INC1 (2026-10-13 집 클로드, hx8/increment; 사용자: '고EC 8기록의 수준을 맞히는 방법 더 찾아봐').  서술(채택 판정 아님).
질문: 고EC 날의 '하루 증분'을 그날 입력에서 만든 증발산·물 수지 대리변수로 설명할 수 있나?
  목표 W  = EC[23] - EC[0]           (하루 안 변화; 같은 동 자정 차는 ≈ 0 이므로 하루 수준 변화의 본체)
       LR = log(EC[23]/EC[0])         (농축은 비례적이라는 물리 가정용, 전체 400 에서)
       DM = mean(EC_b) - mean(EC_a)   (같은 동 연속 날, 진단 연결 a1->b + 학습 전용 달력 dmt3e 로 하루 차 1 확인)
  대리변수(시간별 입력에서 계산, 하루 합/개수): 실내 포차 VPD(in_temp·in_hum), 낮/밤 VPD 합, 투과 일사(차광·보온 가중 3안),
    복사항 Priestley-Taylor 형 s/(s+γ)·Rin, 공기역학항 VPD×(1+환기/20+유동팬/50), 밀폐 시간(환기 0 / 환기 0 & 유동팬<5),
    밀폐 중 절대습도 상승 합(증산이 닫힌 공기에 쌓이는 양), 환기·유동팬·난방·CO2 공급·포그 합, 실내온도 23시-0시(센서 온도 영향 통제), 낮 길이.
  교차적합: 고EC 표본은 에피소드(진단 연결 성분) 단위 한 개씩 빼기, 전체 표본은 달력 5일 블록 단위 10겹.
    모형: (0) 상수(겹 밖 평균) — 기준, (1) 단변수 선형(변수별), (2) 겹 안에서 고른 최선 단변수, (3) 릿지(α 1/10/100, 전부 보고).
    R2cv = 1 - SSE(모형)/SSE(상수).  우연 기준: W 를 섞은 200회의 (2)·(3) R2cv 95% 분위.
규정: 학습 기록만 사용(정답·입력), 평가 기록 미사용.  DM11 미사용(학습 전용 달력 dmt3e_E0 만).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
from feat4 import EC, FARM, NUM, N, TRAIN, S3
import numpy as np, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
here = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
V = ["out_temp", "out_hum", "out_rad", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
raw = pd.concat([pd.read_csv(os.path.join(env.DATA, "train_X.csv")), pd.read_csv(os.path.join(env.DATA, "test_X.csv"))])
raw = raw[raw.row_id.str[:3].isin(["F13", "F47"])].copy(); raw["rec"] = raw.row_id.str[:7]; raw["h"] = raw.row_id.str[8:].astype(int)
A = np.full((N, 24, len(V)), np.nan)
for r_, g in raw.groupby("rec"):
    if r_ in idx: A[idx[r_], g.h.values] = g[V].values
# fill small gaps of indoor sensors within record (interpolation along hour; record-internal, used only for training-side descriptive regression)
for i in range(N):
    for j in range(len(V)):
        s = pd.Series(A[i, :, j])
        if s.isna().any() and s.notna().sum() >= 12: A[i, :, j] = s.interpolate(limit_direction="both").values
vi = {v: k for k, v in enumerate(V)}
g = lambda v: A[:, :, vi[v]]
T, RH, RAD = g("in_temp"), g("in_hum"), g("out_rad")
es = 0.6108 * np.exp(17.27 * T / (T + 237.3)); VPD = es * (1 - RH / 100)              # kPa
AH = 2.1674 * RH / 100 * es * 10 / (273.15 + T)                                        # g/m3 (≈ 216.74·e/T, e in kPa*10 -> hPa)
s_ = 4098 * es / (T + 237.3) ** 2; gam = 0.066
light = RAD > 10
vent, circ, shade, therm = g("act_vent"), g("act_circfan"), g("act_shade"), g("act_thermal")
def tau(ks, kt): return (1 - ks * shade / 100) * (1 - kt * therm / 100)
FE = {}
FE["rad_out"] = np.nansum(RAD, 1)
for nm, ks, kt in [("rin_a", .5, .3), ("rin_b", .75, .5), ("rin_c", .9, .7)]:
    FE[nm] = np.nansum(RAD * tau(ks, kt), 1)
FE["et_rad"] = np.nansum(s_ / (s_ + gam) * RAD * tau(.75, .5), 1)
gv = 1 + vent / 20 + circ / 50
FE["et_aero"] = np.nansum(VPD * gv, 1)
FE["vpd_day"] = np.nansum(np.where(light, VPD, 0), 1)
FE["vpd_night"] = np.nansum(np.where(~light, VPD, 0), 1)
FE["vpd_mean"] = np.nanmean(VPD, 1)
FE["sealed_v"] = np.nansum(vent < 1, 1).astype(float)
FE["sealed_vc"] = np.nansum((vent < 1) & (circ < 5), 1).astype(float)
dAH = np.diff(AH, axis=1, prepend=AH[:, :1])
FE["ah_rise_sealed"] = np.nansum(np.where((vent < 1) & (dAH > 0), dAH, 0), 1)
FE["ah_rise_day"] = np.nansum(np.where(light & (dAH > 0), dAH, 0), 1)
FE["hum_range"] = np.nanmax(RH, 1) - np.nanmin(RH, 1)
for v in ["act_vent", "act_circfan", "act_heating", "act_co2", "act_fog"]: FE[v + "_sum"] = np.nansum(g(v), 1)
FE["dT_in"] = T[:, 23] - T[:, 0]
FE["T_in_mean"] = np.nanmean(T, 1)
FE["daylen"] = np.nansum(light, 1).astype(float)
FE["et_combo"] = FE["et_rad"] / np.nanstd(FE["et_rad"][TRAIN]) + FE["et_aero"] / np.nanstd(FE["et_aero"][TRAIN])
F = pd.DataFrame(FE, index=[rid(i) for i in range(N)])
F.to_csv(os.path.join(here, "inc1_features_v1.csv"))
FEATS = list(F.columns)
# ---------- targets
CAL = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); pre = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
tr = [rid(i) for i in range(N) if TRAIN[i]]
y = pd.Series({r: float(np.nanmean(EC[idx[r]])) for r in tr})
W = pd.Series({r: EC[idx[r], 23] - EC[idx[r], 0] for r in tr})
LR = pd.Series({r: np.log(EC[idx[r], 23] / EC[idx[r], 0]) for r in tr})
def consec(a, b):
    return a in CAL.index and b in CAL.index and CAL.seg[a] == CAL.seg[b] and CAL.day[b] == CAL.day[a] + 1
# episode groups = connected components over predecessor links (training)
par = {r: r for r in tr}
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for b, a in pre.items():
    if a in par and b in par and consec(a, b): par[fd(b)] = fd(a)
comp = pd.Series({r: fd(r) for r in tr})
blk = pd.Series({r: (f"s{CAL.seg[r]}_{CAL.day[r] // 5}" if r in CAL.index else "nd_" + r) for r in tr})

def cv_r2(Xm, t, groups, model, alpha=10.0):
    """leave-group-out predictions; returns R2 vs out-of-fold constant and OOF preds."""
    t = np.asarray(t, float); P = np.full(len(t), np.nan); C = np.full(len(t), np.nan)
    ug = pd.unique(groups)
    if len(ug) > 12:   # 10 folds of groups
        rng = np.random.default_rng(7); perm = rng.permutation(ug); fold = {gg: k % 10 for k, gg in enumerate(perm)}
        fid = np.array([fold[gg] for gg in groups])
    else:
        fid = pd.factorize(groups)[0]
    for f in np.unique(fid):
        te = fid == f; trn = ~te
        C[te] = t[trn].mean()
        if model == "const": P[te] = t[trn].mean(); continue
        Xtr, Xte = Xm[trn], Xm[te]
        mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
        Ztr, Zte = np.nan_to_num((Xtr - mu) / sd), np.nan_to_num((Xte - mu) / sd)
        if model == "best1":   # choose best single feature by in-fold correlation
            cc = [abs(np.corrcoef(Ztr[:, j], t[trn])[0, 1]) if np.std(Ztr[:, j]) > 0 else 0 for j in range(Ztr.shape[1])]
            j = int(np.nanargmax(cc)); Ztr, Zte = Ztr[:, [j]], Zte[:, [j]]; alpha_ = 0.0
        else:
            alpha_ = alpha
        Zt = np.c_[np.ones(len(Ztr)), Ztr]; Ie = np.eye(Zt.shape[1]); Ie[0, 0] = 0
        w = np.linalg.solve(Zt.T @ Zt + alpha_ * Ie, Zt.T @ t[trn])
        P[te] = np.c_[np.ones(len(Zte)), Zte] @ w
    return 1 - np.sum((t - P) ** 2) / np.sum((t - C) ** 2), P

def block(name, recs, t, groups, nperm=200):
    recs = list(recs); Xm = F.loc[recs, FEATS].values.astype(float); t = np.asarray(t, float); groups = np.asarray(groups)
    print(f"\n=== {name}: n={len(recs)}, groups={len(pd.unique(groups))}, mean {t.mean():+.3f}, sd {t.std():.3f}")
    res = {}
    res["best1"] = cv_r2(Xm, t, groups, "best1")[0]
    for a in (1.0, 10.0, 100.0): res[f"ridge{a:g}"] = cv_r2(Xm, t, groups, "ridge", a)[0]
    uni = {}
    for j, f in enumerate(FEATS):
        uni[f] = cv_r2(Xm[:, [j]], t, groups, "ridge", 0.0)[0]
    rng = np.random.default_rng(11); null = {k: [] for k in res}
    for _ in range(nperm):
        tp = rng.permutation(t)
        null["best1"].append(cv_r2(Xm, tp, groups, "best1")[0])
        for a in (1.0, 10.0, 100.0): null[f"ridge{a:g}"].append(cv_r2(Xm, tp, groups, "ridge", a)[0])
    for k in res:
        q95 = np.quantile(null[k], .95); p = (np.sum(np.array(null[k]) >= res[k]) + 1) / (nperm + 1)
        print(f"   {k:9s} R2cv {res[k]:+.3f} | chance 95% {q95:+.3f} | p {p:.3f}")
    rho = {f: pd.Series(F.loc[recs, f].values).corr(pd.Series(t), method="spearman") for f in FEATS}
    U = pd.DataFrame(dict(R2cv_uni=uni, rho=rho)).sort_values("R2cv_uni", ascending=False)
    print("   top single-feature R2cv (linear, leave-group-out) and Spearman rho:")
    print(U.head(8).round(3).to_string())
    return res, U

# (1) high-EC records: within-day change W
H = [r for r in tr if y[r] >= 1.0]
resH, UH = block("HIGH (day mean>=1) W=EC23-EC0", H, W[H], comp[H])
# (1b) high-EC records relative change
block("HIGH LR=log(EC23/EC0)", H, LR[H], comp[H])
# (2) consecutive same-dong pairs with a high: DM = y_b - y_a, features = inputs of b (and a's second half is in W_a; check identity)
P = [(a, b) for b, a in pre.items() if a in y.index and b in y.index and consec(a, b)]
PH = [(a, b) for a, b in P if y[a] >= 1.0]
DM = np.array([y[b] - y[a] for a, b in PH]); approx = np.array([(W[a] + W[b]) / 2 + (EC[idx[b], 0] - EC[idx[a], 23]) for a, b in PH])
print(f"\n(identity check) high pairs {len(PH)}: DM vs (W_a+W_b)/2 + midnight jump  corr {np.corrcoef(DM, approx)[0,1]:.3f}, rmse {np.sqrt(np.mean((DM-approx)**2)):.3f}; "
      f"mean DM {DM.mean():+.3f}, median {np.median(DM):+.3f}; mean W(high) {W[H].mean():+.3f}, median {W[H].median():+.3f}; midnight jump mean {np.mean([EC[idx[b],0]-EC[idx[a],23] for a,b in PH]):+.4f}")
# features for pair = average of a and b features (the level change spans a's afternoon and b's morning)
Fp = pd.DataFrame([(F.loc[a, FEATS].values + F.loc[b, FEATS].values) / 2 for a, b in PH], columns=FEATS, index=[b for a, b in PH])
F_backup = F; F = pd.concat([F, Fp.add_prefix("")]).groupby(level=0).last()   # pair rows keyed by b (b's own row replaced by pair-average only inside this block)
block("HIGH pairs DM=y_b-y_a (pair-avg inputs)", [b for a, b in PH], DM, comp[[b for a, b in PH]].values)
F = F_backup
# (3) all training records (physics should hold for all if concentration by uptake): W and LR
block("ALL train W", tr, W[tr], blk[tr].values, nperm=100)
resA, UA = block("ALL train LR", tr, LR[tr], blk[tr].values, nperm=100)
# (4) sealed-ish non-high records (same operating state but normal EC) — is W different?
S = [r for r in tr if F.loc[r, "sealed_vc"] >= 18]
print(f"\nsealed (vent 0 & circfan<5 >= 18 h) training records: {len(S)}; W mean high {W[[r for r in S if y[r]>=1]].mean():+.3f} (n {sum(y[r]>=1 for r in S)}) vs normal {W[[r for r in S if y[r]<1]].mean():+.3f} (n {sum(y[r]<1 for r in S)})")
print("W(high) distribution:", np.round(np.quantile(W[H], [0, .1, .25, .5, .75, .9, 1]), 3))
OUT = pd.DataFrame(dict(y=y, W=W, LR=LR, comp=comp, blk=blk)); OUT.to_csv(os.path.join(here, "inc1_targets_v1.csv"))
