# -*- coding: utf-8 -*-
"""PP1 (2026-10-13 집 클로드, hx8/physproxy; 사용자: '고EC 8기록의 수준을 맞히는 방법 더 찾아봐').  진단·서술.
가설: 고EC 동(밀폐 운영)의 증산량 대리지표(물 수지·에너지 수지·CO2 소비에서 계산한 '파생 물리량')가
      (a) 고EC 하루 수준, (b) 현재 모델(OOF CUR sg) 잔차를 설명한다.
6.697(HS4) 과 다른 점: HS4 는 13개 원시 변수의 하루/0~5시 단순 평균 26종. 여기서는
  절대습도(실내 온·습도의 비선형 결합)·실내외 절대습도 차·밀폐 낮 시간 절대습도 상승률·물 수지 증산 추정(환기 손실 보정)·
  일사 대비 승온(난방 없는 낮)·잠열/현열 비(ΔAH/ΔT)·ln AH 의 온도 기울기·무공급 밀폐 낮 CO2 하강률·이슬점 여유(결로 시간)·
  밤 절대습도 하강(결로 제거) 등 '조건부 비율·변화율'. CO2 하강률은 6.116(H19, 코덱스: 모델 특징으로 넣어 악화)과 같은 계열 —
  여기서는 고EC 표본의 수준·잔차 설명력만 본다(다른 질문).
판정(서술, 사전 고정): 표본 = (i) S_pred: DIAG10 OOF 하루 평균 예측 ≥ 0.9 (평가 8기록처럼 '예측으로 고른' 표본, 선택 편향 없음),
  (ii) S_true: 정답 하루 평균 ≥ 1.0 (HS4 와 같은 표본, 참고).  사례 단위 = 같은 동 연속 사슬(truth.pkl a1→b, 표본 안) 에피소드.
  교차적합 = 에피소드 하나 빼기(LOEO).  비교: q(현재 모델) / q+절편(고예측 평균 잔차, 이미 닫힌 '펴기'류) / q+절편+β·f.
  설명력 기준: f 의 LOEO RMSE 가 q+절편보다 낮고, 그 개선이 순열 최대통계(특징 전체 중 최대 개선) 95% 분위를 넘을 것
  (기록 순열 + 에피소드 블록 순열 두 가지, 더 엄한 쪽 적용).  넘으면 인과판(0..h)·검증기·시드 일관성 확인 후 평가 8기록 추정.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261013)
pd.set_option("display.width", 220)

# ---------------- data ----------------
def load(fn):
    D = pd.read_csv(os.path.join(env.DATA, fn)); D = D[D.row_id.str[:3].isin(["F13", "F47"])].copy()
    D["rec"] = D.row_id.str[:7]; D["h"] = D.row_id.str[8:].astype(int); return D.sort_values(["rec", "h"])
TX = load("train_X.csv"); EX = load("test_X.csv")
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["rec"] = Y.row_id.str[:7]; Y["h"] = Y.row_id.str[8:].astype(int)
ymat = Y.pivot(index="rec", columns="h", values="sub_ec")
V = ["out_temp", "out_hum", "out_rad", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]

def arrays(D, causal):
    out = {}
    for rec, g in D.groupby("rec"):
        g = g.set_index("h").reindex(range(24))[V]
        g = g.ffill() if causal else g.interpolate(limit_direction="both")
        out[rec] = {v: g[v].values.astype(float) for v in V}
    return out
A_tr = arrays(TX, causal=False)   # 학습 입력 보수: 앞뒤 사용 허용(10-10)
A_ev = arrays(EX, causal=True)    # 평가: 앞 시각만

def es(T): return 0.6108 * np.exp(17.27 * T / (T + 237.3))
def ah(T, RH): return 2165.0 * (RH / 100.0) * es(T) / (T + 273.15)   # g/m3

def nanmean(x): x = np.asarray(x, float); return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
def slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.nanstd(x[m]) < 1e-9: return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])

FEATS = ["AH", "AHx", "AHx_day", "AHx_night", "dAH_closed_day", "AH_rise", "Ebal05", "Ebal2", "Ebal2_per_rad",
         "dT_per_rad", "bowen_inv", "dlnAH_dT", "co2_draw", "co2_draw_per_rad", "co2_drop", "vpd_day", "vpd_x_rad",
         "radE_sum", "cond_hours", "AH_night_slope", "vent_resp"]

def feats(a, h):
    """features from hours 0..h only (h=23 -> full day)."""
    s = slice(0, h + 1)
    T, RH, To, RHo = a["in_temp"][s], a["in_hum"][s], a["out_temp"][s], a["out_hum"][s]
    rad, C, vent, shade = a["out_rad"][s], a["in_co2"][s], a["act_vent"][s], a["act_shade"][s]
    heat, co2a, fog = a["act_heating"][s], a["act_co2"][s], a["act_fog"][s]
    AHi, AHo = ah(T, RH), ah(To, np.clip(RHo, 1, 100))
    radE = rad * (1 - 0.5 * shade / 100.0)            # 차광 50% 투과 가정
    day = rad > 50; night = rad <= 10
    dAH = np.r_[np.nan, np.diff(AHi)]; dC = np.r_[np.nan, np.diff(C)]
    closed = vent < 5; nofog = fog <= 0
    f = {}
    f["AH"] = nanmean(AHi)
    f["AHx"] = nanmean(AHi - AHo)
    f["AHx_day"] = nanmean((AHi - AHo)[day]) if day.any() else np.nan
    f["AHx_night"] = nanmean((AHi - AHo)[night]) if night.any() else np.nan
    m = day & closed & nofog
    f["dAH_closed_day"] = nanmean(dAH[m]) if m.sum() >= 2 else np.nan
    if day.any():
        k0 = int(np.argmax(day)); f["AH_rise"] = float(np.nanmax(AHi[k0:]) - AHi[k0])
    else: f["AH_rise"] = np.nan
    for c, nm in [(0.5, "Ebal05"), (2.0, "Ebal2")]:      # 물 수지: dAH + c·환기율·(AH실내−AH외기)  (g/m3/h, 낮 합)
        e = dAH + c * (vent / 100.0) * (AHi - AHo)
        f[nm] = float(np.nansum(e[day & nofog])) if (day & nofog).sum() >= 2 else np.nan
    f["Ebal2_per_rad"] = f["Ebal2"] / (np.nansum(radE[day]) / 1000 + 1e-3) if np.isfinite(f["Ebal2"]) else np.nan
    mh = day & (heat < 5)
    f["dT_per_rad"] = float(np.nansum((T - To)[mh]) / (np.nansum(radE[mh]) / 100 + 1e-3)) if mh.sum() >= 2 else np.nan
    if day.any():
        k0 = int(np.argmax(day)); kT = k0 + int(np.nanargmax(T[k0:])) if np.isfinite(T[k0:]).any() else k0
        dTT = T[kT] - T[k0]; f["bowen_inv"] = float((AHi[kT] - AHi[k0]) / dTT) if dTT > 1 else np.nan
    else: f["bowen_inv"] = np.nan
    f["dlnAH_dT"] = slope(T[day], np.log(AHi[day])) if day.sum() >= 3 else np.nan
    m = day & closed & (co2a <= 0)
    f["co2_draw"] = nanmean(dC[m]) if m.sum() >= 2 else np.nan
    f["co2_draw_per_rad"] = f["co2_draw"] / (nanmean(radE[m]) / 100 + 1e-3) if np.isfinite(f["co2_draw"]) else np.nan
    if day.any():
        k0 = int(np.argmax(day)); f["co2_drop"] = float(np.nanmax(C[:k0 + 1]) - np.nanmin(C[k0:]))
    else: f["co2_drop"] = np.nan
    vpd = es(T) * (1 - RH / 100.0)
    f["vpd_day"] = nanmean(vpd[day]) if day.any() else np.nan
    f["vpd_x_rad"] = float(np.nansum((vpd * radE)[day]) / 1000) if day.any() else 0.0
    f["radE_sum"] = float(np.nansum(radE[day])) if day.any() else 0.0
    e_ = (RH / 100.0) * es(T); lg = np.log(np.clip(e_, 1e-3, None) / 0.6108); Td = 237.3 * lg / (17.27 - lg)
    f["cond_hours"] = float(np.nansum((T - Td) < 1.5))
    mn = night & (np.arange(h + 1) <= 8)
    f["AH_night_slope"] = nanmean(dAH[mn]) if mn.sum() >= 2 else np.nan
    dv = np.r_[np.nan, np.diff(vent)]; m = dv > 5
    f["vent_resp"] = nanmean(dAH[m]) if m.any() else np.nan
    return f

def table(A, h):
    return pd.DataFrame({r: feats(a, h) for r, a in A.items()}).T[FEATS]

# ---------------- OOF & samples ----------------
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
TL = pd.read_pickle(os.path.join(env.LOCAL, "linker_v3", "truth.pkl"))
import pickle
R = pickle.load(open(os.path.join(env.LOCAL, "linker_v3", "base.pkl"), "rb")); rr = R["recs"].reset_index(drop=True)
nm = [f"{f}_{int(d):03d}" for f, d in zip(rr.farm, rr.day)]
links = [(nm[int(a)], nm[int(b)]) for a, b in zip(TL.a1, TL.b)]
yday = ymat.mean(1)

def episodes(recs):
    recs = set(recs); par = {r: r for r in recs}
    def fd(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in links:
        if a in recs and b in recs: par[fd(a)] = fd(b)
    return {r: fd(r) for r in recs}

def loeo(y, q, f, groups):
    """returns LOEO predictions for: intercept-only residual model and intercept+slope on f (f standardised inside fold)."""
    r = y - q; p0 = np.zeros_like(y); p1 = np.zeros_like(y)
    for g in np.unique(groups):
        tr = groups != g; te = ~tr
        p0[te] = q[te] + r[tr].mean()
        ft = f[tr]; ok = np.isfinite(ft)
        mu = np.nanmedian(ft) if ok.any() else 0.0
        fi = np.where(np.isfinite(f), f, mu)
        if ok.sum() >= 5 and np.std(fi[tr]) > 1e-9:
            b = np.polyfit(fi[tr], r[tr], 1); p1[te] = q[te] + np.polyval(b, fi[te])
        else: p1[te] = p0[te]
    return p0, p1

rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())) if m.sum() > 5 else np.nan

def analyse(Fd, val, seed_list, sel, label, nperm=500, show=True):
    Ov = O[O.validator == val]
    out = {}
    for sd in seed_list:
        Os = Ov if sd == "avg" else Ov[Ov.seed == sd]
        q = Os.groupby("rec").sg.mean()
        recs = [r for r in q.index if r in yday.index and r in Fd.index]
        q = q[recs]; y = yday[recs]
        keep = (q >= 0.9) if sel == "pred" else (y >= 1.0)
        recs = list(q.index[keep.values]); q = q[recs].values; y = y[recs].values
        ep = episodes(recs); groups = np.array([ep[r] for r in recs])
        res = []
        for c in FEATS:
            f = Fd.loc[recs, c].values.astype(float)
            p0, p1 = loeo(y, q, f, groups)
            res.append(dict(feat=c, n_ok=int(np.isfinite(f).sum()), rho_y=spear(f, y), rho_r=spear(f, y - q),
                            rmse_q=rmse(q, y), rmse_icpt=rmse(p0, y), rmse_f=rmse(p1, y),
                            gain_vs_icpt=rmse(p0, y) - rmse(p1, y), better_share=float(np.mean(np.abs(p1 - y) < np.abs(p0 - y)))))
        T = pd.DataFrame(res).set_index("feat")
        out[sd] = dict(T=T, recs=recs, y=y, q=q, groups=groups)
    # permutation null on seed 'avg' (or first)
    k0 = seed_list[0]; d = out[k0]; y, q, groups, recs = d["y"], d["q"], d["groups"], d["recs"]
    Fm = Fd.loc[recs, FEATS].values.astype(float)
    ug = np.unique(groups); gidx = [np.where(groups == g)[0] for g in ug]
    def maxgain(Fp):
        best = -9
        for j in range(Fp.shape[1]):
            p0, p1 = loeo(y, q, Fp[:, j], groups); best = max(best, rmse(p0, y) - rmse(p1, y))
        return best
    def maxrho(Fp): return max(abs(spear(Fp[:, j], y - q)) for j in range(Fp.shape[1]))
    nullA, nullB, nullRA, nullRB = [], [], [], []
    for it in range(nperm):
        Fp = Fm[rng.permutation(len(y))]                      # 기록 순열
        nullA.append(maxgain(Fp)); nullRA.append(maxrho(Fp))
        Fq = np.empty_like(Fm)                                # 에피소드 블록 순열(다른 에피소드 값을 순환 배치)
        perm = rng.permutation(len(ug))
        for gi, src in zip(gidx, perm):
            s = gidx[src]; Fq[gi] = Fm[s[np.arange(len(gi)) % len(s)]]
        nullB.append(maxgain(Fq)); nullRB.append(maxrho(Fq))
    thr = dict(gainA=float(np.quantile(nullA, .95)), gainB=float(np.quantile(nullB, .95)),
               rhoA=float(np.quantile(nullRA, .95)), rhoB=float(np.quantile(nullRB, .95)))
    if show:
        T = out[k0]["T"]
        print(f"\n=== {label}: validator {val}, seed {k0}, sample={sel} n={len(y)} episodes={len(ug)} | mean y {y.mean():.3f}, mean q {q.mean():.3f}, mean resid {np.mean(y-q):+.3f}")
        print(f"    RMSE q {rmse(q,y):.4f} | q+icpt(LOEO) {T.rmse_icpt.iloc[0]:.4f}")
        print(f"    null 95%: max gain vs icpt  record-perm {thr['gainA']:.4f}  block-perm {thr['gainB']:.4f} | max|rho_r| record {thr['rhoA']:.2f} block {thr['rhoB']:.2f}")
        print(T.sort_values("gain_vs_icpt", ascending=False).round(3).to_string())
    return out, thr

if __name__ == "__main__":
    F23 = table(A_tr, 23)
    F23.to_csv(os.path.join(here, "pp1_train_feats_full_v1.csv"))
    print("feature coverage (train, full day):"); print(F23.notna().mean().round(2).to_string())
    res = {}
    for sel in ["pred", "true"]:
        res[sel] = analyse(F23, "DIAG10", ["avg", 47, 1414, 6464], sel, f"FULL-DAY (non-causal upper bound)", nperm=300)
    # per-seed summary of top features
    for sel in ["pred", "true"]:
        out, thr = res[sel]
        G = pd.DataFrame({sd: out[sd]["T"].gain_vs_icpt for sd in out})
        Rr = pd.DataFrame({sd: out[sd]["T"].rho_r for sd in out})
        print(f"\n--- {sel}: LOEO gain vs q+icpt by seed (positive = feature helps)  [null95 record {thr['gainA']:.4f}, block {thr['gainB']:.4f}]")
        print(G.round(4).sort_values("avg", ascending=False).to_string())
        print(f"--- {sel}: rho(feature, residual) by seed [null95 |rho| record {thr['rhoA']:.2f}, block {thr['rhoB']:.2f}]")
        print(Rr.round(2).reindex(G.sort_values('avg', ascending=False).index).to_string())
