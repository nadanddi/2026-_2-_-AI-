# -*- coding: utf-8 -*-
"""HX9 / EP2 — EP1 이어서 (2026-10-13 집 클로드).
EP1 결과 요지: 진단판 '전날 23시 EC - 오늘 예측'(d_gap23)이 고EC 잔차를 r .89 로 설명, 그러나 입력 링커 v4(같은 온실 ID 후보만) 앵커는
  고EC 에서 진짜 전날 적중 .39, 앵커 23시 EC 와 진짜 전날 23시 EC r .19 → 사용 가능판 정보 거의 0. 고EC 에피소드 3~7 은 진짜 전날이 매일 다른 ID(F13↔F47).
이 스크립트(새로 하는 것):
 A. 배타성 앵커(새 사용 가능 변수): 대상 b 의 날짜(seg, day; DMT3e 학습 전용) 전날 묶음의 학습 고EC 기록 중, 오늘 쌍둥이(같은 날짜 학습 기록, 가림 블록 제외)의
    고EC 와 짝 지어지지 않은 기록 = '이어지는 고EC 동' 후보. 그 기록의 23시 EC(u_excl_ec23), 그것 - 오늘 예측(u_excl_gap23).
    다른 온실 ID 학습 기록 사용 = 10-11 사용자 규정 결정(허용). 평가 입력은 쓰지 않음(대상 자신의 0시 실외만으로 날짜).
    (카탈로그 6.412 GX0~GX2 격자 배제법의 '고EC 크기' 판 — GX 는 일반 앵커 MSE 기준이었고, 여기서는 고EC 잔차와의 연관·정밀도만 본다.)
 B. EP1 의 사용 가능 '신호' 판정 변수(u_lk_steps, u_lk_p1, u_nd_n, u_unclaimed_n, u_pass2)가 기하(블록 가림·ID 교차) 인공물인지 점검.
 C. 갈래 전체 다변량 상한(비인과): 모든 변수 -> Y2, 릿지·작은 GBM, 에피소드 단위 LOCO; 사용 가능 변수만 따로.
 D. 평가 60 기록 포괄: 날짜 결정(0시 쌍둥이), 전날 묶음, 배타성 앵커 후보 수, 링커 v4 확신 앵커 수.
계산 전 예측 (A): 배타성 식별이 고EC 에서 통한다면 (i) unclaimed 가 있는 기록 중 실제 고EC 비율(정밀도) >= .7,
   (ii) 고EC 에서 u_excl_ec23 이 진짜 전날 23시 EC 와 r >= .8 (같은 기록을 고르는 비율 >= .7), (iii) u_excl_gap23 과 Y2 r >= .5 & LOCO 이득 > 0.
   보이면 안 되는 것: 보통 기록에서 unclaimed 가 자주 생김(정밀도 낮음) — 그러면 평가에서 가짜 고EC 를 만든다.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE); RES = os.path.dirname(HX9)
rng = np.random.default_rng(1013)
NB = 2000
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
V = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec")
LK = pd.read_csv(os.path.join(RES, "local", "linker_v4", "cv_all400_choice_v4.csv"), encoding="utf-8-sig").set_index("rec")
EV = pd.read_csv(os.path.join(RES, "local", "linker_v4", "eval60_predecessor_v4.csv"), encoding="utf-8-sig").set_index("row")
sys.path.insert(0, os.path.join(RES, "linker_v4"))
from feat4 import FARM, NUM  # noqa: E402
LO = {f: int(NUM[FARM == f].min()) for f in ("F13", "F47")}
def block_start(r):
    if r in LK.index: return int(LK.at[r, "block"].split("_")[1].split("-")[0])
    f, n = r[:3], int(r[4:]); return LO[f] + 10 * ((n - LO[f]) // 10)
HID = {}
for r in R.index:
    s = block_start(r); f = r[:3]
    HID[r] = {x for x in R.index if x[:3] == f and s - 1 <= int(x[4:]) <= s + 10}
E = R.ec_mean.to_dict(); E23 = R.ec_h23.to_dict(); HI = (R.ec_mean >= 1.0).to_dict()
hi = R.ec_mean >= 1.0; ok = R.has_sg == 1; Y2 = R.resid_mean

# ---------------- A. exclusivity anchor ----------------
bydate = {}
for r, (s, d) in R[["dmt_seg", "dmt_day"]].iterrows(): bydate.setdefault((int(s), int(d)), []).append(r)
def excl_for(today_twins, prev_recs):
    """greedy: today's high twins claim the closest prev-date high record; return unclaimed prev-date high records (sorted by EC desc)."""
    ph = sorted([x for x in prev_recs if HI[x]], key=lambda x: -E[x]); rem = list(ph)
    for t in [x for x in today_twins if HI[x]]:
        if rem:
            j = int(np.argmin([abs(E[t] - E[p]) for p in rem])); rem.pop(j)
    return rem
rows = []
for r in R.index:
    s, d = int(R.at[r, "dmt_seg"]), int(R.at[r, "dmt_day"])
    tw = R.at[r, "twin_recs"]; tw = tw.split(";") if isinstance(tw, str) and tw else []
    tw = [x for x in tw if x not in HID[r] and x != r]
    pdv = [x for x in bydate.get((s, d - 1), []) if x not in HID[r] and x != r]
    rem = excl_for(tw, pdv)
    o = dict(rec=r, u_excl_n=len(rem) if pdv else np.nan)
    if rem:
        x = rem[0]; o.update(u_excl_rec=x, u_excl_ec23=E23[x], u_excl_mean=E[x], u_excl_cross_id=int(x[:3] != r[:3]))
    else:
        o.update(u_excl_rec=None, u_excl_ec23=np.nan, u_excl_mean=np.nan, u_excl_cross_id=np.nan)
    rows.append(o)
X_ = pd.DataFrame(rows).set_index("rec")
X_["u_excl_gap23"] = X_.u_excl_ec23 - R.sg_mean
X_["u_excl_gap"] = X_.u_excl_mean - R.sg_mean
X_["u_excl_flag"] = (X_.u_excl_n >= 1).astype(float); X_.loc[X_.u_excl_n.isna(), "u_excl_flag"] = np.nan
V2 = V.join(X_.drop(columns=["u_excl_rec"]))
V2.to_csv(os.path.join(HERE, "vars_v2.csv"), encoding="utf-8")
X_[["u_excl_rec"]].to_csv(os.path.join(HERE, "vars_v2_excl_rec.csv"), encoding="utf-8")

print("=== A. 배타성 앵커 ===")
fl = X_.u_excl_flag == 1
print(f"  날짜 전날 묶음 있음 {X_.u_excl_n.notna().sum()}/400 | unclaimed 있음 {fl.sum()} -> 그중 실제 고EC {int((fl & hi).sum())} (정밀도 {(fl & hi).sum()/max(fl.sum(),1):.2f}); "
      f"고EC 35 중 unclaimed 있음 {int((fl & hi).sum())} (재현율 {(fl & hi).sum()/hi.sum():.2f})")
print(f"  unclaimed 인데 보통인 기록: {sorted(R.index[fl & ~hi].tolist())}")
same = [(X_.at[r, "u_excl_rec"] == R.at[r, "prev_rec"]) for r in R.index[fl & hi & R.prev_rec.notna()]]
print(f"  고EC & unclaimed & 진짜 전날 있음 n {len(same)}: 같은 기록 고름 {np.mean(same):.2f}; 다른 ID 교차 비율 {X_.u_excl_cross_id[fl & hi].mean():.2f}")
m = fl & hi & V.d_prev_h23.notna()
print(f"  u_excl_ec23 vs 진짜 전날 23시 r {stats.pearsonr(X_.u_excl_ec23[m], V.d_prev_h23[m])[0]:.3f} (n {m.sum()}), |차| 중앙 {np.median(np.abs(X_.u_excl_ec23[m]-V.d_prev_h23[m])):.3f}")

def boot_r(x, y, cl):
    m = x.notna() & y.notna(); x, y, cl = x[m].values, y[m].values, cl[m].values
    u = np.unique(cl); gi = {c: np.where(cl == c)[0] for c in u}; out = []
    for _ in range(NB):
        s = rng.choice(u, len(u), replace=True); ii = np.concatenate([gi[c] for c in s])
        if np.std(x[ii]) < 1e-12 or np.std(y[ii]) < 1e-12: continue
        out.append(stats.pearsonr(x[ii], y[ii])[0])
    return np.nanpercentile(out, [2.5, 97.5]) if len(out) > 50 else (np.nan, np.nan)
def ols(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr]); B = np.column_stack([np.ones(len(Xte)), Xte])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None); return B @ w
def loco_gain(x, y, sg, cl):
    m = x.notna() & y.notna() & sg.notna(); x, y, sg, cl = x[m].values, y[m].values, sg[m].values, cl[m].values
    pb = np.full(len(y), np.nan); pa = pb.copy()
    for c in np.unique(cl):
        te = cl == c
        if (~te).sum() < 5: continue
        pb[te] = ols(sg[~te, None], y[~te], sg[te, None]); pa[te] = ols(np.column_stack([sg[~te], x[~te]]), y[~te], np.column_stack([sg[te], x[te]]))
    f = np.isfinite(pb) & np.isfinite(pa); sst = np.sum((y[f] - y[f].mean()) ** 2)
    return int(f.sum()), 1 - np.sum((y[f] - pa[f]) ** 2) / sst - (1 - np.sum((y[f] - pb[f]) ** 2) / sst), int(np.sum(np.abs(y[f] - pa[f]) < np.abs(y[f] - pb[f])))
par = {r: r for r in R.index}
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for b, a in R.prev_rec.dropna().items(): par[fd(b)] = fd(a)
CL_all = pd.Series({r: fd(r) for r in R.index}); CL_hi = R.ep_id
newv = ["u_excl_n", "u_excl_flag", "u_excl_ec23", "u_excl_mean", "u_excl_gap23", "u_excl_gap", "u_excl_cross_id"]
newres = []
for v in newv:
    x = V2[v]; d = dict(variable=v)
    for nm, mask, cl in (("Y2h", hi & ok, CL_hi), ("Y2a", ok, CL_all)):
        mm = mask & x.notna()
        if mm.sum() >= 8 and x[mm].std() > 0:
            r_, p_ = stats.pearsonr(x[mm], Y2[mm]); lo_, hi_ = boot_r(x[mask], Y2[mask], cl[mask]); n_, g_, imp = loco_gain(x[mask], Y2[mask], R.sg_mean[mask], cl[mask])
            r1 = stats.pearsonr(x[mm & (R.pass2 == 0)], Y2[mm & (R.pass2 == 0)])[0] if (mm & (R.pass2 == 0)).sum() > 4 else np.nan
            r2 = stats.pearsonr(x[mm & (R.pass2 == 1)], Y2[mm & (R.pass2 == 1)])[0] if (mm & (R.pass2 == 1)).sum() > 4 and x[mm & (R.pass2 == 1)].std() > 0 else np.nan
            d.update({f"{nm}_n": int(mm.sum()), f"{nm}_r": r_, f"{nm}_p": p_, f"{nm}_ci_lo": lo_, f"{nm}_ci_hi": hi_, f"{nm}_cv_gain": g_, f"{nm}_improved": imp, f"{nm}_r_p1": r1, f"{nm}_r_p2": r2})
    mm = hi & x.notna()
    if mm.sum() >= 8 and x[mm].std() > 0: d["Y1_r"] = stats.pearsonr(x[mm], R.ec_mean[mm])[0]; d["Y1_n"] = int(mm.sum())
    newres.append(d)
NR = pd.DataFrame(newres); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(NR.round(3).to_string(index=False))
NR.to_csv(os.path.join(HERE, "ep2_excl_results_v1.csv"), index=False, encoding="utf-8-sig")
# 배타성 앵커만으로 고EC 수준 (값이 있는 고EC)
m = hi & ok & X_.u_excl_ec23.notna()
print(f"  고EC & 배타성 앵커 있음 n {m.sum()}: 모델 |잔차| 중앙 {Y2[m].abs().median():.3f} vs |y - 앵커23| 중앙 {(R.ec_mean[m]-X_.u_excl_ec23[m]).abs().median():.3f}; "
      f"앵커가 더 가까움 {np.mean((R.ec_mean[m]-X_.u_excl_ec23[m]).abs() < Y2[m].abs()):.2f}")
m2 = (~hi) & ok & X_.u_excl_ec23.notna()
print(f"  보통 & 배타성 앵커 있음 n {m2.sum()}: 모델 |잔차| 중앙 {Y2[m2].abs().median() if m2.sum() else np.nan:.3f} vs |y - 앵커23| 중앙 {(R.ec_mean[m2]-X_.u_excl_ec23[m2]).abs().median() if m2.sum() else np.nan:.3f}")
# 전체 기록에서 '배타성 앵커 있으면 앵커로 바꿈' 단순 규칙 (서술; 기준선 비교)
yy = R.ec_mean[ok]; base = R.sg_mean[ok]; alt = base.copy(); mm = X_.u_excl_ec23[ok].notna(); alt[mm] = X_.u_excl_ec23[ok][mm]
print(f"  [서술] 전체 360: 하루 수준 RMSE 모델 {np.sqrt(np.mean((yy-base)**2)):.4f} vs 배타성 앵커로 교체 {np.sqrt(np.mean((yy-alt)**2)):.4f} (바뀐 기록 {mm.sum()}); "
      f"고EC {np.sqrt(np.mean((yy-base)[hi[ok]]**2)):.4f} -> {np.sqrt(np.mean((yy-alt)[hi[ok]]**2)):.4f}; 보통 {np.sqrt(np.mean((yy-base)[~hi[ok]]**2)):.4f} -> {np.sqrt(np.mean((yy-alt)[~hi[ok]]**2)):.4f}")
for ps_ in (0, 1):
    q = ok & (R.pass2 == ps_); qq = q & X_.u_excl_ec23.notna()
    print(f"    pass{ps_+1}: 바뀐 {qq.sum()}  RMSE {np.sqrt(np.mean((R.ec_mean[q]-R.sg_mean[q])**2)):.4f} -> {np.sqrt(np.mean((R.ec_mean[q]-np.where(qq[q], X_.u_excl_ec23[q], R.sg_mean[q]))**2)):.4f}")

# ---------------- B. geometry artifact checks for EP1 usable 'signals' ----------------
print("\n=== B. EP1 사용 가능 '신호' 의 기하 인공물 점검 (고EC, OOF 있음) ===")
H = R.index[hi & ok]
cross = pd.Series({r: float(R.at[r, "prev_rec"][:3] != r[:3]) if isinstance(R.at[r, "prev_rec"], str) else np.nan for r in R.index})
for v in ["u_lk_steps", "u_lk_p1", "u_nd_n", "u_unclaimed_n", "u_pass2"]:
    x = V[v][H]; y = Y2[H]
    m = x.notna()
    rs = stats.spearmanr(x[m], y[m])[0]
    # 최대 영향 기록 제외
    infl = []
    for k in x[m].index:
        mk = m.copy(); mk[k] = False; infl.append(stats.pearsonr(x[mk], y[mk])[0])
    # 부분상관: 진짜 전날이 다른 ID 인지(진단) 통제
    c = cross[H]; mm = m & c.notna()
    def resid_on(a, b_):
        A = np.column_stack([np.ones(len(b_)), b_]); w, *_ = np.linalg.lstsq(A, a, rcond=None); return a - A @ w
    pc = stats.pearsonr(resid_on(x[mm].values, c[mm].values), resid_on(y[mm].values, c[mm].values))[0] if mm.sum() > 6 else np.nan
    print(f"  {v:14s} n {m.sum()} r {stats.pearsonr(x[m], y[m])[0]:+.3f} rho {rs:+.3f} | 한 기록 빼기 r 범위 [{min(infl):+.3f},{max(infl):+.3f}] | "
          f"r(var, 진짜 전날 다른 ID) {stats.pearsonr(x[mm], c[mm])[0]:+.3f} | 다른 ID 통제 부분 r {pc:+.3f} (n {mm.sum()})")
print(f"  참고: 고EC 잔차 평균 - 진짜 전날 다른 ID {Y2[H][cross[H] == 1].mean():+.3f} (n {(cross[H]==1).sum()}) / 같은 ID {Y2[H][cross[H] == 0].mean():+.3f} (n {(cross[H]==0).sum()}) / 전날 없음 {Y2[H][cross[H].isna()].mean():+.3f} (n {cross[H].isna().sum()})")

# ---------------- C. multivariate non-causal upper bound ----------------
print("\n=== C. 갈래 다변량 상한 (비인과, LOCO) ===")
ALLV = [c for c in V2.columns if c.startswith("d_") or c.startswith("u_")]
USE = [c for c in ALLV if c.startswith("u_")]
def mv(cols, mask, cl, label):
    D = V2.loc[mask, cols].copy(); y = Y2[mask].values; sg = R.sg_mean[mask].values; c_ = cl[mask].values
    # missing indicators for columns with NaN
    nanc = [c for c in cols if D[c].isna().any()]
    for c in nanc: D[c + "_na"] = D[c].isna().astype(float)
    D["sg_mean"] = sg
    out = {}
    for mdl in ("base", "ridge", "gbm"):
        p = np.full(len(y), np.nan)
        for g in np.unique(c_):
            te = c_ == g; tr = ~te
            Dtr = D[tr].copy(); Dte = D[te].copy()
            med = Dtr.median(); Dtr = Dtr.fillna(med).fillna(0); Dte = Dte.fillna(med).fillna(0)
            if mdl == "base":
                p[te] = ols(Dtr[["sg_mean"]].values, y[tr], Dte[["sg_mean"]].values); continue
            mu, sd = Dtr.mean(), Dtr.std().replace(0, 1)
            Ztr, Zte = ((Dtr - mu) / sd).values, ((Dte - mu) / sd).values
            if mdl == "ridge":
                m_ = RidgeCV(alphas=np.logspace(-1, 3, 20)).fit(Ztr, y[tr]); p[te] = m_.predict(Zte)
            else:
                m_ = GradientBoostingRegressor(max_depth=2, n_estimators=150, learning_rate=0.05, subsample=0.8, min_samples_leaf=3, random_state=0).fit(Ztr, y[tr]); p[te] = m_.predict(Zte)
        out[mdl] = p
    sst = np.sum((y - y.mean()) ** 2)
    hm = hi[mask].values
    for mdl in ("base", "ridge", "gbm"):
        r2 = 1 - np.sum((y - out[mdl]) ** 2) / sst
        imp_b = int(np.sum((np.abs(y - out[mdl]) < np.abs(y - out["base"]))[hm])) if mdl != "base" else np.nan
        imp_0 = int(np.sum((np.abs(y - out[mdl]) < np.abs(y))[hm]))
        rm_h = np.sqrt(np.mean((y - out[mdl])[hm] ** 2)); rm_0 = np.sqrt(np.mean(y[hm] ** 2))
        print(f"  {label:34s} {mdl:5s} n {len(y)} feats {len(cols)}  R2(LOCO) {r2:+.3f} | 고EC {hm.sum()} 중 개선(vs base) {imp_b} / (vs 원 모델) {imp_0} | 고EC 잔차 RMSE {rm_0:.3f} -> {rm_h:.3f}")
    return out
mv(ALLV, hi & ok, CL_hi, "고EC: 전체 변수(진단+사용)")
mv(USE, hi & ok, CL_hi, "고EC: 사용 가능 변수만")
mv([c for c in USE if not c.startswith("u_lk_steps") and c not in ("u_lk_p1", "u_nd_n", "u_pass2", "u_lk_conf")], hi & ok, CL_hi, "고EC: 사용 가능 - 기하 변수 제외")
mv(["d_gap23"], hi & ok, CL_hi, "고EC: d_gap23 하나(오라클 지속)")
mv(["u_excl_gap23", "u_excl_flag"], hi & ok, CL_hi, "고EC: 배타성 앵커 2개")
mv(ALLV, ok, CL_all, "전체: 전체 변수(진단+사용)")
mv(USE, ok, CL_all, "전체: 사용 가능 변수만")
mv(["u_excl_gap23", "u_excl_flag", "u_excl_n"], ok, CL_all, "전체: 배타성 앵커 3개")

# ---------------- E. 배타성 플래그 정밀도: 대상 자신의 입력 '밀폐 지문'으로 거르면? (서술, n 30) ----------------
print("\n=== E. 배타성 플래그 30기록: 고EC 15 vs 보통 15 를 대상 자신의 입력으로 가를 수 있나 (서술) ===")
HR = pd.read_csv(os.path.join(HX9, "data", "hourly.csv"))
def fp(h_upto):
    g = HR[HR.hour <= h_upto].groupby("rec")
    return pd.DataFrame({"fan_mean": g.act_circfan.mean(), "vent0_frac": g.act_vent.apply(lambda s: (s == 0).mean()),
                         "co2in_mean": g.in_co2.mean(), "heat_mean": g.act_heating.mean(), "intemp_mean": g.in_temp.mean()})
for hu in (5, 11, 23):
    F = fp(hu); m = X_.u_excl_flag == 1
    y_ = hi[m].astype(int)
    for c in F.columns:
        x = F.loc[m[m].index, c]
        auc = stats.mannwhitneyu(x[y_[m[m].index] == 1], x[y_[m[m].index] == 0]).statistic / (y_.sum() * (len(y_) - y_.sum()))
        print(f"  h<= {hu:2d} {c:11s} AUC(고EC 쪽 큼) {auc:.2f}", end=" |")
    print()
F = fp(23)
seal = (F.fan_mean < 10) & (F.vent0_frac > 0.85)
m = X_.u_excl_flag == 1
print(f"  하루 밀폐(팬<10 & 환기0 >.85, 6.11 정의; h=23 판정) 걸러내기: 남는 {int((m & seal).sum())} 중 고EC {int((m & seal & hi).sum())}; 빠지는 {int((m & ~seal).sum())} 중 고EC {int((m & ~seal & hi).sum())}")

# ---------------- D. eval-60 coverage ----------------
print("\n=== D. 평가 60 포괄 (사용 가능 변수가 실제로 계산되는가) ===")
TX = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TX = TX[TX.row_id.str[:3].isin(["F13", "F47"])].copy()
TX["rec"] = TX.row_id.str[:7]; TX["h"] = TX.row_id.str[8:].astype(int)
TR0 = pd.read_csv(os.path.join(HX9, "data", "hourly.csv")); TR0 = TR0[TR0.hour == 0].set_index("rec")
k = lambda s: tuple(np.round(s[["out_temp", "out_hum", "out_rad", "out_wspd"]].values.astype(float), 3))
trk = {r: k(TR0.loc[r]) for r in TR0.index}
rows = []
for e, g in TX[TX.h == 0].groupby("rec"):
    ke = k(g.iloc[0]); tw = [r for r, kk in trk.items() if kk == ke]
    sd = R.loc[tw, ["dmt_seg", "dmt_day"]].drop_duplicates() if tw else pd.DataFrame()
    o = dict(rec=e, n_twin=len(tw), n_date=len(sd))
    if len(sd) >= 1:
        s, d = sd.iloc[0]; s, d = int(s), int(d)
        pdv = bydate.get((s, d - 1), []); rem = excl_for(tw, pdv)
        o.update(seg=s, day=d, n_prevdate=len(pdv), n_prevdate_hi=sum(HI[x] for x in pdv), n_twin_hi=sum(HI[x] for x in tw),
                 excl_rec=rem[0] if rem else None, excl_ec23=E23[rem[0]] if rem else np.nan, excl_n=len(rem))
    lk_ = EV.loc[e] if e in EV.index else None
    if lk_ is not None:
        o.update(lk_steps=lk_.steps_to_anchor, lk_conf=lk_.anchor_conf, lk_ec23=lk_.anchor_ec23)
    rows.append(o)
C = pd.DataFrame(rows).set_index("rec")
C.to_csv(os.path.join(HERE, "ep2_eval_cov_v1.csv"), encoding="utf-8")
print(f"  0시 쌍둥이(학습) 있음 {int((C.n_twin > 0).sum())}/60 | 날짜 정해짐 {int(C.seg.notna().sum())} (날짜 후보 2개 이상 {int((C.n_date > 1).sum())}) | 전날 묶음 있음 {int((C.n_prevdate > 0).sum())}")
print(f"  전날 묶음에 학습 고EC 있음 {int((C.n_prevdate_hi > 0).sum())} | 배타성 앵커 후보 있음 {int((C.excl_n > 0).sum())}: {C[C.excl_n > 0][['excl_rec','excl_ec23','n_twin','n_twin_hi','n_prevdate','n_prevdate_hi']].round(3).to_dict('index')}")
print(f"  링커 v4 평가 앵커: 1단계&p>=.8 {int(((C.lk_steps == 1) & (C.lk_conf >= .8)).sum())}, 그중 앵커 23시 EC >= 1.0: {int(((C.lk_steps == 1) & (C.lk_conf >= .8) & (C.lk_ec23 >= 1)).sum())}; 앵커 23시 >= 1.0 전체 {int((C.lk_ec23 >= 1).sum())}")
print("saved", os.path.join(HERE, "vars_v2.csv"), os.path.join(HERE, "ep2_excl_results_v1.csv"), os.path.join(HERE, "ep2_eval_cov_v1.csv"))
