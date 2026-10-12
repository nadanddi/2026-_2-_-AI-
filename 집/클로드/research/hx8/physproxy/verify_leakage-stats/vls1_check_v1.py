# -*- coding: utf-8 -*-
"""VLS1 (2026-10-13 집 클로드, 회의론 검증; physproxy PP4/PP5 숫자 재확인 + 누수·표본·기준선 점검). 서술 전용."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, P)
from pp4_pseudoeval_fix_v1 import *   # noqa
import warnings; warnings.filterwarnings("ignore")
rng2 = np.random.default_rng(777)

def L_generic(tgt, mode, corr_fn, sd=47):
    """corr_fn(fit_recs, res_fit, t) -> correction for t. returns DataFrame with y, M, P."""
    q = qday_of("DIAG10", sd); rows = []
    for t in tgt:
        ex = excl(t, mode); fit = [r for r in tgt if r not in ex]
        if len(fit) < 5: continue
        res = yday[fit].values - q[fit].values
        rows.append(dict(t=t, seg=Cal.loc[t, "seg"], farm=t[:3], y=yday[t], M=q[t], P=q[t] + corr_fn(fit, res, t), nfit=len(fit)))
    return pd.DataFrame(rows)

def ahx_corr(f):
    def g(fit, res, t):
        b, a = fit_lin(f[fit].values, res); return a + b * f[t]
    return g
def farm_corr(fit, res, t):
    m = np.array([r[:3] == t[:3] for r in fit])
    return res[m].mean() if m.sum() >= 3 else res.mean()
def gain(D): return rmse(D.M.values, D.y.values) - rmse(D.P.values, D.y.values)

print("== 0) 재현: PP4 L seed47 ==")
for mode in ["BUF", "LOSO+"]:
    D = L_generic(TGT, mode, ahx_corr(AHXc))
    print(f"  {mode}: n {len(D)} RMSE M {rmse(D.M.values,D.y.values):.4f} M_AHxc {rmse(D.P.values,D.y.values):.4f} gain {gain(D):+.4f}")

print("\n== 1) seg 별 기여 (제곱오차 감소 합) / seg 하나 빼고 이득 ==")
for mode in ["BUF", "LOSO+"]:
    D = L_generic(TGT, mode, ahx_corr(AHXc)); D["gsq"] = (D.M - D.y) ** 2 - (D.P - D.y) ** 2
    tot = D.gsq.sum()
    print(f"  {mode}: total SSE drop {tot:.3f} | by seg:", D.groupby("seg").gsq.agg(['size','sum']).round(3).to_dict('index'))
    print("     drop-one-seg gain:", {int(s): round(gain(D[D.seg != s]), 4) for s in sorted(D.seg.unique())})
    print("     by farm:", {f: (len(g), round(rmse(g.M.values,g.y.values),3), round(rmse(g.P.values,g.y.values),3)) for f, g in D.groupby("farm")})
    top = D.sort_values("gsq", ascending=False)
    print(f"     top1 share {top.gsq.iloc[0]/tot:.2f}, top2 {top.gsq.iloc[:2].sum()/tot:.2f}, top3 {top.gsq.iloc[:3].sum()/tot:.2f}; gain w/o top1 {gain(D.drop(top.index[:1])):+.4f}, w/o top2 {gain(D.drop(top.index[:2])):+.4f}")

print("\n== 2) AHx 가 동 ID(F13/F47) 대리인가: 같은 ID 평균 잔차 보정 vs AHx ==")
print("  TGT farm counts:", pd.Series([t[:3] for t in TGT]).value_counts().to_dict(),
      "| AHxc mean by farm:", {f: round(AHXc[[t for t in TGT if t[:3]==f]].mean(),2) for f in ["F13","F47"]},
      "| resid mean by farm (avg seed):", {f: round((yday[[t for t in TGT if t[:3]==f]] - qavg[[t for t in TGT if t[:3]==f]]).mean(),3) for f in ["F13","F47"]})
for mode in ["BUF", "LOSO+"]:
    for sd in SEEDS:
        Df = L_generic(TGT, mode, farm_corr, sd); Da = L_generic(TGT, mode, ahx_corr(AHXc), sd)
        # within-farm AHx: demean AHx by farm
        fm = {f: AHXc[[t for t in TGT if t[:3]==f]].mean() for f in ["F13","F47"]}
        Aw = pd.Series({r: AHXc[r] - fm[r[:3]] for r in TGT})
        def both(fit, res, t):
            m = np.array([r[:3] == t[:3] for r in fit]); base = res[m].mean() if m.sum()>=3 else res.mean()
            rr = res - np.array([res[np.array([x[:3]==r[:3] for x in fit])].mean() for r in fit])
            b, a = fit_lin(Aw[fit].values, rr); return base + a + b * Aw[t]
        Db = L_generic(TGT, mode, both, sd)
        print(f"  {mode} s{sd}: gain farm-icpt {gain(Df):+.4f} | AHxc {gain(Da):+.4f} | farm-icpt + within-farm AHx {gain(Db):+.4f}")

print("\n== 3) seg 사이 vs seg 안: AHx 를 seg 평균 / seg 안 편차로 쪼갬 ==")
sm = pd.Series({r: AHXc[[x for x in TGT if Cal.loc[x,'seg']==Cal.loc[r,'seg']]].mean() for r in TGT})
for mode in ["BUF", "LOSO+"]:
    print(f"  {mode}: seg-mean-only gain {gain(L_generic(TGT, mode, ahx_corr(sm))):+.4f} | within-seg-dev gain {gain(L_generic(TGT, mode, ahx_corr(AHXc - sm))):+.4f}")

print("\n== 4) 갈림길(문턱 × 방식 × 시드): AHxc 이득 (RMSE M - RMSE M_AHxc), 표적·적합 풀 모두 문턱으로 재정의 ==")
for thr in [0.8, 0.85, 0.9, 0.95, 1.0, 1.1]:
    tg = [r for r in qavg.index[qavg >= thr] if r in E.index]
    s = f"  thr {thr:.2f} n {len(tg):2d}:"
    for mode in ["BUF", "LOSO+"]:
        g = [gain(L_generic(tg, mode, ahx_corr(AHXc), sd)) for sd in SEEDS]
        s += f" {mode} " + "/".join(f"{x:+.3f}" for x in g)
    print(s)

print("\n== 5) 평가 8기록과 닮았나: 낮은 AHx 표적 / 하향 보정 표적 / 밀폐 표적 ==")
mv = pd.Series({r: np.nanmax(A_tr[r]["act_vent"]) for r in TGT})
fan = pd.Series({r: np.nanmean(A_tr[r]["act_circfan"]) for r in TGT})
for mode in ["BUF", "LOSO+"]:
    D = L_generic(TGT, mode, ahx_corr(AHXc)).set_index("t"); D["corr"] = D.P - D.M; D["ahx"] = AHXc[D.index]
    for lab, m in [("AHxc<=4.12(평가8 최대 이하)", D.ahx <= 4.12), ("AHxc<=3.3", D.ahx <= 3.3), ("보정<0(하향)", D["corr"] < 0),
                   ("max vent<=3(평가8처럼 밀폐)", mv[D.index] <= 3), ("closed & fan<5", (mv[D.index] <= 3) & (fan[D.index] < 5))]:
        S = D[m]
        if len(S) == 0: print(f"  {mode} {lab}: n 0"); continue
        print(f"  {mode} {lab}: n {len(S)} RMSE M {rmse(S.M.values,S.y.values):.3f} M_AHxc {rmse(S.P.values,S.y.values):.3f} better {np.mean(np.abs(S.P-S.y)<np.abs(S.M-S.y)):.2f} | mean y {S.y.mean():.2f} mean corr {S['corr'].mean():+.3f}")
print("  eval8 max act_vent all <=3 (PP4 log); TGT closed share", round(float((mv <= 3).mean()), 2))

print("\n== 6) seg 평균 순열 기준선 (LOSO+, seed47): seg 안 편차는 두고 seg 평균 AHx 만 seg 사이에서 순열 ==")
segs_ = sorted(Cal.loc[TGT, "seg"].unique()); segm = {s: AHXc[[x for x in TGT if Cal.loc[x,'seg']==s]].mean() for s in segs_}
real = gain(L_generic(TGT, "LOSO+", ahx_corr(AHXc)))
import itertools
vals = []
perms = list(itertools.permutations(range(len(segs_))))
idx = rng2.choice(len(perms), 1500, replace=False)
for k in idx:
    pm = perms[k]; mp = {segs_[i]: segm[segs_[pm[i]]] for i in range(len(segs_))}
    f = pd.Series({r: AHXc[r] - segm[Cal.loc[r,'seg']] + mp[Cal.loc[r,'seg']] for r in TGT})
    vals.append(gain(L_generic(TGT, "LOSO+", ahx_corr(f))))
vals = np.array(vals)
print(f"  real {real:+.4f} | seg-mean perm null 95% {np.quantile(vals,.95):+.4f}, p {np.mean(vals >= real):.3f} (n perms {len(vals)}, segs {len(segs_)})")
# big-4-seg only version
big = [s for s in segs_ if (Cal.loc[TGT,'seg']==s).sum() >= 5]
print("  big segs:", big, {s: round(segm[s],2) for s in segs_}, "| seg mean resid (avg):", {int(s): round(float((yday[[x for x in TGT if Cal.loc[x,'seg']==s]] - qavg[[x for x in TGT if Cal.loc[x,'seg']==s]]).mean()),3) for s in segs_})

print("\n== 7) DIAG10 fold: 표적과 같은 동 이웃(사슬 1걸음)이 다른 폴드(=모델 학습에 들어감)인 비율 ==")
Od = O[(O.validator == "DIAG10") & (O.seed == 47)].groupby("rec").fold.first()
nbr = [(t, x) for t in TGT for x in nb.get(t, ()) if x in Od.index and t in Od.index]
print(f"  pairs {len(nbr)}, different fold share {np.mean([Od[t] != Od[x] for t, x in nbr]):.2f}; twins(same seg/day other ID) diff fold share",
      round(float(np.mean([Od[t] != Od[x] for t in TGT for x in Cal.index[(Cal.seg==Cal.loc[t,'seg'])&(Cal.day==Cal.loc[t,'day'])&(Cal.index!=t)] if x in Od.index])), 2))
