# -*- coding: utf-8 -*-
"""HX9 / EP3 — EP1·EP2 마무리 (2026-10-13 집 클로드, 사용량 한도 뒤 이어서).
EP1·EP2 에서 남은 빈칸만 채운다 (새로 하는 것):
 A. 갈래 전체 다중검정: 사용 가능 변수 39개(u_*)를 고EC 잔차(Y2, n<=31)에 하나씩 붙였을 때의 max|r|·max LOCO 이득이
    우연(잔차 순열)보다 큰가. 기록 단위 순열은 에피소드 안 상관을 무시하므로 반보수적(통과하기 쉬움) -> 여기서 실패하면 확실히 실패.
 B. 추세 예측 가능성(정리판): Y3 = 오늘 평균 - 전날 평균 은 '자정 이어짐'을 포함해 기계적. 이를 빼고
    Y3n = 오늘 평균 - 전날 23시 (자정 이후 새 변화), Yw = 오늘 23시 - 오늘 0시 (하루 안 변화) 를 앞 추세·수준·위치로 설명되는지 (LOCO R2).
 C. 잔차 분해: 고EC 잔차 = (0시 수준 오차 ec_h0 - sg_h0) + (하루 안 모양 오차). 지속(전날 23시 ≈ 오늘 0시) 정보가 닿는 몫은 앞쪽뿐.
 D. 평가 60 에서 지속 경로가 실제로 닿는 기록 수 (링커 v4 평가 앵커, 배타성 후보) 와 u_lk_steps 분포 이동(학습 CV vs 평가).
계산 전 예측:
 A: 39개 중 최고 |r| .46(u_lk_steps)은 n 31·39개 검정에서 우연 최대 |r| 의 95% 분위(약 .45~.50 예상) 근처 -> 가족 단위 p > .05 예상.
 B: 상태 전환형이면 Y3n·Yw 는 앞 추세와 무관(|r| < .2, LOCO R2 <= 0), 전날 수준과 음(되돌림) 정도만.
 C: 고EC 잔차 분산의 대부분(>= 60%)이 0시 수준 오차 쪽 -> 지속 정보가 있으면 크게 줄 수 있는 구조.
 D: 평가에서 확신 앵커(1단계 p>=.8) 중 앵커 23시 EC >= 1.0 은 0 (EP2 확인), 지속 경로가 닿는 고EC 후보는 많아야 7.
보이면 안 되는 것: A 에서 가족 단위 p < .01 이면서 B·C 와 모순 (예: 추세 변수만 강함) -> 기하 인공물 재점검.
관찰만. 인과 해석(염류 누적·급액 관리)은 일반 농업 지식이며 이 자료로 검증 안 됨.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE); RES = os.path.dirname(HX9)
rng = np.random.default_rng(31013)
NP = 2000; NB = 2000
R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
V = pd.read_csv(os.path.join(HERE, "vars_v2.csv")).set_index("rec")
LK = pd.read_csv(os.path.join(RES, "local", "linker_v4", "cv_all400_choice_v4.csv"), encoding="utf-8-sig").set_index("rec")
EV = pd.read_csv(os.path.join(RES, "local", "linker_v4", "eval60_predecessor_v4.csv"), encoding="utf-8-sig").set_index("row")
COV = pd.read_csv(os.path.join(HERE, "ep2_eval_cov_v1.csv")).set_index("rec")
hi = R.ec_mean >= 1.0; ok = R.has_sg == 1; Y2 = R.resid_mean; CL = R.ep_id
USE = [c for c in V.columns if c.startswith("u_")]
print(f"usable variables {len(USE)}; diag {sum(c.startswith('d_') for c in V.columns)}")

def ols(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr]); B = np.column_stack([np.ones(len(Xte)), Xte])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None); return B @ w
def loco_gain_arr(x, y, sg, cl):
    pb = np.full(len(y), np.nan); pa = pb.copy()
    for c in np.unique(cl):
        te = cl == c
        if (~te).sum() < 5: continue
        pb[te] = ols(sg[~te, None], y[~te], sg[te, None]); pa[te] = ols(np.column_stack([sg[~te], x[~te]]), y[~te], np.column_stack([sg[te], x[te]]))
    f = np.isfinite(pb) & np.isfinite(pa); sst = np.sum((y[f] - y[f].mean()) ** 2)
    return (np.sum((y[f] - pb[f]) ** 2) - np.sum((y[f] - pa[f]) ** 2)) / sst

# ---------------- A. family-wise permutation (usable, Y2 high) ----------------
print("\n=== A. 사용 가능 39개 가족 단위 순열 (고EC 잔차) ===")
H = R.index[hi & ok]
y0 = Y2[H].values; sg0 = R.sg_mean[H].values; cl0 = CL[H].values
cols = []
for v in USE:
    x = V.loc[H, v].values.astype(float); m = np.isfinite(x)
    if m.sum() >= 8 and np.nanstd(x[m]) > 1e-12: cols.append((v, x, m))
def stats_for(y):
    rs, gs = [], []
    for v, x, m in cols:
        rs.append(abs(np.corrcoef(x[m], y[m])[0, 1]))
        gs.append(loco_gain_arr(x[m], y[m], sg0[m], cl0[m]))
    return np.array(rs), np.array(gs)
r_obs, g_obs = stats_for(y0)
mx_r, mx_g = [], []
for _ in range(NP):
    yp = rng.permutation(y0); a, b = stats_for(yp); mx_r.append(np.nanmax(a)); mx_g.append(np.nanmax(b))
mx_r, mx_g = np.array(mx_r), np.array(mx_g)
order = np.argsort(-r_obs)
print(f"  검정 변수 {len(cols)} (n<8 또는 상수 제외) | 우연 max|r| 95% 분위 {np.percentile(mx_r,95):.3f}, max LOCO 이득 95% 분위 {np.percentile(mx_g,95):.3f}")
fw = []
for i in order[:10]:
    v = cols[i][0]; p_r = np.mean(mx_r >= r_obs[i]); p_g = np.mean(mx_g >= g_obs[i])
    fw.append(dict(variable=v, n=int(cols[i][2].sum()), abs_r=r_obs[i], fw_p_r=p_r, loco_gain=g_obs[i], fw_p_gain=p_g))
    print(f"  {v:16s} n {cols[i][2].sum():2d} |r| {r_obs[i]:.3f} 가족 p {p_r:.3f} | LOCO 이득 {g_obs[i]:+.3f} 가족 p {p_g:.3f}")
pd.DataFrame(fw).to_csv(os.path.join(HERE, "ep3_familywise_v1.csv"), index=False, encoding="utf-8-sig")
print(f"  최대 이득 변수 {cols[int(np.nanargmax(g_obs))][0]} {np.nanmax(g_obs):+.3f} 가족 p {np.mean(mx_g >= np.nanmax(g_obs)):.3f}")

# ---------------- B. trend predictability (clean) ----------------
print("\n=== B. 추세 예측 (자정 이어짐 제거판) ===")
PREV = R.prev_rec.dropna().to_dict()
def g(r, c):
    p = PREV.get(r); return R.at[p, c] if p is not None else np.nan
def g2(r, c):
    p = PREV.get(r); p = PREV.get(p) if p is not None else None; return R.at[p, c] if p is not None else np.nan
T = pd.DataFrame(index=R.index)
T["Y3"] = R.ec_mean - R.prev_ec_mean
T["Y3n"] = R.ec_mean - R.prev_ec_h23                     # 자정 이후 새 변화
T["Yw"] = R.ec_h23 - R.ec_h0                              # 오늘 하루 안 변화
T["mid_jump"] = R.ec_h0 - R.prev_ec_h23                   # 자정 이음새 (진단: 링크 품질)
T["prev_w"] = [g(r, "ec_h23") - g(r, "ec_h0") for r in R.index]      # 전날 하루 안 변화
T["prev_trend1"] = V.d_trend1; T["prev_level"] = V.d_prev_mean; T["ep_pos"] = V.d_ep_pos; T["run_hi"] = V.d_run_hi
T["prev_Y3n"] = [g(r, "ec_mean") - g2(r, "ec_h23") for r in R.index]  # 전날의 새 변화
T["season"] = R.dmt_day
mask_ep = hi & R.prev_rec.notna() & (R.prev_ec_mean >= 1.0)
mask_hi_all = hi
print(f"  고EC 에피소드 안 깨끗한 링크 n {int(mask_ep.sum())}; 자정 이음새 |ec_h0 - 전날 h23| 중앙 {T.mid_jump[mask_ep].abs().median():.3f} (전체 링크 {T.mid_jump.abs().median():.3f})")
def boot_r(x, y, cl):
    m = x.notna() & y.notna(); x, y, cl = x[m].values, y[m].values, cl[m].values
    u = np.unique(cl); gi = {c: np.where(cl == c)[0] for c in u}; out = []
    for _ in range(NB):
        s = rng.choice(u, len(u), replace=True); ii = np.concatenate([gi[c] for c in s])
        if np.std(x[ii]) < 1e-12 or np.std(y[ii]) < 1e-12: continue
        out.append(np.corrcoef(x[ii], y[ii])[0, 1])
    return np.nanpercentile(out, [2.5, 97.5]) if len(out) > 50 else (np.nan, np.nan)
def loco_r2(Xc, y, cl, mask):
    m = mask & y.notna() & T[Xc].notna().all(1)
    ym, Xm, cm = y[m].values, T.loc[m, Xc].values, cl[m].values
    p = np.full(len(ym), np.nan)
    for c in np.unique(cm):
        te = cm == c
        if (~te).sum() < 4: continue
        p[te] = ols(Xm[~te], ym[~te], Xm[te])
    f = np.isfinite(p); return int(m.sum()), 1 - np.sum((ym[f] - p[f]) ** 2) / np.sum((ym[f] - ym[f].mean()) ** 2)
brows = []
for tgt in ("Y3n", "Yw"):
    for xc in ("prev_w", "prev_trend1", "prev_Y3n", "prev_level", "ep_pos", "run_hi", "season"):
        for nm, mask in (("ep", mask_ep), ("hi", mask_hi_all & R.prev_rec.notna())):
            m = mask & T[xc].notna() & T[tgt].notna()
            if m.sum() < 8: continue
            r_, p_ = stats.pearsonr(T[xc][m], T[tgt][m]); lo, hi_ = boot_r(T[xc][mask], T[tgt][mask], CL[mask])
            n_, r2 = loco_r2([xc], T[tgt], CL, mask)
            brows.append(dict(target=tgt, x=xc, set=nm, n=int(m.sum()), r=r_, p=p_, ci_lo=lo, ci_hi=hi_, loco_r2=r2))
            print(f"  {tgt:4s} ~ {xc:11s} [{nm}] n {m.sum():2d} r {r_:+.3f} (p {p_:.3f}) CI [{lo:+.2f},{hi_:+.2f}] LOCO R2 {r2:+.3f}")
# all-record within-day persistence (reference for 6.635 rho .29)
m = R.prev_rec.notna() & T.prev_w.notna()
print(f"  참고(전체 링크) Yw ~ prev_w n {m.sum()} r {stats.pearsonr(T.prev_w[m], T.Yw[m])[0]:+.3f} rho {stats.spearmanr(T.prev_w[m], T.Yw[m])[0]:+.3f}")
n_, r2 = loco_r2(["prev_w", "prev_trend1", "prev_level", "ep_pos"], T.Y3n, CL, mask_ep)
print(f"  Y3n ~ 4개 함께 [ep] n {n_} LOCO R2 {r2:+.3f}")
n_, r2 = loco_r2(["prev_w", "prev_trend1", "prev_level", "ep_pos"], T.Yw, CL, mask_ep)
print(f"  Yw  ~ 4개 함께 [ep] n {n_} LOCO R2 {r2:+.3f}")
pd.DataFrame(brows).to_csv(os.path.join(HERE, "ep3_trend_v1.csv"), index=False, encoding="utf-8-sig")

# ---------------- C. residual decomposition ----------------
print("\n=== C. 고EC 잔차 분해: 0시 수준 오차 + 하루 안 모양 오차 ===")
e0 = (R.ec_h0 - R.sg_h0)[H]; shape = Y2[H] - e0
vt = Y2[H].var(); print(f"  n {len(H)} | 잔차 평균 {Y2[H].mean():+.3f} = 0시 오차 {e0.mean():+.3f} + 모양 오차 {shape.mean():+.3f}")
print(f"  분산: 전체 {vt:.4f} | 0시 오차 {e0.var():.4f} | 모양 {shape.var():.4f} | 2cov {2*np.cov(e0, shape)[0,1]:.4f}; 평균제곱: 전체 {np.mean(Y2[H]**2):.4f} 0시 {np.mean(e0**2):.4f} 모양 {np.mean(shape**2):.4f}")
for ps_ in (0, 1):
    hh = [r for r in H if R.at[r, "pass2"] == ps_]
    print(f"  pass{ps_+1} n {len(hh)}: 잔차 평균 {Y2[hh].mean():+.3f} 0시 오차 {e0[hh].mean():+.3f} 모양 {shape[hh].mean():+.3f}")
# 지속 오라클: 0시를 전날 23시로 맞췄다면 (sg 모양 유지) 남는 잔차
hp = [r for r in H if isinstance(R.at[r, "prev_rec"], str)]
alt = R.ec_mean[hp] - (R.sg_mean[hp] - R.sg_h0[hp] + R.prev_ec_h23[hp])
print(f"  [진단 오라클] 0시를 진짜 전날 23시로 이동(모양은 모델) n {len(hp)}: 고EC RMSE {np.sqrt(np.mean(Y2[hp]**2)):.3f} -> {np.sqrt(np.mean(alt**2)):.3f}; 개선 기록 {int(np.sum(alt.abs() < Y2[hp].abs()))}/{len(hp)}")
pp = [r for r in R.index[ok] if isinstance(R.at[r, "prev_rec"], str) and not hi[r]]
alt2 = R.ec_mean[pp] - (R.sg_mean[pp] - R.sg_h0[pp] + R.prev_ec_h23[pp])
print(f"  [진단 오라클] 같은 이동, 보통 기록 n {len(pp)}: RMSE {np.sqrt(np.mean(Y2[pp]**2)):.3f} -> {np.sqrt(np.mean(alt2**2)):.3f}")

# ---------------- D. eval reach & steps distribution shift ----------------
print("\n=== D. 평가 60 에서 지속 경로가 닿는 곳 ===")
print(f"  학습 CV 고EC steps 분포 {V.u_lk_steps[hi].value_counts().sort_index().to_dict()} | 학습 CV 전체 {V.u_lk_steps.value_counts().sort_index().to_dict()}")
print(f"  평가 steps 분포 {EV.steps_to_anchor.value_counts().sort_index().to_dict()} (평가는 학습 정답 전부 보임 -> 1단계가 다수; 학습 CV 는 블록 10개 가림)")
hiA = EV[EV.anchor_ec23 >= 1.0]
print(f"  평가 앵커 23시 EC >= 1.0: {len(hiA)} 기록 -> {hiA[['steps_to_anchor','anchor','anchor_conf','anchor_ec23']].round(3).to_dict('index')}")
print(f"  배타성 후보 평가 기록: {COV[COV.excl_n > 0][['excl_rec','excl_ec23']].to_dict('index')}")
print(f"  평가 0시 쌍둥이 중 학습 고EC 있음: {int((COV.n_twin_hi > 0).sum())} 기록 (같은 날 다른 동이 고EC -> 배타성상 이 기록은 고EC 가능성 낮음, 서술)")
