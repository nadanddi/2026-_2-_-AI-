# -*- coding: utf-8 -*-
"""HX9 / EP1 시간·에피소드 구조(정답 구조) 갈래 — 2026-10-13 집 클로드.
사용자 요청: '고EC 의 크기 예측에 인과관계가 조금이라도 있는 데이터 전부 확인'. 이 갈래 = 지속(AR)·추세·에피소드 위치·같은 날짜 다른 동.

=== 계산 전에 적은 예측 (지문) — 이 메커니즘(지속/상태 기억)이 고EC '크기'를 만든다면 ===
P1 (지속) 고EC 기록의 하루 평균 EC(Y1)는 진단용 깨끗한 전날 EC 와 강하게 이어진다: r >= .7, 기울기 .7~1.0.
P2 (모델이 모르는 상태) 고EC 안 모델 잔차(Y2 = y - sg)는 '전날 EC - 오늘 예측'(gap) 과 양의 상관 r >= .5, 전날 잔차와 r >= .35.
P3 (추세) 누적형(염류가 쌓임)이라면 에피소드 안 위치와 수준이 함께 오르고(ρ > .3), 전날 변화(Δ) 가 오늘 변화(Y3)와 같은 부호(ρ > .3).
    상태 전환형이라면 위치와 무관(ρ ≈ 0), Y3 은 앞 변화와 무관 또는 음(되돌림, ρ < 0), 전날 수준이 높을수록 Y3 음(평균 회귀).
    사전 판단(카탈로그 6.217 AG0: 지속 일수와 부족분 무관): 상태 전환형 + 되돌림 쪽을 예상.
P4 (같은 날짜 다른 동) EC 수준은 동 고유 상태(6.146·6.402) → 같은 날 다른 동 EC(쌍둥이)는 고EC 크기·잔차와 |r| < .2.
    배타성(같은 날 두 동이 함께 고EC 인 적 없음) 때문에 '전날 고EC 였는데 오늘 쌍둥이 중엔 없는' 값(unclaimed)은 크기와 양의 상관을 보여야 함.
P5 (사용 가능판) 입력만 쓴 링커(v4 CV) 앵커로 바꾸면 진단판 신호의 일부(적중률 ~.77 비율, 1단계·확신 구간은 대부분)가 남아야 함.
    고EC 는 앵커가 대개 2단계 이상 떨어져(블록 가림) 신호가 많이 줄 것으로 예상.
보이면 안 되는 것 (보이면 이 해석을 의심):
 N1 다음날 EC(미래)가 전날 EC 보다 Y1 을 훨씬 더 잘 설명(대칭이 아니면 링크/달력 인공물 의심).
 N2 쌍둥이 EC 가 고EC 잔차와 양의 상관(날짜 공통 요인 = 계절/농장 전체 관리 쪽 해석이 맞게 됨).
 N3 진단판보다 사용 가능판(링커 앵커)이 더 강함(누설 의심).
 N4 2차(번호 >= 179)에서 1차와 반대 부호.
관찰과 해석 구분: 이 스크립트는 '연관 정도'와 '평가 시점에서 계산 가능한지'만 판정한다. 인과 해석(염류 누적, 급액 관리)은 일반 농업 지식이며 이 자료로 검증 안 됨.

자료 (읽기만): hx9/data/records.csv, local/linker_v4/cv_all400_choice_v4.csv (입력만 쓴 전날 링커의 블록 CV 선택: 대상 블록 ±1 정답 가림),
  local/linker_v4/eval60_predecessor_v4.csv, local/linker_v3/truth.pkl (학습 기록끼리 정답 연속 a1), env.DATA test_X/train_X (평가 포괄 계산).
가림 규칙(사용 가능판): 학습 대상 b 의 링커 CV 블록(같은 온실 번호 [s-1, s+10]) 기록은 정답이 없는 것으로 취급(쌍둥이·전날 날짜·다음날 날짜·앵커 사슬 모두).
출력: hx9/episode/vars_v1.csv (기록별 변수), ep1_results_v1.csv (변수 × 목표 통계), ep1_eval_cov_v1.csv (평가 60 포괄), 로그.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
HX9 = os.path.dirname(HERE)
RES = os.path.dirname(HX9)
rng = np.random.default_rng(20261013)
NB = 2000

R = pd.read_csv(os.path.join(HX9, "data", "records.csv")).set_index("rec")
LK = pd.read_csv(os.path.join(RES, "local", "linker_v4", "cv_all400_choice_v4.csv"), encoding="utf-8-sig").set_index("rec")
TL = pd.read_pickle(os.path.join(RES, "local", "linker_v3", "truth.pkl"))
EV = pd.read_csv(os.path.join(RES, "local", "linker_v4", "eval60_predecessor_v4.csv"), encoding="utf-8-sig")
idx2rec = {int(i): r for r, i in R.idx.items()}

# ---------- hidden block H(b) (같은 온실 번호 [s-1, s+10]) ----------
LO = {"F13": 8, "F47": 6}
sys.path.insert(0, os.path.join(RES, "linker_v4"))
from feat4 import FARM, NUM, N, TRAIN  # noqa: E402
for f in LO: LO[f] = int(NUM[FARM == f].min())
def block_start(r):
    if r in LK.index:
        s = int(LK.at[r, "block"].split("_")[1].split("-")[0]); return s
    f, n = r[:3], int(r[4:]); return LO[f] + 10 * ((n - LO[f]) // 10)
HID = {}
for r in R.index:
    s = block_start(r); f = r[:3]
    HID[r] = {x for x in R.index if x[:3] == f and s - 1 <= int(x[4:]) <= s + 10}
assert all(r in HID[r] for r in R.index)

# ---------- label-continuity predecessor among TRAINING records (a1, cert) — usable when the start is a training record ----------
A1 = {}
for b, a, cert in zip(TL.b, TL.a1, TL.cert):
    if bool(cert): A1[idx2rec[int(b)]] = idx2rec[int(a)]

E = R.ec_mean.to_dict(); E23 = R.ec_h23.to_dict(); RS = R.resid_mean.to_dict(); HI = (R.ec_mean >= 1.0).to_dict()
PREV = R.prev_rec.dropna().to_dict()

V = pd.DataFrame(index=R.index)
# ================= DIAGNOSTIC (진단용: 깨끗한 진짜 사슬 = b 자신 정답 + DM11 사용) =================
def back(r, k):
    x = r
    for _ in range(k):
        x = PREV.get(x)
        if x is None: return None
    return x
p1 = {r: back(r, 1) for r in R.index}; p2 = {r: back(r, 2) for r in R.index}; p3 = {r: back(r, 3) for r in R.index}
V["d_prev_mean"] = [E[p1[r]] if p1[r] else np.nan for r in R.index]
V["d_prev_h23"] = [E23[p1[r]] if p1[r] else np.nan for r in R.index]
V["d_prev2_mean"] = [E[p2[r]] if p2[r] else np.nan for r in R.index]
V["d_trend1"] = V.d_prev_mean - V.d_prev2_mean
def slope3(r):
    xs = [p3[r], p2[r], p1[r]]
    if any(x is None for x in xs): return np.nan
    return np.polyfit([0, 1, 2], [E[x] for x in xs], 1)[0]
V["d_trend3"] = [slope3(r) for r in R.index]
V["d_prev_mom"] = V.d_prev_h23 - V.d_prev_mean
V["d_prev_resid"] = [RS[p1[r]] if p1[r] else np.nan for r in R.index]
V["d_gap"] = V.d_prev_mean - R.sg_mean            # 전날 수준 - 오늘 모델 예측
V["d_gap23"] = V.d_prev_h23 - R.sg_mean
V["d_prev_max3"] = [max([E[x] for x in (p1[r], p2[r], p3[r]) if x]) if p1[r] else np.nan for r in R.index]
def run_hi(r, prevmap):
    k = 0; x = prevmap.get(r)
    while x is not None and HI[x]:
        k += 1; x = prevmap.get(x)
        if k > 30: break
    return k, (x is None)
rr = {r: run_hi(r, PREV) for r in R.index}
V["d_run_hi"] = [rr[r][0] if p1[r] else np.nan for r in R.index]
V["d_ep_pos"] = R.ep_pos; V["d_ep_n"] = R.ep_n; V["d_ep_span"] = R.ep_span
V["d_ep_frac"] = R.ep_pos / (R.ep_span - 1).replace(0, np.nan)   # 에피소드 안 상대 위치 (끝을 알아야 함)
V["d_succ_mean"] = R.succ_ec_mean; V["d_succ_h0"] = R.succ_ec_h0
V["d_prev_amp"] = [R.at[p1[r], "ec_amp"] if p1[r] else np.nan for r in R.index]
V["d_dm11_day"] = R.dm11_day                      # 계절 (DM11 = 평가 입력 사용, 서술)

# ================= USABLE (사용 가능: 입력만 쓴 링커 CV 앵커 + 학습 정답, 블록 가림) =================
def lk(r, c):
    return LK.at[r, c] if r in LK.index else np.nan
V["u_lk_steps"] = [lk(r, "steps_to_anchor") for r in R.index]
V.loc[V.u_lk_steps < 0, "u_lk_steps"] = np.nan
V["u_lk_conf"] = [lk(r, "anchor_conf") for r in R.index]
V["u_lk_p1"] = [lk(r, "p") for r in R.index]
anc = {r: (LK.at[r, "anchor"] if r in LK.index and isinstance(LK.at[r, "anchor"], str) else None) for r in R.index}
for r in R.index:   # sanity: 앵커는 가림 블록 밖
    if anc[r] is not None: assert anc[r] not in HID[r], (r, anc[r])
V["u_lk_ec23"] = [E23[anc[r]] if anc[r] else np.nan for r in R.index]
V["u_lk_mean"] = [E[anc[r]] if anc[r] else np.nan for r in R.index]
V["u_lk_hi"] = [float(HI[anc[r]]) if anc[r] else np.nan for r in R.index]
conf1 = (V.u_lk_steps == 1) & (V.u_lk_conf >= 0.8)
V["u_lk_ec23_conf"] = np.where(conf1, V.u_lk_ec23, np.nan)     # 확신 1단계 앵커만
V["u_lk_resid"] = [RS[anc[r]] if anc[r] else np.nan for r in R.index]   # 앵커 기록의 OOF 잔차
V["u_lk_mom"] = [E23[anc[r]] - E[anc[r]] if anc[r] else np.nan for r in R.index]
V["u_lk_gap"] = V.u_lk_ec23 - R.sg_mean
V["u_lk_gap_conf"] = V.u_lk_ec23_conf - R.sg_mean
def a1_vis(x, r):
    y = A1.get(x)
    if y is None or y in HID[r] or y == r: return None
    return y
def lk_back(r):
    a = anc[r]
    if a is None: return np.nan, np.nan, np.nan
    b2 = a1_vis(a, r)
    tr = E[a] - E[b2] if b2 else np.nan
    k = 0; x = a
    while x is not None and HI[x]:
        k += 1; x = a1_vis(x, r)
        if k > 30: break
    return tr, k, (E[b2] if b2 else np.nan)
lb = {r: lk_back(r) for r in R.index}
V["u_lk_trend"] = [lb[r][0] for r in R.index]
V["u_lk_run_hi"] = [lb[r][1] for r in R.index]             # 앵커까지 이어진 고EC 일수 (앵커 포함)
V["u_lk_dsn"] = np.where(V.u_lk_run_hi > 0, V.u_lk_run_hi + V.u_lk_steps - 1, 0.0)  # 마지막 보통 날 이후 추정 일수
V.loc[V.u_lk_run_hi.isna(), "u_lk_dsn"] = np.nan

# ---- same date / prev date / next date (DMT3e seg-day; 쌍둥이 = 0시 실외 4변수 동일 학습 기록) ----
SD = R[["dmt_seg", "dmt_day"]].copy()
bydate = {}
for r, (s, d) in SD.iterrows(): bydate.setdefault((int(s), int(d)), []).append(r)
def group_stats(recs, r, pre):
    recs = [x for x in recs if x != r and x not in HID[r]]
    ev = [E[x] for x in recs]
    out = {pre + "_n": len(recs), pre + "_max": max(ev) if ev else np.nan, pre + "_mean": np.mean(ev) if ev else np.nan,
           pre + "_nhi": sum(e >= 1.0 for e in ev) if ev else np.nan}
    return out, recs
rows = []
for r in R.index:
    s, d = int(SD.at[r, "dmt_seg"]), int(SD.at[r, "dmt_day"])
    tw = R.at[r, "twin_recs"]; tw = tw.split(";") if isinstance(tw, str) and tw else []
    o = {}
    a, twv = group_stats(tw, r, "u_tw"); o.update(a)
    b_, pdv = group_stats(bydate.get((s, d - 1), []), r, "u_pd"); o.update(b_)
    c_, ndv = group_stats(bydate.get((s, d + 1), []), r, "u_nd"); o.update(c_)
    # 배타성: 어제 고EC 값 중 오늘 쌍둥이 고EC 와 짝 지어지지 않은 것
    ph = sorted([E[x] for x in pdv if HI[x]], reverse=True); th = [E[x] for x in twv if HI[x]]
    rem = list(ph)
    for t in th:
        if rem:
            j = int(np.argmin([abs(t - p) for p in rem])); rem.pop(j)
    o["u_unclaimed_n"] = len(rem) if len(pdv) else np.nan
    o["u_unclaimed_max"] = (max(rem) if rem else 0.0) if len(pdv) else np.nan
    rows.append(dict(rec=r, **o))
G = pd.DataFrame(rows).set_index("rec")
V = V.join(G)
V["u_tw_trend"] = V.u_tw_mean - V.u_pd_mean
V["u_dmt_day"] = R.dmt_day; V["u_dmt_seg"] = R.dmt_seg
V["u_pass2"] = R.pass2
V.to_csv(os.path.join(HERE, "vars_v1.csv"), encoding="utf-8")

USABLE = {
 "d_prev_mean": "diag (b 자신 정답으로 고른 링크)", "d_prev_h23": "diag", "d_prev2_mean": "diag", "d_trend1": "diag", "d_trend3": "diag",
 "d_prev_mom": "diag", "d_prev_resid": "diag", "d_gap": "diag", "d_gap23": "diag", "d_prev_max3": "diag", "d_run_hi": "diag",
 "d_ep_pos": "diag (b 자신 고EC 라벨)", "d_ep_n": "diag (미래 포함)", "d_ep_span": "diag (미래 포함)", "d_ep_frac": "diag (미래 포함)",
 "d_succ_mean": "diag (미래 정답·링크)", "d_succ_h0": "diag (미래)", "d_prev_amp": "diag", "d_dm11_day": "descriptive (평가 입력 달력)",
 "u_lk_steps": "usable (h=0 입력 링커)", "u_lk_conf": "usable", "u_lk_p1": "usable", "u_lk_ec23": "usable", "u_lk_mean": "usable",
 "u_lk_hi": "usable", "u_lk_ec23_conf": "usable", "u_lk_resid": "usable (앵커 OOF)", "u_lk_mom": "usable",
 "u_lk_gap": "usable (sg_h 로 바꿔 시각별)", "u_lk_gap_conf": "usable (sg_h)", "u_lk_trend": "usable (앵커의 학습 정답 사슬)",
 "u_lk_run_hi": "usable", "u_lk_dsn": "usable",
 "u_tw_n": "usable (0시 실외 쌍둥이)", "u_tw_max": "usable", "u_tw_mean": "usable", "u_tw_nhi": "usable",
 "u_pd_n": "usable-in-principle (DMT 날짜; 평가는 쌍둥이로 날짜)", "u_pd_max": "usable-in-principle", "u_pd_mean": "usable-in-principle",
 "u_pd_nhi": "usable-in-principle", "u_nd_n": "usable-in-principle (뒷날 학습 정답 허용 10-12)", "u_nd_max": "usable-in-principle",
 "u_nd_mean": "usable-in-principle", "u_nd_nhi": "usable-in-principle", "u_unclaimed_n": "usable-in-principle", "u_unclaimed_max": "usable-in-principle",
 "u_tw_trend": "usable-in-principle", "u_dmt_day": "usable-in-principle", "u_dmt_seg": "usable-in-principle", "u_pass2": "usable",
}
VARS = [c for c in V.columns if c in USABLE]
assert len(VARS) == len(V.columns), set(V.columns) - set(USABLE)

# ================= targets / clusters =================
hi = R.ec_mean >= 1.0
ok = R.has_sg == 1
Y1 = R.ec_mean
Y2 = R.resid_mean
# Y3: 고EC 에피소드 안 깨끗한 링크 하루 평균 변화
y3mask = hi & R.prev_rec.notna() & (R.prev_ec_mean >= 1.0)
Y3 = R.ec_mean - R.prev_ec_mean
# 군집: 고EC = ep_id, 전체 = 깨끗한 링크 연결 성분
par = {r: r for r in R.index}
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for b, a in PREV.items(): par[fd(b)] = fd(a)
COMP = pd.Series({r: fd(r) for r in R.index})
CL_hi = R.ep_id.copy()
CL_all = COMP.copy()

def boot_r(x, y, cl, fn="pearson"):
    m = x.notna() & y.notna(); x, y, cl = x[m].values, y[m].values, cl[m].values
    u = np.unique(cl); out = []
    gi = {c: np.where(cl == c)[0] for c in u}
    for _ in range(NB):
        s = rng.choice(u, len(u), replace=True); ii = np.concatenate([gi[c] for c in s])
        if np.std(x[ii]) < 1e-12 or np.std(y[ii]) < 1e-12: continue
        out.append(stats.pearsonr(x[ii], y[ii])[0] if fn == "pearson" else stats.spearmanr(x[ii], y[ii])[0])
    return np.nanpercentile(out, [2.5, 97.5]) if len(out) > 50 else (np.nan, np.nan)

def ols_fit_pred(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr]); B = np.column_stack([np.ones(len(Xte)), Xte])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None); return B @ w
def loco_gain(x, y, sg, cl):
    """Y2 ~ 1 + sg  vs  Y2 ~ 1 + sg + x ; leave-one-cluster-out; rows where x,y,sg finite. returns (n, R2_base, R2_aug, gain, n_improved)."""
    m = x.notna() & y.notna() & sg.notna(); x, y, sg, cl = x[m].values, y[m].values, sg[m].values, cl[m].values
    if len(y) < 8 or np.std(x) < 1e-12: return len(y), np.nan, np.nan, np.nan, np.nan
    pb = np.full(len(y), np.nan); pa = np.full(len(y), np.nan)
    for c in np.unique(cl):
        te = cl == c; tr = ~te
        if tr.sum() < 5: continue
        pb[te] = ols_fit_pred(sg[tr, None], y[tr], sg[te, None])
        pa[te] = ols_fit_pred(np.column_stack([sg[tr], x[tr]]), y[tr], np.column_stack([sg[te], x[te]]))
    f = np.isfinite(pb) & np.isfinite(pa)
    sst = np.sum((y[f] - y[f].mean()) ** 2)
    r2b = 1 - np.sum((y[f] - pb[f]) ** 2) / sst; r2a = 1 - np.sum((y[f] - pa[f]) ** 2) / sst
    return int(f.sum()), r2b, r2a, r2a - r2b, int(np.sum(np.abs(y[f] - pa[f]) < np.abs(y[f] - pb[f])))

def corr(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 5 or x[m].std() < 1e-12: return m.sum(), np.nan, np.nan, np.nan, np.nan
    pr = stats.pearsonr(x[m], y[m]); sr = stats.spearmanr(x[m], y[m])
    return int(m.sum()), pr[0], pr[1], sr[0], sr[1]

res = []
for v in VARS:
    x = V[v]
    d = dict(variable=v, usable=USABLE[v])
    # Y1: 고EC 하루 평균
    n, r, p, s, sp = corr(x[hi], Y1[hi]); d.update(Y1_n=n, Y1_r=r, Y1_p=p, Y1_rho=s)
    # Y2: 고EC 잔차 (OOF 있음)
    m2 = hi & ok
    n, r, p, s, sp = corr(x[m2], Y2[m2]); d.update(Y2h_n=n, Y2h_r=r, Y2h_p=p, Y2h_rho=s, Y2h_rho_p=sp)
    if n >= 8 and np.isfinite(r):
        lo_, hi_ = boot_r(x[m2], Y2[m2], CL_hi[m2]); d.update(Y2h_ci_lo=lo_, Y2h_ci_hi=hi_)
        n_, b_, a_, g_, imp = loco_gain(x[m2], Y2[m2], R.sg_mean[m2], CL_hi[m2]); d.update(Y2h_cv_n=n_, Y2h_cv_base=b_, Y2h_cv_aug=a_, Y2h_cv_gain=g_, Y2h_cv_improved=imp)
        # 기록 단위 LOO (민감도)
        n_, b_, a_, g_, imp = loco_gain(x[m2], Y2[m2], R.sg_mean[m2], pd.Series(np.arange(m2.sum()), index=R.index[m2])); d.update(Y2h_loo_gain=g_)
        # 1차/2차
        for ps_, nm in ((0, "p1"), (1, "p2")):
            mm = m2 & (R.pass2 == ps_)
            nn, rr_, *_ = corr(x[mm], Y2[mm]); d[f"Y2h_r_{nm}"] = rr_; d[f"Y2h_n_{nm}"] = nn
    # Y2 전체 기록
    n, r, p, s, sp = corr(x[ok], Y2[ok]); d.update(Y2a_n=n, Y2a_r=r, Y2a_p=p, Y2a_rho=s)
    if n >= 20 and np.isfinite(r):
        lo_, hi_ = boot_r(x[ok], Y2[ok], CL_all[ok]); d.update(Y2a_ci_lo=lo_, Y2a_ci_hi=hi_)
        n_, b_, a_, g_, imp = loco_gain(x[ok], Y2[ok], R.sg_mean[ok], CL_all[ok]); d.update(Y2a_cv_gain=g_, Y2a_cv_n=n_)
        for ps_, nm in ((0, "p1"), (1, "p2")):
            mm = ok & (R.pass2 == ps_)
            nn, rr_, *_ = corr(x[mm], Y2[mm]); d[f"Y2a_r_{nm}"] = rr_
    # Y3: 고EC 에피소드 안 하루 변화
    n, r, p, s, sp = corr(x[y3mask], Y3[y3mask]); d.update(Y3_n=n, Y3_r=r, Y3_p=p, Y3_rho=s)
    if n >= 8 and np.isfinite(r):
        lo_, hi_ = boot_r(x[y3mask], Y3[y3mask], CL_hi[y3mask]); d.update(Y3_ci_lo=lo_, Y3_ci_hi=hi_)
    res.append(d)
RES_ = pd.DataFrame(res)
# 다중검정 (Y2 고EC 피어슨 p): BH
def multipletests(p, method="fdr_bh"):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return None, np.minimum(q, 1)
m = RES_.Y2h_p.notna()
RES_.loc[m, "Y2h_q_bh"] = multipletests(RES_.loc[m, "Y2h_p"], method="fdr_bh")[1]
m = RES_.Y2a_p.notna()
RES_.loc[m, "Y2a_q_bh"] = multipletests(RES_.loc[m, "Y2a_p"], method="fdr_bh")[1]
def verdict(row):
    r, lo_, hi_, g = row.Y2h_r, row.get("Y2h_ci_lo", np.nan), row.get("Y2h_ci_hi", np.nan), row.get("Y2h_cv_gain", np.nan)
    if not np.isfinite(r): return "none"
    ci_ex = np.isfinite(lo_) and (lo_ > 0 or hi_ < 0)
    if abs(r) >= .35 and ci_ex and np.isfinite(g) and g > 0: return "signal"
    if (abs(r) >= .35) or ci_ex or (np.isfinite(g) and g > 0.02): return "weak"
    return "none"
RES_["verdict_Y2h"] = RES_.apply(verdict, axis=1)
RES_.to_csv(os.path.join(HERE, "ep1_results_v1.csv"), index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
print(f"variables tested: {len(VARS)} (diag {sum(v.startswith('d_') for v in VARS)}, usable {sum(v.startswith('u_') for v in VARS)}); targets Y1 hi n={int(hi.sum())}, Y2 hi n={int((hi&ok).sum())}, Y2 all n={int(ok.sum())}, Y3 n={int(y3mask.sum())}")
print(f"clusters: high episodes {CL_hi[hi].nunique()} (sizes {CL_hi[hi].value_counts().sort_index().tolist()}); all-record link components {CL_all.nunique()} (largest {CL_all.value_counts().max()})")
cols = ["variable", "Y1_n", "Y1_r", "Y2h_n", "Y2h_r", "Y2h_rho", "Y2h_ci_lo", "Y2h_ci_hi", "Y2h_q_bh", "Y2h_cv_gain", "Y2h_loo_gain", "Y2h_cv_improved",
        "Y2h_r_p1", "Y2h_r_p2", "Y2h_n_p2", "Y2a_r", "Y2a_ci_lo", "Y2a_ci_hi", "Y2a_cv_gain", "Y2a_r_p1", "Y2a_r_p2", "Y3_n", "Y3_r", "Y3_ci_lo", "Y3_ci_hi", "verdict_Y2h"]
print(RES_[cols].round(3).to_string(index=False))

# ================= persistence (AR) quantification =================
print("\n=== 지속(AR) 정량 ===")
def loco_r2(Xcols, y, cl, mask):
    D = pd.concat([V, R[["sg_mean", "sg_h0"]]], axis=1)
    m = mask & y.notna() & D[Xcols].notna().all(1)
    Xm, ym, cm = D.loc[m, Xcols].values, y[m].values, cl[m].values
    p = np.full(len(ym), np.nan)
    for c in np.unique(cm):
        te = cm == c; p[te] = ols_fit_pred(Xm[~te], ym[~te], Xm[te])
    f = np.isfinite(p); sst = np.sum((ym[f] - ym[f].mean()) ** 2)
    A = np.column_stack([np.ones(m.sum()), Xm]); w, *_ = np.linalg.lstsq(A, ym, rcond=None)
    ins = 1 - np.sum((ym - A @ w) ** 2) / sst
    return int(m.sum()), ins, 1 - np.sum((ym[f] - p[f]) ** 2) / sst, w
for nm, mask in (("고EC(Y1)", hi & ok), ("전체(EC 수준)", ok)):
    cl = CL_hi if nm.startswith("고") else CL_all
    for xc in (["sg_mean"], ["d_prev_mean"], ["d_prev_h23"], ["sg_mean", "d_prev_mean"], ["sg_mean", "d_prev_h23"], ["u_lk_ec23"], ["sg_mean", "u_lk_ec23"], ["sg_mean", "u_lk_ec23_conf"]):
        n, ins, cv, w = loco_r2(xc, Y1, cl, mask & V.d_prev_mean.notna() if xc[-1] != "u_lk_ec23_conf" else mask)
        print(f"  {nm:12s} Y1 ~ {'+'.join(xc):28s} n {n:3d}  R2 in {ins:.3f}  LOCO {cv:.3f}  coef {np.round(w,3)}")
# lag decay along clean chains
print("  지연 k 별 상관 (같은 진짜 사슬, 하루 평균):")
for k in range(1, 7):
    pk = {r: back(r, k) for r in R.index}
    xa = pd.Series({r: E[pk[r]] if pk[r] else np.nan for r in R.index})
    for nm, mask in (("all", pd.Series(True, index=R.index)), ("hi", hi)):
        n, r_, *_ = corr(xa[mask], Y1[mask])
        print(f"    k={k} {nm:3s} n {n:3d} r {r_:.3f}", end="")
    # residual persistence among hi
    xr = pd.Series({r: RS[pk[r]] if pk[r] else np.nan for r in R.index})
    n, r_, *_ = corr(xr[hi & ok], Y2[hi & ok]); print(f" | resid(hi) n {n:3d} r {r_:.3f}")
# AR on residual and slope (mean reversion)
m = hi & V.d_prev_mean.notna()
sl = stats.linregress(V.d_prev_mean[m], Y1[m]); print(f"  고EC Y1 ~ prev: slope {sl.slope:.3f} (se {sl.stderr:.3f}) int {sl.intercept:.3f} r {sl.rvalue:.3f} n {m.sum()}")
m = hi & V.d_prev_mean.notna() & (V.d_prev_mean >= 1.0)
sl = stats.linregress(V.d_prev_mean[m], Y1[m]); print(f"  고EC(전날도 고) Y1 ~ prev: slope {sl.slope:.3f} (se {sl.stderr:.3f}) r {sl.rvalue:.3f} n {m.sum()}")
print(f"  고EC 첫날(전날 보통/없음) vs 이후 잔차: 첫날 {Y2[hi & ok & ~(V.d_prev_mean >= 1)].mean():+.3f} (n {(hi & ok & ~(V.d_prev_mean >= 1)).sum()}) / 이후 {Y2[hi & ok & (V.d_prev_mean >= 1)].mean():+.3f} (n {(hi & ok & (V.d_prev_mean >= 1)).sum()})")
# |Δ| day-to-day within episodes vs normal
mN = (~hi) & R.prev_rec.notna() & (R.prev_ec_mean < 1.0)
print(f"  하루 변화 |Δ| 중앙값: 고EC 에피소드 안 {Y3[y3mask].abs().median():.3f} (sd {Y3[y3mask].std():.3f}, n {y3mask.sum()}) vs 보통→보통 {(R.ec_mean - R.prev_ec_mean)[mN].abs().median():.3f} (sd {(R.ec_mean - R.prev_ec_mean)[mN].std():.3f}, n {mN.sum()})")
print(f"  모델 잔차 sd 고EC {Y2[hi & ok].std():.3f} vs 전날 그대로(persistence) 오차 sd {(Y1 - V.d_prev_mean)[hi & ok & V.d_prev_mean.notna()].std():.3f} / 모델 오차 같은 행 {Y2[hi & ok & V.d_prev_mean.notna()].std():.3f}")
# 추세: Y3 ~ 앞 변화, 수준, 모델 예측 변화
print("\n=== 추세(Y3) ===")
sgprev = pd.Series({r: R.at[p1[r], "sg_mean"] if p1[r] else np.nan for r in R.index})
V3 = pd.DataFrame({"trend1": V.d_trend1, "prev_level": V.d_prev_mean, "prev_mom": V.d_prev_mom, "ep_pos": V.d_ep_pos,
                   "model_change": R.sg_mean - sgprev, "u_lk_trend": V.u_lk_trend})
for c in V3.columns:
    n, r_, p_, s_, _ = corr(V3[c][y3mask], Y3[y3mask])
    lo_, hi_ = boot_r(V3[c][y3mask], Y3[y3mask], CL_hi[y3mask]) if n >= 8 else (np.nan, np.nan)
    sgn = (np.sign(V3[c][y3mask]) == np.sign(Y3[y3mask]))[V3[c][y3mask].notna()].mean() if c in ("trend1", "model_change", "u_lk_trend") else np.nan
    print(f"  Y3 ~ {c:12s} n {n:3d} r {r_:+.3f} (p {p_:.3f}) rho {s_:+.3f} CI [{lo_:+.2f},{hi_:+.2f}]  sign-agree {sgn:.2f}")
# LOCO R2 of Y3 from model_change and trend1
for xc in (["model_change"], ["trend1"], ["prev_level"], ["model_change", "prev_level"], ["model_change", "trend1", "prev_level", "prev_mom"]):
    D = V3[xc]; m = y3mask & D.notna().all(1)
    ym, Xm, cm = Y3[m].values, D[m].values, CL_hi[m].values
    p = np.full(len(ym), np.nan)
    for c in np.unique(cm):
        te = cm == c; p[te] = ols_fit_pred(Xm[~te], ym[~te], Xm[te])
    f = np.isfinite(p); print(f"  Y3 LOCO R2 {'+'.join(xc):45s} n {m.sum():3d} R2 {1 - np.sum((ym[f]-p[f])**2)/np.sum((ym[f]-ym[f].mean())**2):+.3f}")
# N1: 미래(다음날) vs 과거(전날)
m = hi & V.d_prev_mean.notna() & V.d_succ_mean.notna()
print(f"\n  N1 대칭 검사 (고EC, 둘 다 있음 n {m.sum()}): r(prev) {stats.pearsonr(V.d_prev_mean[m], Y1[m])[0]:.3f} vs r(succ) {stats.pearsonr(V.d_succ_mean[m], Y1[m])[0]:.3f}")

# ================= linker usable vs diagnostic agreement on high records =================
print("\n=== 사용 가능판(링커 v4 CV) vs 진단판 ===")
for nm, mask in (("고EC", hi), ("전체", pd.Series(True, index=R.index))):
    mm = mask & R.prev_rec.notna()
    hit = np.mean([LK.at[r, "prev"] == R.at[r, "prev_rec"] for r in R.index[mm] if r in LK.index])
    one = mask & (V.u_lk_steps == 1); c1 = mask & conf1
    print(f"  {nm}: 링커 1순위 = 진짜 전날 {hit:.2f} (n {mm.sum()}) | 앵커 있음 {V.u_lk_ec23[mask].notna().sum()}/{mask.sum()} | 1단계 {one.sum()} | 1단계&p>=.8 {c1.sum()} | steps 분포 {V.u_lk_steps[mask].value_counts().sort_index().to_dict()}")
    m3 = mask & V.u_lk_ec23.notna() & V.d_prev_h23.notna()
    print(f"     앵커 23시 EC vs 진짜 전날 23시 EC r {stats.pearsonr(V.u_lk_ec23[m3], V.d_prev_h23[m3])[0]:.3f} (n {m3.sum()}); |차| 중앙 {np.median(np.abs(V.u_lk_ec23[m3]-V.d_prev_h23[m3])):.3f}")
hh = R.index[hi]
T = pd.DataFrame({"y": Y1[hh], "sg": R.sg_mean[hh], "prev": V.d_prev_mean[hh], "lk_ec23": V.u_lk_ec23[hh], "steps": V.u_lk_steps[hh], "conf": V.u_lk_conf[hh].round(2),
                  "anchor": [anc[r] for r in hh], "true_prev": R.prev_rec[hh], "tw_max": V.u_tw_max[hh], "pd_max": V.u_pd_max[hh], "uncl": V.u_unclaimed_max[hh], "ep": R.ep_id[hh], "pos": R.ep_pos[hh]})
print(T.sort_values(["ep", "pos"]).round(3).to_string())
V.join(R[["ec_mean", "sg_mean", "resid_mean", "hi10", "ep_id", "pass2"]]).to_csv(os.path.join(HERE, "vars_v1_with_targets.csv"), encoding="utf-8")
print("saved", os.path.join(HERE, "vars_v1.csv"), os.path.join(HERE, "ep1_results_v1.csv"))
