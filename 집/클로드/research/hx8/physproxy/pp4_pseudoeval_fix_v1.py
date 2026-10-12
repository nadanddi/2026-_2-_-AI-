# -*- coding: utf-8 -*-
"""PP4 (2026-10-13 집 클로드, hx8/physproxy, 이전 에이전트 중단분 이어서).  PP3 v2 의 가짜 평가 설계 결함 보수 + 남은 점검.
PP3 v2 결함: 에피소드(사슬 성분)가 3개뿐이고 그중 하나가 사례 43개 중 거의 전부 → 그 성분을 빼면 적합 표본이 5쌍(평균 증가 +.3)만 남아
  P1·P2 가 앵커 + .3·k 로 폭주(RMSE 1.19). '물리량이 나쁘다'가 아니라 교차적합 단위가 무너진 것.
보수(사전 고정, 실행 전):
 * 달력 = 학습 전용 local/dmt3e_E0_days.csv (seg, seg 안 day).  seg 사이 날짜 차는 모름.
 * 교차적합 두 가지(둘 다 보고, 판정은 엄한 쪽):
     BUF  : 표적 t 와 같은 seg 에서 |day 차| <= 4 인 기록 + 진단 연결(c1<=5, 양방향) 4걸음 안 기록을 적합에서 제외.
     LOSO+: 표적 seg 전체 + 연결 4걸음 안 기록들의 seg 전체를 적합에서 제외(가장 엄함).
 * 표적 = DIAG10 OOF 하루 평균(3시드 평균) >= 0.9 인 학습 기록(평가 8기록처럼 '예측으로 고름', 정답 선택 편향 없음) 40개.
 * 사례 유형: L = 앵커 없음(평가 F13_223·225, F47_221~227 처럼 가까운 정답이 멀다) / A_k = k(2..4)일 앞 같은 동 23시 정답만 있음
   (평가 F13_221 = F13_217 에서 2일).  사슬 걸음은 cert 또는 c1<=1 연결만.
 * 추정: M(현재 모델 시드별 OOF 하루 평균) / M_icpt(+ 적합 표본 평균 잔차; 닫힌 '펴기'류, 참고) /
   M_AHxc(+ a + b·AHx 인과판: 각 시 h 의 AHx(0..h) 를 하루 평균) / M_AHxf(하루 전체 AHx, 비인과 참고) /
   A_k 만: P0(앵커 지속) · BA(β 섞기, β 는 적합 표본 A 사례에서) · BA_AHxc(M_AHxc 와 앵커 섞기).
 * 판정(사전 고정): M_AHxc 가 '유망'이려면 (i) L 유형에서 BUF·LOSO+ 모두, 3시드 모두 RMSE(M_AHxc) < RMSE(M) 이고 < RMSE(M_icpt),
   (ii) seg 군집 부트스트랩 95% 구간(RMSE 차 vs M) 상한 < 0, (iii) EL1·P2LOO 에서도 같은 방향.  하나라도 어기면 '약함/기각'.
 * (A) PP3 v2 (c) 의 유일한 기준선 초과: vent_resp(환기 열 때 절대습도 변화) 와 같은 동 하루 평균 변화 Δ 의 ρ .70 (기록 순열 95% .60).
   n_ok · 성분 블록 순열 · 평가 8기록 계산 가능 여부 확인(서술).
 * (C) 평가 8기록: AHx 인과판 값과 학습 고예측 분포 비교, M_AHxc 계수(학습 40개 전부, 시드 평균)를 13회차 하루 평균에 붙인 값 — 참고(채택 아님).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pp1_physproxy_v1 import *   # noqa  (데이터·A_tr·A_ev·feats·O·links·yday·episodes·rmse·spear)
import warnings; warnings.filterwarnings("ignore")

F23 = pd.read_csv(os.path.join(here, "pp1_train_feats_full_v1.csv"), index_col=0)
Cal = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")).set_index("rec")
cert = dict(zip(links, TL.cert)); c1 = dict(zip(links, TL.c1))
E = ymat
SEEDS = [47, 1414, 6464]
EV8 = ["F13_221", "F13_223", "F13_225", "F13_237", "F47_221", "F47_223", "F47_225", "F47_227"]

def ahx_causal_daymean(a):
    T, RH, To, RHo = a["in_temp"], a["in_hum"], a["out_temp"], a["out_hum"]
    d = ah(T, RH) - ah(To, np.clip(RHo, 1, 100))
    cm = np.array([np.nanmean(d[:h + 1]) if np.isfinite(d[:h + 1]).any() else np.nan for h in range(24)])
    return float(np.nanmean(cm))
AHXc = pd.Series({r: ahx_causal_daymean(a) for r, a in A_tr.items()})
AHXf = F23.AHx

# ---------- neighbours ----------
nb = {}
for (a, b) in links:
    if c1[(a, b)] <= 5.0:
        nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
def within_steps(r, k=4):
    seen = {r}; fr = {r}
    for _ in range(k):
        fr = {y for x in fr for y in nb.get(x, ())} - seen; seen |= fr
    return seen
pred1 = {}
for (a, b) in links:
    if a in E.index and b in E.index and (cert[(a, b)] or c1[(a, b)] <= 1.0):
        pred1[b] = a

def excl(t, mode, extra=()):
    near = within_steps(t, 4) | set(extra)
    if mode == "BUF":
        s, d = Cal.loc[t, "seg"], Cal.loc[t, "day"]
        same = set(Cal.index[(Cal.seg == s) & ((Cal.day - d).abs() <= 4)])
        return near | same
    segs = {Cal.loc[x, "seg"] for x in near if x in Cal.index}
    return near | set(Cal.index[Cal.seg.isin(segs)])

def qday_of(val, sd):
    Ov = O[O.validator == val]; Os = Ov if sd == "avg" else Ov[Ov.seed == sd]
    return Os.groupby("rec").sg.mean()

qavg = qday_of("DIAG10", "avg")
TGT = [r for r in qavg.index[qavg >= 0.9] if r in E.index]

def fit_lin(x, r):
    ok = np.isfinite(x) & np.isfinite(r)
    if ok.sum() < 5 or np.std(x[ok]) < 1e-9: return 0.0, float(np.nanmean(r))
    b, a = np.polyfit(x[ok], r[ok], 1); return float(b), float(a)

def run_L(val, sd, mode, tgt):
    q = qday_of(val, sd); rows = []
    pool = [r for r in qday_of(val, "avg").index if r in E.index and qday_of(val, "avg")[r] >= 0.9]
    for t in tgt:
        if t not in q.index: continue
        ex = excl(t, mode)
        fit = [r for r in pool if r not in ex and r in q.index]
        if len(fit) < 5: continue
        res = yday[fit].values - q[fit].values
        bc, ac = fit_lin(AHXc[fit].values, res); bf, af = fit_lin(AHXf[fit].values, res)
        rows.append(dict(t=t, seg=Cal.loc[t, "seg"], y=yday[t], M=q[t], M_icpt=q[t] + res.mean(),
                         M_AHxc=q[t] + ac + bc * AHXc[t], M_AHxf=q[t] + af + bf * AHXf[t], nfit=len(fit), b_c=bc))
    return pd.DataFrame(rows)

def boot_seg(D, c, B=2000):
    segs = D.seg.unique(); d = []
    for _ in range(B):
        pick = rng.choice(segs, len(segs)); S = pd.concat([D[D.seg == s] for s in pick])
        d.append(rmse(S[c].values, S.y.values) - rmse(S.M.values, S.y.values))
    d = np.array(d); return float(np.quantile(d, .025)), float(np.quantile(d, .975)), float(np.mean(d >= 0))

if __name__ == "__main__":
    print(f"targets (DIAG10 q_avg>=0.9): {len(TGT)} | segs {Cal.loc[TGT].seg.value_counts().to_dict()}")
    print("AHx causal vs full (targets) spearman %.2f" % spear(AHXc[TGT].values, AHXf[TGT].values))
    # ---------------- (B-L) ----------------
    print("\n(B-L) no-anchor pseudo-eval, DIAG10: RMSE / better-share vs M")
    keep = {}
    for mode in ["BUF", "LOSO+"]:
        for sd in SEEDS:
            D = run_L("DIAG10", sd, mode, TGT); keep[(mode, sd)] = D
            s = f"   {mode:5s} seed {sd}: n {len(D)} nfit med {int(D.nfit.median())}"
            for c in ["M", "M_icpt", "M_AHxc", "M_AHxf"]:
                s += f" | {c} {rmse(D[c].values, D.y.values):.4f}"
                if c != "M": s += f" ({np.mean(np.abs(D[c]-D.y) < np.abs(D.M-D.y)):.2f})"
            s += f" | b_c med {D.b_c.median():+.3f} [{D.b_c.min():+.3f},{D.b_c.max():+.3f}]"
            print(s)
        D = keep[(mode, 47)]
        for c in ["M_icpt", "M_AHxc", "M_AHxf"]:
            lo, hi, pw = boot_seg(D, c); print(f"      seed 47 {mode} seg-bootstrap RMSE({c})-RMSE(M): [{lo:+.4f}, {hi:+.4f}] P(>=0) {pw:.3f}")
    # true-high vs false-high split of the AHx correction (seed 47, LOSO+)
    D = keep[("LOSO+", 47)].copy(); D["th"] = D.y >= 1.0
    for th, g in D.groupby("th"):
        print(f"   LOSO+ s47 true_high={th}: n {len(g)} RMSE M {rmse(g.M.values,g.y.values):.3f} M_AHxc {rmse(g.M_AHxc.values,g.y.values):.3f} M_icpt {rmse(g.M_icpt.values,g.y.values):.3f}")
    # other validators
    print("\n(B-L) EL1 / P2LOO (2nd-period records, q_avg>=0.9 of that validator)")
    for val in ["EL1", "P2LOO"]:
        qa = qday_of(val, "avg"); tg = [r for r in qa.index[qa >= 0.9] if r in E.index]
        for mode in ["BUF", "LOSO+"]:
            for sd in SEEDS:
                D = run_L(val, sd, mode, tg)
                if len(D) == 0: print(f"   {val} {mode} seed {sd}: no case (fit pool < 5)"); continue
                print(f"   {val} {mode} seed {sd}: n {len(D)} nfit med {int(D.nfit.median())} | M {rmse(D.M.values,D.y.values):.4f} icpt {rmse(D.M_icpt.values,D.y.values):.4f} AHxc {rmse(D.M_AHxc.values,D.y.values):.4f} AHxf {rmse(D.M_AHxf.values,D.y.values):.4f}")
    # ---------------- (B-A) anchor k days before ----------------
    print("\n(B-A) anchor k days before (chain steps cert|c1<=1), DIAG10")
    cases = []
    for t in TGT:
        x = t; path = [t]
        for k in range(1, 5):
            x = pred1.get(x)
            if x is None: break
            path.append(x)
            if k >= 2: cases.append(dict(t=t, k=k, anc=x, hidden=tuple(path[:-1])))
    C = pd.DataFrame(cases); print(f"   cases {len(C)} | k {C.k.value_counts().sort_index().to_dict()} | targets {C.t.nunique()} | segs {Cal.loc[C.t].seg.value_counts().to_dict()}")
    outA = []
    for mode in ["BUF", "LOSO+"]:
        for sd in SEEDS:
            q = qday_of("DIAG10", sd); rows = []
            pool = [r for r in TGT if r in q.index]
            for _, c in C.iterrows():
                ex = excl(c.t, mode, extra=c.hidden)
                fit = [r for r in pool if r not in ex]
                if len(fit) < 5: continue
                res = yday[fit].values - q[fit].values; bc, ac = fit_lin(AHXc[fit].values, res)
                mA = q[c.t] + ac + bc * AHXc[c.t]
                # beta from other anchor cases whose target & anchor are outside the exclusion
                Fc = C[(~C.t.isin(ex)) & (~C.anc.isin(ex)) & (C.k == c.k)]
                def beta(base_fn):
                    if len(Fc) < 3: return 0.0
                    dx = np.array([E.loc[a, 23] - base_fn(tt) for a, tt in zip(Fc.anc, Fc.t)])
                    dy = np.array([yday[tt] - base_fn(tt) for tt in Fc.t])
                    return float(np.clip((dx * dy).sum() / ((dx * dx).sum() + 1e-9), 0, 1))
                bM = beta(lambda tt: q[tt])
                bA = beta(lambda tt: q[tt] + ac + bc * AHXc[tt])
                anc = E.loc[c.anc, 23]
                rows.append(dict(t=c.t, k=c.k, seg=Cal.loc[c.t, "seg"], y=yday[c.t], M=q[c.t], P0=anc, M_AHxc=mA,
                                 BA=q[c.t] + bM * (anc - q[c.t]), BA_AHxc=mA + bA * (anc - mA), bM=bM, nFc=len(Fc)))
            D = pd.DataFrame(rows)
            for kk in ["all"] + sorted(D.k.unique().tolist()):
                S = D if kk == "all" else D[D.k == kk]
                s = f"   {mode:5s} seed {sd} k={kk}: n {len(S)} βM med {S.bM.median():.2f}"
                for col in ["M", "P0", "M_AHxc", "BA", "BA_AHxc"]:
                    s += f" | {col} {rmse(S[col].values, S.y.values):.3f}"
                print(s)
            if mode == "LOSO+" and sd == 47: D.to_csv(os.path.join(here, "pp4_anchor_cases_v1.csv"), index=False)
    # ---------------- (A) vent_resp hit check ----------------
    print("\n(A) vent_resp vs same-house day-mean change (PP3 v2 (c) rho .70)")
    for lab, cond in [("c1<=5", lambda l: c1[l] <= 5.0), ("cert|c1<=1", lambda l: cert[l] or c1[l] <= 1.0)]:
        P = pd.DataFrame([dict(a=a, b=b, d=yday[b] - yday[a]) for (a, b) in links
                          if a in E.index and b in E.index and cond((a, b)) and yday[a] >= 1.0])
        f = F23.loc[P.b, "vent_resp"].values.astype(float); ok = np.isfinite(f)
        comp = episodes(list(set(P.a) | set(P.b))); g = np.array([comp[x] for x in P.b])
        rho = spear(f, P.d.values)
        nullB = []
        ug = np.unique(g); gidx = [np.where(g == u)[0] for u in ug]
        for it in range(2000):
            fq = np.empty_like(f); perm = rng.permutation(len(ug))
            for gi, src in zip(gidx, perm):
                s_ = gidx[src]; fq[gi] = f[s_[np.arange(len(gi)) % len(s_)]]
            nullB.append(abs(spear(fq, P.d.values)) if np.isfinite(fq).sum() > 5 else 0)
        nullR = [abs(spear(f[rng.permutation(len(f))], P.d.values)) for _ in range(2000)]
        print(f"   {lab}: pairs {len(P)}, vent_resp finite {ok.sum()}, components {len(ug)} | rho {rho:+.2f} | single-feature null95 record {np.nanquantile(nullR,.95):.2f}, block {np.nanquantile(nullB,.95):.2f}")
        print("      finite pairs:", [(a, b, round(float(dd), 2), round(float(v), 2)) for a, b, dd, v in zip(P.a[ok], P.b[ok], P.d[ok], f[ok])])
    ev_vr = {r: feats(A_ev[r], 23)["vent_resp"] for r in EV8}
    ev_vent = {r: round(float(np.nanmax(A_ev[r]["act_vent"])), 1) for r in EV8}
    print("   eval 8 vent_resp (full day, causal ffill):", {k: (None if not np.isfinite(v) else round(v, 3)) for k, v in ev_vr.items()})
    print("   eval 8 max act_vent:", ev_vent)
    # ---------------- (C) eval 8 descriptive ----------------
    print("\n(C) eval 8: causal AHx vs training high-pred distribution; reference M_AHxc estimate (not adopted)")
    ev_ahx = {r: ahx_causal_daymean(A_ev[r]) for r in EV8}
    S22 = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", "submission_22.csv"))
    S22["rec"] = S22.row_id.str[:7]; s22 = S22.groupby("rec").sub_ec.mean()
    res = yday[TGT].values - qavg[TGT].values; bc, ac = fit_lin(AHXc[TGT].values, res)
    print(f"   training targets AHxc: quantiles 10/50/90 = {np.nanquantile(AHXc[TGT],.1):.2f}/{np.nanquantile(AHXc[TGT],.5):.2f}/{np.nanquantile(AHXc[TGT],.9):.2f}; true-high mean {AHXc[[r for r in TGT if yday[r]>=1]].mean():.2f}, false-high mean {AHXc[[r for r in TGT if yday[r]<1]].mean():.2f}")
    print(f"   coef (all 40, seed avg): resid = {ac:+.3f} + {bc:+.3f}*AHxc  | mean resid {res.mean():+.3f}")
    for r in EV8:
        print(f"   {r}: AHxc {ev_ahx[r]:.2f} | 13th day-mean {s22[r]:.3f} -> ref {s22[r] + ac + bc*ev_ahx[r]:.3f} (icpt-only {s22[r]+res.mean():.3f})")
