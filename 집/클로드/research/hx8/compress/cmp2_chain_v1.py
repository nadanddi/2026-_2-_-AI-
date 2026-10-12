# -*- coding: utf-8 -*-
"""CMP2 '같은 동 앞 기록 사슬'로 경과일 대리 → 고EC 크기 보정 (2026-10-13 집 클로드, hx8/compress).

[사전 고정 — 실행 전에 적음]
대상(학습 가짜 평가 사례): DIAG10 저장 OOF(CUR) 하루 평균 pm >= .9 인 학습 기록(예측 고EC; 실제 보통도 포함 = 헛경보 비용 포함).
같은 동 전날 = 진단용 연결 truth.pkl a1→b 중 c1 <= 2.0 (비용 큰 4개 연결 6.15/4.82/3.92/3.83 등 9일 건너뛰기 제외).
경과일 대리(모두 '전날부터' 센다, 자기 자신 제외):
  E_true   : 정답 하루 평균 >= 1 인 전날이 몇 개 연속인가 (오라클·상한)
  C_pred   : OOF 예측 하루 평균 >= .9 인 전날이 몇 개 연속인가 (정답 0 사용 — 평가의 '같은 ID 앞 번호 예측 사슬' 흉내)
  C_hide_k : 가까운 k−1 일은 숨김(예측 깃발), k 일 전부터는 정답 깃발 (평가 F13_221/223/225 = k 2/3/4, 끝 = F13_217 정답)
보정 모형(예측 고EC 기록만, 하루 수준 덧셈, 계수는 '고EC 에피소드' 단위로 빼고 맞춤 = 사례 단위 교차적합):
  M0 기준 pm | SH pm + a (닫힌 길 HG3/U 계열 기준선) | CE(X) pm + a + b*min(X,6) | KH 2단계: (1−w) pm + w * 고EC 학습끼리 입력 최근접 5 평균
판정(사전): CE(C_pred) 가 SH 보다 하루 수준 RMSE 가 시드 3개 × {sg, nosg} 모두 낮고, 에피소드 부트스트랩 P(SH 보다 나쁨) < .025/4(변형 4개 본페로니),
  그리고 M0 보다도 모두 낮으면 '유망'. 아니면 약함/종료. C_hide_k 는 같은 기준의 보조. E_true 는 상한 참고(판정 아님).
평가 8기록 추정: 같은 ID 번호 −2 사슬(순수 창, 6.549), 지워진 번호는 이어진 것으로 간주(GR2 6.676), 학습 기록 만나면 그 정답·E_true 로 이음.
평가 예측값(13회차)은 사슬 깃발에만 쓰고 무엇도 맞추지 않음. 계수는 전부 학습 OOF 에서.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cm_common import *

TAU = 0.9; CAP = 6; HIT = 1.0
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

pa = {int(b): int(a) for a, b, c in zip(T.a1, T.b, T.c1) if c <= 2.0}
sgP, _ = oof("P2LOO"); sgE, _ = oof("EL1")


def walk(i, flag, maxk=30):
    k, x = 0, i
    while k < maxk:
        p = pa.get(x)
        if p is None or not TRAIN[p]: return k
        f = flag(p, k + 1)
        if f is None or not f: return k
        k += 1; x = p
    return k


def build(col, seed):
    sg, fold = oof("DIAG10", col, seed)
    alt = {**{k: v for k, v in oof("EL1", col, seed)[0].items()}, **{k: v for k, v in oof("P2LOO", col, seed)[0].items()}}
    pmall = {}
    for k in r.rid[TRAIN]:
        v = sg.get(k, alt.get(k))
        if v is not None: pmall[rid2i[k]] = float(np.mean(v))
    rows = []
    for k, v in sg.items():
        i = rid2i[k]
        if not TRAIN[i]: continue
        pm = float(np.mean(v))
        d = dict(i=i, rid=k, y=ydm[i], pm=pm, fold=fold[k], hourly=np.asarray(v), yh=Ytr[i])
        d["E_true"] = walk(i, lambda p, s: ydm[p] >= HIT)
        d["C_pred"] = walk(i, lambda p, s: (pmall[p] >= TAU) if p in pmall else None)
        for kk in (2, 3, 4):
            d[f"C_hide{kk}"] = walk(i, lambda p, s, kk=kk: ((pmall[p] >= TAU) if p in pmall else None) if s < kk else (ydm[p] >= HIT))
        rows.append(d)
    Dd = pd.DataFrame(rows)
    return Dd, pmall


# 고EC 에피소드(교차적합 단위): 예측 고EC 또는 실제 고EC 기록들을 연결로 묶은 덩어리
def groups(Dd):
    S = set(Dd.i[(Dd.pm >= TAU) | (Dd.y >= HIT)])
    par = {i: i for i in Dd.i}
    def fd(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for b, a in pa.items():
        if b in S and a in S and a in par and b in par: par[fd(b)] = fd(a)
    return np.array([fd(i) for i in Dd.i])


# 입력 공간 S2(시간 벡터) — 2단계 KH 용
trn = np.where(TRAIN)[0]
mu = np.nanmean(A[trn].reshape(-1, 14), 0); sdv = np.nanstd(A[trn].reshape(-1, 14), 0) + 1e-9
S2 = np.nan_to_num(((A - mu) / sdv).reshape(N, -1))


def fit_eval(Dd, xcol, g):
    """예측 고EC 기록만, 에피소드 빼고 맞춤. 반환: 보정 예측 배열(하루 수준), 그리고 시간별 인과판(0..h 누적 예측 평균 >= TAU 일 때만 더함)."""
    m = (Dd.pm >= TAU).values
    sub = Dd[m].reset_index(drop=True); gg = g[m]
    pred = np.zeros(len(sub)); hourly = []
    for j in range(len(sub)):
        o = gg != gg[j]
        res = (sub.y - sub.pm).values[o]
        if xcol == "SH":
            a = res.mean(); dlt = a
        elif xcol == "KH":
            # 고EC 학습끼리(다른 에피소드, 실제 고EC) 최근접 5
            hi_o = sub[o & (sub.y >= HIT).values]
            # 학습 전체 고EC(예측 고EC 집합 밖 포함) 후보 중 같은 에피소드 제외
            gmap = dict(zip(Dd.i, g))
            cand = [ii for ii in Dd.i[(Dd.y >= HIT).values] if gmap[ii] != gg[j]]
            dist = ((S2[cand] - S2[sub.i[j]]) ** 2).sum(1); nb = np.array(cand)[np.argsort(dist)[:5]]
            knn_j = ydm[nb].mean()
            # w: 다른 에피소드 예측 고EC 기록에서 같은 계산으로 최소제곱
            zs, ts = [], []
            for jj in np.where(o)[0]:
                c2 = [ii for ii in cand if gmap[ii] != gg[jj]]
                d2 = ((S2[c2] - S2[sub.i[jj]]) ** 2).sum(1); n2 = np.array(c2)[np.argsort(d2)[:5]]
                zs.append(ydm[n2].mean() - sub.pm[jj]); ts.append(sub.y[jj] - sub.pm[jj])
            zs, ts = np.array(zs), np.array(ts)
            w = float(np.clip((zs * ts).sum() / (zs * zs).sum(), 0, 1))
            dlt = w * (knn_j - sub.pm[j])
        else:
            x = np.minimum(sub[xcol].values, CAP)
            Xm = np.c_[np.ones(o.sum()), x[o]]
            coef = np.linalg.lstsq(Xm, res, rcond=None)[0]
            dlt = coef[0] + coef[1] * x[j]
        pred[j] = sub.pm[j] + dlt
        hp = sub.hourly[j]; cm = np.cumsum(hp) / np.arange(1, 25)
        hourly.append(np.where(cm >= TAU, hp + dlt, hp))
    return sub, gg, pred, np.array(hourly)


def boot_p(e_new, e_ref, gg, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(gg); d = {u: ((e_new[gg == u] ** 2).sum() - (e_ref[gg == u] ** 2).sum()) for u in ug}
    dv = np.array([d[u] for u in ug]); tot = dv.sum()
    bs = np.array([dv[rng.integers(0, len(ug), len(ug))].sum() for _ in range(B)])
    return float((bs >= 0).mean()), tot


MODELS = ["SH", "E_true", "C_pred", "C_hide2", "C_hide3", "C_hide4", "KH"]
summary = []
for col in ["sg", "nosg"]:
    for seed in SEEDS:
        Dd, pmall = build(col, seed); g = groups(Dd)
        base = None
        for mname in ["M0"] + MODELS:
            if mname == "M0":
                sub = Dd[Dd.pm >= TAU].reset_index(drop=True); gg = g[(Dd.pm >= TAU).values]; pred = sub.pm.values
                hourly = np.stack(sub.hourly.values)
            else:
                sub, gg, pred, hourly = fit_eval(Dd, mname, g)
            e = sub.y.values - pred
            eh = np.stack(sub.yh.values) - hourly; ehd = eh.mean(1)
            hi = sub.y.values >= HIT
            rec = dict(col=col, seed=seed, model=mname, n=len(sub), n_hi=int(hi.sum()), rmse=rmse(e), rmse_hi=rmse(e[hi]),
                       rmse_fp=rmse(e[~hi]) if (~hi).any() else np.nan, rmse_causal=rmse(ehd), rmse_hour=rmse(eh))
            if mname == "M0":
                base = e; base_c = ehd
            else:
                rec["impr_vs_M0"] = float(np.mean(np.abs(e) < np.abs(base)))
                rec["P_vs_M0"], rec["dSSE_vs_M0"] = boot_p(e, base, gg)
                if mname != "SH":
                    rec["P_vs_SH"], rec["dSSE_vs_SH"] = boot_p(e, e_sh, gg)
                    rec["impr_vs_SH"] = float(np.mean(np.abs(e) < np.abs(e_sh)))
            if mname == "SH": e_sh = e
            summary.append(rec)
        if col == "sg" and seed == SEEDS[0]:
            P(f"[{col} 시드 {seed}] 예측 고EC 기록 {int((Dd.pm>=TAU).sum())} (실제 고EC {int(((Dd.pm>=TAU)&(Dd.y>=HIT)).sum())}), 에피소드 {len(np.unique(g[(Dd.pm>=TAU).values]))}")
            sub = Dd[Dd.pm >= TAU].copy(); sub["res"] = sub.y - sub.pm
            for xc in ["E_true", "C_pred", "C_hide2", "C_hide3", "C_hide4"]:
                tb = sub.groupby(np.minimum(sub[xc], 3)).res.agg(["count", "mean"]).round(3)
                rho = np.corrcoef(sub[xc].rank(), sub.res.rank())[0, 1]
                P(f"   {xc}: 잔차 평균(0/1/2/3+) {tb.to_dict('index')}, 순위상관 {rho:.2f}, E_true 와 순위상관 {np.corrcoef(sub[xc].rank(), sub.E_true.rank())[0,1]:.2f}")
            P("   기록별(예측 고EC): rid y pm E_true C_pred C_hide2 C_hide3 C_hide4")
            for _, rr in sub.sort_values("rid").iterrows():
                P(f"     {rr.rid} {rr.y:.2f} {rr.pm:.2f} {rr.E_true} {rr.C_pred} {rr.C_hide2} {rr.C_hide3} {rr.C_hide4}")
            # 실제 고EC 인데 예측 < .9 (보정 대상 밖)
            miss = Dd[(Dd.pm < TAU) & (Dd.y >= HIT)]
            P(f"   실제 고EC 인데 예측 < .9: {len(miss)} {list(zip(miss.rid, miss.y.round(2), miss.pm.round(2)))}")

S = pd.DataFrame(summary)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
P("\n=== 예측 고EC 학습 기록, 하루 수준 RMSE (사례 단위 교차적합) ===")
P(S[["col", "seed", "model", "n", "n_hi", "rmse", "rmse_hi", "rmse_fp", "rmse_causal", "impr_vs_M0", "P_vs_M0", "impr_vs_SH", "P_vs_SH", "dSSE_vs_SH"]].round(4).to_string())
P("\n=== 모형별 요약(6칸 평균, SH·M0 보다 나은 칸 수) ===")
for mname in MODELS:
    s = S[S.model == mname]; m0 = S[S.model == "M0"].set_index(["col", "seed"]).rmse; sh = S[S.model == "SH"].set_index(["col", "seed"]).rmse
    ss = s.set_index(["col", "seed"]).rmse
    P(f"{mname:8s} RMSE {ss.mean():.4f} (M0 {m0.mean():.4f}, SH {sh.mean():.4f}) | M0 보다 나은 칸 {(ss < m0).sum()}/6, SH 보다 나은 칸 {(ss < sh).sum()}/6, "
      f"P_vs_SH 최대 {s.P_vs_SH.max() if 'P_vs_SH' in s and s.P_vs_SH.notna().any() else float('nan'):.3f}, 인과판 {s.rmse_causal.mean():.4f}")
S.to_csv(os.path.join(HERE, "cmp2_chain_v1_summary.csv"), index=False)

# ---------------- 평가 8기록 추정 ----------------
P("\n=== 평가 8기록: 같은 ID 번호 −2 사슬 ===")
Dd, pmall = build("sg", None)
g = groups(Dd)
E_tr = {i: walk(i, lambda p, s: ydm[p] >= HIT) for i in range(N) if TRAIN[i]}
def eval_chain(k):
    f, n = k.split("_"); n = int(n)
    steps = []  # (kind, flag)
    x = n
    while True:
        x -= 2
        kk = f"{f}_{x:03d}"
        if kk not in rid2i:
            if f == "F13" and not (216 <= x <= 238): break
            if f == "F47" and not (211 <= x <= 231): break
            steps.append(("del", None)); continue
        i = rid2i[kk]
        if TRAIN[i]:
            steps.append(("train", i)); break
        steps.append(("eval", float(sub22[kk].mean())))
    # C_pred: 평가 예측 깃발만(지워진 번호는 이어짐 간주, 학습 기록은 OOF 예측 깃발)
    c = 0
    for kind, v in steps:
        if kind == "del": c += 1; continue
        if kind == "eval":
            if v >= TAU: c += 1; continue
            break
        if kind == "train":
            if pmall.get(v, 0) >= TAU: c += 1 + walk(v, lambda p, s: (pmall[p] >= TAU) if p in pmall else None)
            break
    # C_hide: 평가·지워진 구간은 예측 깃발, 학습 기록 만나면 정답 깃발 + 그 기록의 E_true
    ch = 0; kdist = 0
    for kind, v in steps:
        kdist += 1
        if kind == "del": ch += 1; continue
        if kind == "eval":
            if v >= TAU: ch += 1; continue
            break
        if kind == "train":
            if ydm[v] >= HIT: ch += 1 + E_tr[v]
            break
    # 사슬이 평가 비고EC 에서 끊겼으면 ch 를 거기서 멈춤 (위 break 처리), 지워진 번호가 사슬 끝이면 되돌림
    while steps and steps[0][0] == "del" and len(steps) == 1:
        break
    return steps, c, ch, kdist
P("기록 | 13회차 하루 평균 | 사슬(−2 단계) | C_pred | C_hide(학습 끝 E 포함) | 학습 끝까지 거리 k")
EV = {}
for k in EVAL8:
    steps, c, ch, kd = eval_chain(k)
    desc = " ← ".join([("지움" if s[0] == "del" else (f"평가 {s[1]:.2f}" if s[0] == "eval" else f"학습 {r.rid[s[1]]} y{ydm[s[1]]:.2f} E{E_tr[s[1]]}")) for s in steps])
    EV[k] = dict(c=c, ch=ch, k=kd)
    P(f"{k} | {sub22[k].mean():.3f} | {desc} | {c} | {ch} | {kd}")

# 전체 학습(예측 고EC)으로 계수 맞춰 평가에 적용
sub = Dd[Dd.pm >= TAU].copy(); res = (sub.y - sub.pm).values
coefs = {}
a_sh = res.mean(); coefs["SH"] = (a_sh, 0)
for xc in ["C_pred", "C_hide2", "C_hide3", "C_hide4", "E_true"]:
    x = np.minimum(sub[xc].values, CAP); cf = np.linalg.lstsq(np.c_[np.ones(len(x)), x], res, rcond=None)[0]; coefs[xc] = tuple(cf)
P("\n계수(전체 학습 예측 고EC, 시드 평균 OOF): " + ", ".join(f"{k} a {v[0]:+.3f} b {v[1]:+.3f}" for k, v in coefs.items()))
P("평가 추정(하루 평균): 기록 | 13회차 | SH | C_pred 판 | C_hide 판(k 에 맞는 계수: k<=2→C_hide2, 3→C_hide3, >=4→C_hide4)")
rows = []
for k in EVAL8:
    p0 = sub22[k].mean(); e = EV[k]
    sh = p0 + coefs["SH"][0]
    cp = p0 + coefs["C_pred"][0] + coefs["C_pred"][1] * min(e["c"], CAP)
    key = "C_hide2" if e["k"] <= 2 else ("C_hide3" if e["k"] == 3 else "C_hide4")
    chv = p0 + coefs[key][0] + coefs[key][1] * min(e["ch"], CAP)
    rows.append(dict(rec=k, sub22=p0, SH=sh, Cpred=cp, Chide=chv, c=e["c"], ch=e["ch"], k=e["k"]))
    P(f"{k} | {p0:.3f} | {sh:.3f} | {cp:.3f} (C {e['c']}) | {chv:.3f} (C {e['ch']}, {key})")
pd.DataFrame(rows).to_csv(os.path.join(HERE, "cmp2_eval8_v1.csv"), index=False)
open(os.path.join(HERE, "cmp2_chain_v1.log"), "w", encoding="utf-8").write("\n".join(out))
