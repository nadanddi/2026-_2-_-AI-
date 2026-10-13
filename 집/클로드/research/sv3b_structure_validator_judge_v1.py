# -*- coding: utf-8 -*-
"""SV3b (2026-10-14 집 클로드; 사용자: '정말 별로라서 리젝인지, 좋은데도 리젝인지 구분 — 진행').  사전 고정(커밋 후 실행).  EC.
SV3a 배치 9개(2차 정답 구간마다 하나; 양끝 del·가운데 val, 나머지 2차는 보임, 잠금 안 보임) x 실제 파이프라인 3시드의 val 예측(fin = SG2 후, 앵커·B5 전).
후보(모두 fin 위에, 2차 val 행만; EA2 꼴 P_h = fin_h + w(A − cm_h), cm_h = fin 0..h 누적평균, 학습 범위로 자름):
  [대조] DOSEw = 진짜 전날(cert)이 그 배치에서 보이면 A = 그 23시 EC, w = .1 .2 .3 1.0  — 크기를 아는 개선(검출 한계 측정용, 실제로는 못 씀).
  [대조] NULL = DOSE1.0 과 같은 기록에 A 를 같은 층(prev_vis) 기록끼리 무작위로 뒤섞음, 300 회 — 효과 없는 가짜(오탐률 측정용).
  [내 모델 부품] EA2 = v7 확신 짝(cv_all400_choice_v7m: p >= .8, 정답 짝, dT <= 1.5)의 짝이 그 배치에서 보이면 A = 짝 23시 EC (w 1).
               B5 = 그 시각 cm_h >= 1.0 이고 0..h 입력이 똑같은 보이는 정답 쌍둥이 중 하루 평균 >= 1.2 가 있으면 +δ, δ = −.29(B5 6.78x 의 기존 값, 새로 맞추지 않음).
               EA2+B5 = 둘 다(B5 는 EA2 결과 위에).
  [지난 후보 사후 재분류] LKH = 시각별 지문 링커(날짜 d−1 후보, 0..h 지문, 확신 2등/1등 >= 1.5 & 거리 <= tau_h; tau_h 는 그 배치 val 을 뺀 cert 쌍으로) 짝의 23시 EC (LH3 P2 와 같은 규칙).
층(후보와 무관): V = 진짜 전날 보임, NV = 안 보임.  단위 = 배치(구간) 9개, 배치별 dSSE(시드 평균).
검정: 배치 부호 뒤집기 정확 순열(2^9) — P_better = P(합 <= 관측), P_worse = P(합 >= 관측)(변화 없는 칸은 '변화 없음');  효과 % = RMSE 변화, 95% 구간 = 배치 재표집 부트스트랩 5,000.
네 판정(미리 정함, 각 후보의 2차 전체와 층 V·NV):
  채택 = 2차 전체 P_better < .025 AND 어느 층도 P_worse < .025 아님;  해로움 = 2차 전체 또는 어느 층 P_worse < .025;
  효과 없음 = 위 둘 아님 AND 2차 전체 95% 구간이 (−MDE, +MDE) 안(MDE = DOSE 중 P_better < .025 가 되는 가장 작은 효과의 크기);  판정 불가 = 나머지.
오탐률 = NULL 300 회 중 '채택' 비율(.025 이하여야 검증기 규칙이 정상).
"""
import os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, X, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
XR = np.round(X, 1); SEEDS = (2323, 4545, 6767); rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
G = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(env.LOCAL, "sv3_ckpt", "draw_*.csv"))], ignore_index=True)
G = G[G.day >= 179].sort_values(["draw", "rec", "hour"]).reset_index(drop=True)
DR = pd.read_csv(os.path.join(HERE, "sv3_draws_v1.csv")); ST = pd.read_csv(os.path.join(HERE, "sv3_strata_v1.csv"))
G = G.merge(ST, on=["draw", "rec"], how="left"); lo_, hi_ = float(np.nanmin(EC[TRAIN])), float(np.nanmax(EC[TRAIN]))
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {rid(int(b)): rid(int(a)) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c}
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
import json
LOCKF = os.path.join(env.ROOT, u"집", u"코덱스", "analysis", "codex_independent", "ec_final_lock", "locked_days.json")
LOCKREC = {f"{s['farm']}_{int(s['day']):03d}" for s in json.load(open(LOCKF, encoding="utf-8"))["selected"]}
LABREC = {rid(i) for i in range(N) if TRAIN[i] and np.isfinite(DM[i])} - LOCKREC
def visible(k): return LABREC - set(DR[DR.draw == k].rec)
# LKH machinery (as SV1, tau recomputed per draw without that draw's val)
SV1 = open(os.path.join(HERE, "sv1_structure_validator_v1.py"), encoding="utf-8").read(); ns = {"__file__": os.path.join(HERE, "sv1_structure_validator_v1.py"), "__name__": "sv1"}
exec(SV1[:SV1.index("TAU = np.nanquantile")], ns)
dist, tdate, DAYT, SEGT, td_pairs = ns["dist"], ns["tdate"], ns["DAYT"], ns["SEGT"], None
CERTI = {idx[b]: idx[a] for b, a in CERT.items()}
def tau_without(valset):
    rows = []
    for b, a in CERTI.items():
        if rid(b) in valset or rid(b) in LOCKREC or rid(a) in LOCKREC: continue
        v = np.array([rid(i) in LABREC for i in range(N)]); v[b] = False; t = tdate(b, v)
        if t is not None and DAYT[a] == t[0] - 1 and SEGT[a] == t[1]: rows.append([dist(a, b, h) for h in range(24)])
    return np.nanquantile(np.array(rows), .9, axis=0)
def lkh_picks(b, vis, tau):
    t = tdate(b, vis)
    if t is None: return [None] * 24
    cand = [a for a in np.where(vis)[0] if a != b and DAYT[a] == t[0] - 1 and SEGT[a] == t[1]]
    out = []
    for h in range(24):
        if not cand: out.append(None); continue
        d = sorted((dist(a, b, h), a) for a in cand); ok = (len(d) == 1 or d[1][0] >= 1.5 * max(d[0][0], 1e-9)) and d[0][0] <= tau[h]
        out.append(d[0][1] if ok else None)
    return out
def twin_high(b, h, vis):
    c = np.where(vis)[0]; c = c[c != b]; t = c[np.all(np.abs(XR[c, :h + 1] - XR[b, :h + 1]) < 1e-6, axis=(1, 2))]; t = t[np.isfinite(DM[t])]
    return bool((DM[t] >= 1.2).any())
# ---- build candidate predictions (per seed) ----
cands = {}
A_true, A_v7, A_lkh, B5m = {}, {}, {}, {}
for k, g in G.groupby("draw"):
    vs = visible(k); vis = np.array([rid(i) in vs for i in range(N)]); tau = tau_without(set(DR[(DR.draw == k) & (DR.role == "val")].rec))
    for r, gr in g.groupby("rec"):
        b = idx[r]
        if r in CERT and CERT[r] in vs: A_true[(k, r)] = float(EC[idx[CERT[r]], 23])
        if r in V7 and V7[r] in vs: A_v7[(k, r)] = float(EC[idx[V7[r]], 23])
        A_lkh[(k, r)] = [(float(EC[a, 23]) if a is not None else None) for a in lkh_picks(b, vis, tau)]
        for s in SEEDS:
            cm = gr[f"fin_{s}"].expanding().mean().values
            B5m[(k, r, s)] = [cm[j] >= 1.0 and twin_high(b, int(h), vis) for j, h in enumerate(gr.hour.values)]
    print("draw", k, "prepared", flush=True)
def apply(kind, w=1.0, Amap=None):
    out = {}
    for s in SEEDS:
        P = G[f"fin_{s}"].values.copy()
        for (k, r), gr in G.groupby(["draw", "rec"]):
            ix = gr.index.values; f = gr[f"fin_{s}"].values; cm = pd.Series(f).expanding().mean().values; hh = gr.hour.values
            if kind == "DOSE" and (k, r) in (Amap or A_true): P[ix] = f + w * ((Amap or A_true)[(k, r)] - cm)
            elif kind in ("EA2", "EA2B5") and (k, r) in A_v7: P[ix] = f + (A_v7[(k, r)] - cm)
            elif kind == "LKH":
                a = A_lkh[(k, r)]; P[ix] = np.array([f[j] + (a[h] - cm[j]) if a[h] is not None else f[j] for j, h in enumerate(hh)])
            if kind in ("B5", "EA2B5"):
                m = np.array(B5m[(k, r, s)]); P[ix] = np.where(m, P[ix] - .29, P[ix])
        out[s] = np.clip(P, lo_, hi_)
    return out
y = G.sub_ec.values; p2rows = np.ones(len(G), bool)
def draw_dsse(Pd, mask):
    d = np.mean([(Pd[s] - y) ** 2 - (G[f"fin_{s}"].values - y) ** 2 for s in SEEDS], axis=0)
    return pd.Series(d[mask]).groupby(G.draw.values[mask]).sum().reindex(sorted(G.draw.unique())).fillna(0).values
FLIPS = np.array([[1 if (m >> j) & 1 else -1 for j in range(9)] for m in range(2 ** 9)])
def perm(v):
    if np.all(np.abs(v) < 1e-12): return np.nan, np.nan
    sims = FLIPS[:, :len(v)] @ np.abs(v) if False else (FLIPS[:, :len(v)] * np.abs(v)).sum(1); obs = v.sum()
    return float((sims <= obs + 1e-12).mean()), float((sims >= obs - 1e-12).mean())
rng = np.random.default_rng(20261209)
def pct_ci(Pd, mask):
    base = np.mean([(G[f"fin_{s}"].values - y) ** 2 for s in SEEDS], axis=0)[mask]; cand = np.mean([(Pd[s] - y) ** 2 for s in SEEDS], axis=0)[mask]; dr = G.draw.values[mask]
    ks = np.unique(dr); bs = pd.Series(base).groupby(dr).sum(); cs = pd.Series(cand).groupby(dr).sum()
    pt = 100 * (np.sqrt(cs.sum() / bs.sum()) - 1); boots = []
    for _ in range(5000):
        s = rng.choice(ks, len(ks)); boots.append(100 * (np.sqrt(cs[s].sum() / bs[s].sum()) - 1))
    return pt, np.percentile(boots, [2.5, 97.5])
strata = {"ALL": np.ones(len(G), bool), "V": G.prev_vis.astype(bool).values, "NV": ~G.prev_vis.astype(bool).values}
def evaluate(name, Pd, ci_on=True):
    res = {}
    for sn, m in strata.items():
        v = draw_dsse(Pd, m); pb, pw = perm(v); pt, ci = pct_ci(Pd, m) if ci_on else (np.nan, (np.nan, np.nan))
        res[sn] = dict(pct=pt, lo=ci[0], hi=ci[1], P_better=pb, P_worse=pw, changed=int((np.abs(v) > 1e-12).sum()))
    return res
RES = {}
for w in (.1, .2, .3, 1.0): RES[f"DOSE{w}"] = evaluate(f"DOSE{w}", apply("DOSE", w))
mde = min([abs(RES[f"DOSE{w}"]["ALL"]["pct"]) for w in (.1, .2, .3, 1.0) if RES[f"DOSE{w}"]["ALL"]["P_better"] < .025] or [np.nan])
print(f"\nMDE (smallest dose effect with P_better < .025): {mde:.2f}%", flush=True)
recs_true = list(A_true.keys()); vals = np.array([A_true[k_] for k_ in recs_true]); fa = 0
for t in range(300):
    perm_map = dict(zip(recs_true, rng.permutation(vals))); r_ = evaluate("NULL", apply("DOSE", 1.0, perm_map), ci_on=False)
    if r_["ALL"]["P_better"] < .025 and not any((r_[s]["P_worse"] or 1) < .025 for s in r_ if not np.isnan(r_[s]["P_worse"])): fa += 1
print(f"NULL false adoption rate: {fa}/300 = {fa / 300:.3f}", flush=True)
for nm in ("EA2", "B5", "EA2B5", "LKH"): RES[nm] = evaluate(nm, apply(nm))
def verdict(r):
    if not np.isnan(r["ALL"]["P_better"]) and r["ALL"]["P_better"] < .025 and not any((not np.isnan(r[s]["P_worse"])) and r[s]["P_worse"] < .025 for s in r): return "채택"
    if any((not np.isnan(r[s]["P_worse"])) and r[s]["P_worse"] < .025 for s in r): return "해로움"
    if r["ALL"]["changed"] == 0: return "변화 없음"
    if np.isfinite(mde) and -mde < r["ALL"]["lo"] and r["ALL"]["hi"] < mde: return "효과 없음"
    return "판정 불가"
rows = []
for nm, r in RES.items():
    for sn in ("ALL", "V", "NV"): rows.append(dict(cand=nm, stratum=sn, **{k_: round(v_, 4) if isinstance(v_, float) else v_ for k_, v_ in r[sn].items()}))
T = pd.DataFrame(rows); pd.set_option("display.width", 220); print(T.to_string(index=False))
print("\nverdicts:", {nm: verdict(r) for nm, r in RES.items()})
print("records: pass-2 val", G.rec.nunique(), "| V", int(G.drop_duplicates('rec').prev_vis.sum()), "| clusters", G.draw.nunique(), "| EA2 fires", len(A_v7), "| true-prev visible", len(A_true))
T.to_csv(os.path.join(HERE, "sv3b_results_v1.csv"), index=False)
