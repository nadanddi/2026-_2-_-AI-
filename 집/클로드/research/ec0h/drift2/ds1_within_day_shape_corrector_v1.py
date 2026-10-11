# -*- coding: utf-8 -*-
"""DS1 (2026-10-12 집 클로드): EC 하루 안 모양 보정기. 계획 = ec0h/drift2/plan_v1.md (실행 전 고정).
대상 r_h = (y_h-q_h)-(y_0-q_0), h>=1; 특징 = 같은 기록 0..h 입력 + q 모양. 0시는 그대로.
보정기는 같은 검증기·같은 시드의 다른 폴드 기록으로만 학습. 실제 상태 / 0시 오라클 상태 둘 다 보고."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np, pandas as pd, pickle
import lightgbm as lgb
from sklearn.linear_model import Ridge
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
OUTD = os.path.join(H_, "ec0h", "drift2")
B = pickle.load(open(os.path.join(H_, "local", "linker_v3", "base.pkl"), "rb"))
br = B["recs"].reset_index(drop=True); bidx = {f"{f}_{d:03d}": i for i, (f, d) in enumerate(zip(br.farm, br.day))}
OUT, IN_, AC = B["OUT"], B["IN"].copy(), B["ACT"]
for i in range(len(IN_)):   # causal fill within record (ffill, then first valid backward only at start -> use 0-diff)
    df = pd.DataFrame(IN_[i]).ffill().bfill(); IN_[i] = df.values

def feats(bi):
    """(24, nf) features; row h uses inputs 0..h only."""
    o, x, a = OUT[bi], IN_[bi], AC[bi]; hh = np.arange(24); n1 = hh + 1
    cm = lambda v: np.cumsum(v) / n1
    F = [hh, cm(a[:, 0]), np.cumsum(a[:, 0]), a[:, 0], cm(a[:, 1]), a[:, 1], cm(a[:, 2]), a[:, 2], cm(a[:, 3]), cm(a[:, 4]),
         np.cumsum(o[:, 2]), o[:, 2], o[:, 0], x[:, 0] - x[0, 0], x[:, 1] - x[0, 1], x[:, 2], cm(x[:, 2]), x[:, 0]]
    return np.stack(F, 1)
FN = ["h", "vent_m", "vent_cum", "vent_h", "shade_m", "shade_h", "therm_m", "therm_h", "heat_m", "fan_m", "rad_cum", "rad_h",
      "otemp_h", "dtin", "dhum", "co2_h", "co2_m", "tin_h"]

def load(v):
    if v == "EV":
        R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv"))
        R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")].copy(); R["q"] = R.EA2
    else:
        R = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv"))
        R = R[(R.validator == "DIAG10") & (R.tag == "CUR")].copy(); R["q"] = R.sg; R["y"] = R.sub_ec; R["rec"] = R.row_id.str[:7]
    R["seed"] = R.seed.astype(str); out = {}
    for s, G in R.groupby("seed"):
        Q = G.pivot(index="rec", columns="hour", values="q"); Y = G.pivot(index="rec", columns="hour", values="y").reindex(Q.index)
        fo = G.groupby("rec").fold.first().reindex(Q.index).values
        out[s] = (Q.index.values, Q.values, Y.values, fo)
    return out

def model_g(arm, recs, Q, Y, fo, tau_lv=None):
    """returns G (n,24) out-of-fold correction (s=1); G[:,0]=0. For M3 returns dict tau_lv -> G."""
    n = len(recs); FX = np.stack([feats(bidx[r]) for r in recs]); F47 = np.array([r[:3] == "F47" for r in recs], float)
    qsh = Q - Q[:, :1]; RT = (Y - Q) - (Y[:, :1] - Q[:, :1])
    full = np.concatenate([FX, Q[:, :, None], np.repeat(Q[:, :1, None], 24, 1), qsh[:, :, None], np.repeat(F47[:, None, None], 24, 1)], 2)
    hs = np.arange(1, 24)
    if arm == "M3":
        res = {lv: np.zeros((n, 24)) for lv in (.5, .7, .8, .9)}
    else:
        G = np.zeros((n, 24))
    for k in np.unique(fo):
        tr, te = fo != k, fo == k
        if arm == "M3":
            for lv in res:
                t = np.quantile(Q[tr, 0], lv); sel = tr & (Q[:, 0] >= t)
                b = np.nanmean(RT[sel], 0); b[:12] = 0
                gate = Q[te, 0] >= t; res[lv][np.where(te)[0][gate]] = b
            continue
        RTt = RT[tr][:, hs]
        if arm == "M2S": RTt = RTt[np.random.default_rng(int(k) + 7).permutation(len(RTt))]   # shuffle control: targets of other records
        Xtr = full[tr][:, hs].reshape(-1, full.shape[2]); ytr = RTt.reshape(-1)
        Xte = full[te][:, hs].reshape(-1, full.shape[2])
        ok = np.isfinite(ytr)
        if arm == "M1":
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            def blk(X):
                Z = (X - mu) / sd; h = X[:, 0]; bl = np.digitize(h, [6, 12, 18])
                return np.concatenate([Z * (bl == j)[:, None] for j in range(4)] + [(bl == j)[:, None].astype(float) for j in range(4)], 1)
            m = Ridge(alpha=30.0).fit(blk(Xtr)[ok], ytr[ok]); p = m.predict(blk(Xte))
        else:
            m = lgb.LGBMRegressor(num_leaves=7, min_child_samples=200, learning_rate=.03, n_estimators=300, colsample_bytree=.8,
                                  subsample=1.0, random_state=20261012, verbose=-1).fit(Xtr[ok], ytr[ok])
            p = m.predict(Xte)
        G[np.ix_(np.where(te)[0], hs)] = p.reshape(-1, 23)
    return res if arm == "M3" else G

rmse = lambda P, Y, m=None: np.sqrt(np.nanmean(((P - Y) if m is None else (P - Y)[m]) ** 2))
SV = [.25, .5, .75, 1.0]
DATA = {v: load(v) for v in ("EV", "DIAG10")}
GG = {}
for v, D in DATA.items():
    for s, (recs, Q, Y, fo) in D.items():
        for arm in ("M1", "M2", "M3", "M2S"):
            GG[(v, s, arm)] = model_g(arm, recs, Q, Y, fo)
        print("done", v, s, flush=True)
p1 = lambda recs: np.array([int(r[4:7]) < 179 for r in recs])

# ---- selection on EV pass-1, real state, seed-mean RMSE ----
choice = {}
print("\n[selection] EV pass-1 (num<179), real state, mean over 3 seeds of RMSE")
for arm in ("M1", "M2", "M3", "M2S"):
    opts = []
    lvs = (.5, .7, .8, .9) if arm == "M3" else (None,)
    for lv in lvs:
        for sv in SV:
            r = []
            for s, (recs, Q, Y, fo) in DATA["EV"].items():
                g = GG[("EV", s, arm)]; g = g[lv] if arm == "M3" else g; m = p1(recs)
                r.append(rmse(Q[m] + sv * g[m], Y[m]))
            base = np.mean([rmse(Q[p1(rc)], Y[p1(rc)]) for rc, Q, Y, _ in DATA["EV"].values()])
            opts.append((np.mean(r), lv, sv)); print(f"  {arm} tau_lv {lv} s {sv}: {np.mean(r):.5f} (base {base:.5f})")
    b = min(opts); choice[arm] = (b[1], b[2]); print(f"  -> {arm} chosen tau_lv {b[1]} s {b[2]}")

# ---- judgement ----
rng = np.random.default_rng(20261012)
rows = []
print("\n[judgement] full RMSE (all 24h cells), real state and 0h-oracle state")
for arm in ("M1", "M2", "M3", "M2S"):
    lv, sv = choice[arm]; recd = {}; recs_seed = {}
    for v, D in DATA.items():
        for s, (recs, Q, Y, fo) in D.items():
            g = GG[(v, s, arm)]; g = g[lv] if arm == "M3" else g
            P = Q + sv * g; e0 = (Y[:, :1] - Q[:, :1]); QO = Q + e0; PO = QO + sv * g
            m2 = ~p1(recs); big = np.abs(Y[:, 23] - Y[:, 0]) >= .15
            E1 = Y[p1(recs)] - Q[p1(recs)]; c = np.array([np.nansum(E1[:, 0] * E1[:, h]) / np.nansum(E1[:, 0] ** 2) for h in range(24)])
            QC = Q + c[None, :] * e0
            rec_d = np.nansum((P - Y) ** 2, 1) - np.nansum((Q - Y) ** 2, 1)
            for r, d in zip(recs, rec_d): recd.setdefault(v, {}).setdefault(r, []).append(d)
            recs_seed[(v, s)] = rec_d
            row = dict(arm=arm, val=v, seed=s, base=rmse(Q, Y), new=rmse(P, Y), base_p2=rmse(Q, Y, m2), new_p2=rmse(P, Y, m2),
                       orc_base=rmse(QO, Y), orc_new=rmse(PO, Y), orc_cdecay=rmse(QC, Y),
                       orc_base_p2=rmse(QO, Y, m2), orc_new_p2=rmse(PO, Y, m2),
                       big_n=int(big.sum()), base_big=rmse(Q, Y, big), new_big=rmse(P, Y, big),
                       h1_23_base=rmse(Q[:, 1:], Y[:, 1:]), h1_23_new=rmse(P[:, 1:], Y[:, 1:]),
                       pm_12_23_base=rmse(Q[:, 12:], Y[:, 12:]), pm_12_23_new=rmse(P[:, 12:], Y[:, 12:]))
            rows.append(row)
            print(f"  {arm} {v:6s} seed {s:>5}: real {row['base']:.5f}->{row['new']:.5f} ({row['new']-row['base']:+.5f}) | p2 {row['new_p2']-row['base_p2']:+.5f}"
                  f" | oracle0h {row['orc_base']:.5f}->{row['orc_new']:.5f} ({row['orc_new']-row['orc_base']:+.5f}; c-decay ref {row['orc_cdecay']:.5f})"
                  f" | big-change days n{row['big_n']} {row['base_big']:.3f}->{row['new_big']:.3f} | 12-23h {row['pm_12_23_base']:.4f}->{row['pm_12_23_new']:.4f}")
    for v in ("DIAG10", "EV"):
        dd = pd.Series({r: np.mean(x) for r, x in recd[v].items()}); a = dd.values
        bs = np.array([a[rng.integers(0, len(a), len(a))].sum() for _ in range(10000)]); pw = (bs >= 0).mean()
        ch = pd.Series(a, index=dd.index).groupby([f"{r[:3]}{int(r[4:7])//10}" for r in dd.index]).sum().values
        bc = np.array([ch[rng.integers(0, len(ch), len(ch))].sum() for _ in range(10000)]); pc = (bc >= 0).mean()
        ps = [(np.array([recs_seed[(v, s_)][rng.integers(0, len(a), len(a))].sum() for _ in range(2000)]) >= 0).mean() for s_ in DATA[v]]
        top5 = np.sort(-a)[::-1][:5].sum() / (-a.sum()) if a.sum() < 0 else np.nan
        print(f"      per-seed P(worse) {', '.join(f'{x:.3f}' for x in ps)} | top-5 record share {top5:.2f}")
        imp = -a.sum(); share = (-a).max() / imp if imp > 0 else np.nan
        print(f"  {arm} {v} bootstrap: sum dSSE {a.sum():+.3f} | P(worse) record {pw:.4f} | chunk {pc:.4f} (n chunks {len(ch)}) | max record share {share:.2f} ({dd.idxmin()})")
        rows.append(dict(arm=arm, val=v, seed="boot", pworse=pw, pworse_chunk=pc, maxshare=share, sum_dsse=a.sum()))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUTD, "ds1_results_v1.csv"), index=False)
print("\n[verdict] (alpha bonferroni .025/3 = .00833)")
for arm in ("M1", "M2", "M3"):
    t = T[(T.arm == arm) & (T.seed != "boot")]; b = T[(T.arm == arm) & (T.seed == "boot") & (T.val == "DIAG10")].iloc[0]
    c1 = bool((t.new < t.base).all()); c2 = b.pworse < .025 / 3; c3 = b.maxshare <= .5
    print(f"  {arm} choice {choice[arm]}: all 6 seed-validator improve {c1} ({int((t.new<t.base).sum())}/6) | DIAG10 P {b.pworse:.4f} {c2} | max share {b.maxshare:.2f} {c3} -> {'ADOPT' if c1 and c2 and c3 else 'REJECT'}")
