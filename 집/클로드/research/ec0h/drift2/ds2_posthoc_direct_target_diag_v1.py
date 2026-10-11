# -*- coding: utf-8 -*-
"""DS2 사후 진단(채택 후보 아님, DS1 결과 보고 뒤 추가). DS1 (2026-10-12 집 클로드): EC 하루 안 모양 보정기. 계획 = ec0h/drift2/plan_v1.md (실행 전 고정).
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


def gfit(recs, Q, Y, fo, target, shuffle=False):
    n = len(recs); FX = np.stack([feats(bidx[r]) for r in recs]); F47 = np.array([r[:3] == "F47" for r in recs], float)
    full = np.concatenate([FX, Q[:, :, None], np.repeat(Q[:, :1, None], 24, 1), (Q - Q[:, :1])[:, :, None], np.repeat(F47[:, None, None], 24, 1)], 2)
    hs = np.arange(1, 24); G = np.zeros((n, 24)); E = Y - Q
    for k in np.unique(fo):
        tr, te = fo != k, fo == k
        c = np.array([np.nansum(E[tr, 0] * E[tr, h]) / np.nansum(E[tr, 0] ** 2) for h in range(24)])
        T = E - c[None, :] * E[:, :1] if target == "cdecay" else E
        Tt = T[tr][:, hs]
        if shuffle: Tt = Tt[np.random.default_rng(int(k) + 7).permutation(len(Tt))]
        Xtr = full[tr][:, hs].reshape(-1, full.shape[2]); ytr = Tt.reshape(-1); ok = np.isfinite(ytr)
        m = lgb.LGBMRegressor(num_leaves=7, min_child_samples=200, learning_rate=.03, n_estimators=300, colsample_bytree=.8,
                              random_state=20261012, verbose=-1).fit(Xtr[ok], ytr[ok])
        G[np.ix_(np.where(te)[0], hs)] = m.predict(full[te][:, hs].reshape(-1, full.shape[2])).reshape(-1, 23)
        if target == "cdecay": G[te, 0] = 0; G[te] += 0  # c applied outside
    return G
rmse = lambda P, Y: np.sqrt(np.nanmean((P - Y) ** 2))
print("DS2: (a) real state, direct target e_h (h>=1, 0h untouched); (b) oracle state, baseline q + c_h*e0 (c from training folds), target e_h - c_h e0")
for v in ("EV", "DIAG10"):
    for s, (recs, Q, Y, fo) in load(v).items():
        E = Y - Q; out = []
        for sh in (False, True):
            Gd = gfit(recs, Q, Y, fo, "direct", sh); Gc = gfit(recs, Q, Y, fo, "cdecay", sh)
            QC = Q.copy()
            for k in np.unique(fo):
                tr, te = fo != k, fo == k
                c = np.array([np.nansum(E[tr, 0] * E[tr, h]) / np.nansum(E[tr, 0] ** 2) for h in range(24)])
                QC[te] = Q[te] + c[None, :] * E[te, :1]
            out.append((sh, [rmse(Q + sv * Gd, Y) for sv in (0, .25, .5, 1)], [rmse(QC + sv * Gc, Y) for sv in (0, .25, .5, 1)]))
        for sh, a, b in out:
            print(f"  {v:6s} seed {s:>5} {'SHUFFLE' if sh else 'real   '}: real-state s=0/.25/.5/1 " + " ".join(f"{x:.5f}" for x in a)
                  + " | oracle(c-decay base) s=0/.25/.5/1 " + " ".join(f"{x:.5f}" for x in b), flush=True)
