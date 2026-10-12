# -*- coding: utf-8 -*-
# hx8/forensics fx3 v1: fx2(복제 검사 0건)의 신뢰성 확인 + 범위 확장. 서술(모델 없음).
# 실행 전 고정한 가설 표
# | 가설 | 맞다면 반드시 보일 흔적 | 맞다면 보이면 안 되는 것 | 확인 | 합격 기준 |
# | H1a 8기록 실내 입력이 다른 온실(51개) 학습 기록의 복제(시간 이동·잡음 포함) |
#     실내 3변수 24h(겹침≥18h, 이동 −6..+6) 최소상관 ≥ .99 상대가 학습에 있음 |
#     8기록의 최대 상관이 학습 F13/F47 기록끼리의 최대 상관 분포(같은 '날짜 쌍둥이 있음/없음' 층) 안쪽 |
#     전 학습 12,932기록 대상 벡터 상관 스캔 | 8기록 중 ≥1개가 최대상관 ≥ .99 이고 기준 분포 99% 분위 초과 |
# | C0 (검출기 점검) fx2 정확 일치 검사가 복사를 실제로 잡는가 | 심은 6시간 복사(시각 이동)를 L=3/4/6 모두 검출 |
#     심은 복사를 못 잡음 | F13_221 사본에 학습 기록 6시간 창을 심어 같은 함수로 검사 | 심은 위치를 정확히 찾음 |
# | H1b F47_225↔F13_232(.978)·F47_227↔F13_233(.949)는 복사가 아니라 같은 날짜 다른 동의 날씨 공유 |
#     같은 날짜(외기 24h 동일) 학습 다른 동 쌍의 0이동 최소상관 분포 안에 듦 | 분포 최댓값을 크게 넘음 | 학습 쌍 분포 | 분포 최댓값 이하 |
# 규정: 평가 기록은 자기 입력만, 매칭 상대는 학습 기록만(평가-평가 비교 없음).
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
from collections import defaultdict

OUT = os.path.dirname(os.path.abspath(__file__))
T8 = ["F13_221", "F13_223", "F13_225", "F13_237", "F47_221", "F47_223", "F47_225", "F47_227"]
cols = ["row_id", "out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2"]
tr = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=cols)
te = pd.read_csv(os.path.join(env.DATA, "test_X.csv"), usecols=cols)
def arr(d, c):
    d = d.copy(); d["rec"] = d.row_id.str[:7]; d["h"] = d.row_id.str[8:].astype(int)
    recs = d.rec.drop_duplicates().tolist(); ix = {r: k for k, r in enumerate(recs)}
    A = np.full((len(recs), 24, len(c)), np.nan)
    A[d.rec.map(ix).values, d.h.values] = d[c].values
    return recs, A
INC = ["in_temp", "in_hum", "in_co2"]; OUC = ["out_temp", "out_hum", "out_rad", "out_wspd"]
rtr, Atr = arr(tr, INC); rte, Ate = arr(te, INC)
_, Otr = arr(tr, OUC); _, Ote = arr(te, OUC)
rtr = np.array(rtr); rte = np.array(rte)
print("train", len(rtr), "eval", len(rte))

# ---------- C0: positive control of fx2 exact matcher ----------
def keyarr(A): return np.round(A * 10)
Ktr = keyarr(Atr)
rng = np.random.default_rng(0)
ctrl = []
for trial in range(5):
    j = int(rng.integers(len(rtr)))
    while np.isnan(Ktr[j]).any(): j = int(rng.integers(len(rtr)))
    s_src = int(rng.integers(0, 19)); s_dst = int(rng.integers(0, 19))
    probe = keyarr(Ate[list(rte).index("F13_221")].copy())
    probe[s_dst:s_dst + 6] = Ktr[j, s_src:s_src + 6]
    for L in (3, 4, 6):
        idx = defaultdict(list)
        for i in range(len(rtr)):
            for s in range(0, 25 - L):
                w = Ktr[i, s:s + L]
                if np.isnan(w).any(): continue
                idx[np.ascontiguousarray(w).tobytes()].append((i, s))
        found = []
        for s in range(0, 25 - L):
            w = probe[s:s + L]
            if np.isnan(w).any(): continue
            for (jj, s2) in idx.get(np.ascontiguousarray(w).tobytes(), []):
                found.append((s, jj, s2))
        ok = any(jj == j and s - s2 == s_dst - s_src for s, jj, s2 in found)
        ctrl.append(dict(trial=trial, L=L, src=rtr[j], shift=s_dst - s_src, hits=len(found), planted_found=ok))
    if trial == 1: break  # 2 trials x 3 L enough (index build is the slow part)
print("\nC0 positive control (planted 6h exact copy into F13_221 copy):")
print(pd.DataFrame(ctrl).to_string(index=False))

# ---------- H1a: all-farm shifted correlation scan ----------
f1347_tr = np.array([k for k, r in enumerate(rtr) if r[:3] in ("F13", "F47")])
Q = np.concatenate([Ate[[list(rte).index(r) for r in T8]], Atr[f1347_tr]], 0)
qname = np.array(T8 + list(rtr[f1347_tr]))
qself = np.array([-1] * 8 + list(f1347_tr))
M = len(rtr)
best = np.full((len(Q), M), -np.inf, dtype=np.float32)
bsh = np.zeros((len(Q), M), dtype=np.int8)
def zrows(A):
    m = A.mean(1, keepdims=True); s = A.std(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (A - m) / s
    z[~np.isfinite(z)] = np.nan
    return z
for sh in range(-6, 7):
    xs = slice(max(0, sh), 24 + min(0, sh)); ys = slice(max(0, -sh), 24 + min(0, -sh)); L = 24 - abs(sh)
    cmin = None
    for v in range(3):
        zx = zrows(Q[:, xs, v]); zy = zrows(Atr[:, ys, v])
        okx = ~np.isnan(zx).any(1); oky = ~np.isnan(zy).any(1)
        zx = np.nan_to_num(zx).astype(np.float32); zy = np.nan_to_num(zy).astype(np.float32)
        c = zx @ zy.T / L
        c[~okx] = np.nan; c[:, ~oky] = np.nan
        cmin = c if cmin is None else np.fmin(np.where(np.isnan(cmin), np.nan, cmin), c) if False else np.minimum(cmin, c)
    cmin = np.where(np.isnan(cmin), -np.inf, cmin)
    upd = cmin > best
    best[upd] = cmin[upd]; bsh[upd] = sh
for q in range(len(Q)):
    if qself[q] >= 0: best[q, qself[q]] = -np.inf
# same-date twin flag (outdoor 24h identical, train partner in F13/F47)
def twins(o24, excl=-1):
    eq = (np.abs(Otr[f1347_tr] - o24) < 1e-6).all(2).all(1)
    t = f1347_tr[eq]; return [k for k in t if k != excl]
Oq = np.concatenate([Ote[[list(rte).index(r) for r in T8]], Otr[f1347_tr]], 0)
rows = []
for q in range(len(Q)):
    o = np.argsort(-best[q])[:3]
    tw = twins(Oq[q], qself[q])
    rows.append(dict(rec=qname[q], kind="EVAL8" if q < 8 else "TRAIN", has_twin=len(tw) > 0,
                     top1=best[q, o[0]], p1=rtr[o[0]], sh1=int(bsh[q, o[0]]), top2=best[q, o[1]], p2=rtr[o[1]],
                     p1_is_twin=o[0] in tw, n99=int((best[q] >= .99).sum()), n999=int((best[q] >= .999).sum())))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "fx3_corrscan_v1.csv"), index=False)
pd.set_option("display.width", 250)
print("\nH1a: 8 eval records, best min-corr over all 12,932 train records (all farms), shifts -6..6:")
print(R[R.kind == "EVAL8"].round(4).to_string(index=False))
b = R[R.kind == "TRAIN"]
for tw in (True, False):
    s = b[b.has_twin == tw].top1
    print(f"baseline TRAIN F13/F47 has_twin={tw}: n={len(s)}, top1 q50 {s.quantile(.5):.4f} q90 {s.quantile(.9):.4f} q99 {s.quantile(.99):.4f} max {s.max():.4f}; n>=.99 {int((s >= .99).sum())}")
print("baseline partner farm of top1 (train, has_twin=True): same-date twin share", round(b[b.has_twin].p1_is_twin.mean(), 3))
print("other-farm top1 share among TRAIN baseline:", round((b.p1.str[:3].isin(['F13', 'F47']) == False).mean(), 3))

# ---------- H1b: same-date different-dong train pairs (shift 0) ----------
vals = []
for a in f1347_tr:
    for bb in twins(Otr[a], a):
        if bb <= a: continue
        x = Atr[a]; y = Atr[bb]
        ok = ~np.isnan(x).any(1) & ~np.isnan(y).any(1)
        if ok.sum() < 18: continue
        with np.errstate(invalid="ignore"):
            c = [np.corrcoef(x[ok, v], y[ok, v])[0, 1] for v in range(3)]
        vals.append(np.nanmin(c))
vals = np.array(vals)
print(f"\nH1b same-date train pairs (F13/F47) shift-0 min-corr: n={len(vals)} q50 {np.median(vals):.4f} q90 {np.quantile(vals, .9):.4f} max {vals.max():.4f}; share >= .978: {(vals >= .978).mean():.3f}, >= .949: {(vals >= .949).mean():.3f}")
