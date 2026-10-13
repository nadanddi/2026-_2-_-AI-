# -*- coding: utf-8 -*-
"""GP1 (2026-10-14 집 클로드; 사용자: '전체 경로 최적화 방식으로 링커 만들어봐').  서술·탐색(판정 아님).
Optiver 7위(전체 경로)·DT2(헝가리안)에서 착안: 기록마다 따로 1순위를 고르는 대신, 한 기록은 많아야 한 기록의 전날이 된다는 1:1 제약을 함께 풂.
짝 후보·확률 = v15R EV 짝 표(v15r15_ev: prev0..5, p0..5, p_null; 폴드마다 정답 보임 기준으로 만든 것).
  BASE   = 1순위 prev0 (현재).
  GP-RS  = 규정판: 후보 a 를 '보이는 학습 기록 b′(정답 있음, 그 폴드 숨김 아님) 가 1순위로 더 높은 확률로 지목' 하면 제외하고 다음 후보로; 다 빠지면 없음.
           (경쟁자로 학습 기록만 봄 → 평가에서도 다른 ID·뒤 번호 평가 입력을 쓰지 않음)
  GP-ALL = 상한(평가에 못 씀): 400기록 전체를 한 번에 최소 비용 1:1 배정(비용 −log p, '없음' 비용 −log p_null), scipy linear_sum_assignment.
정답 = truth.pkl cert 연결(학습 정답 연속만).  지표: 짝 정확도(cert 대상), 확률 구간별 정확도·개수, DS8 원인 짝(F47_126·F13_160) 결과.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.join(HERE, "linker_v4")); from feat4 import EC, FARM, NUM, N, TRAIN, S3  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
with np.errstate(all="ignore"): DM = np.nanmean(EC, 1)
TL = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {rid(int(b)): rid(int(a)) for a, b, c in zip(TL.a1, TL.b, TL.cert) if c}
PE = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "v15r15_ev.csv"))
FO = pd.read_csv(os.path.join(HERE, "ev_validator", "ev_folds_v1.csv")); hidEV = {k: set(g.rec[g.role.isin(["val", "del"])]) for k, g in FO.groupby("fold")}
CAND = {r: [(q[f"prev{j}"], float(q[f"p{j}"])) for j in range(6) if isinstance(q[f"prev{j}"], str)] for r, (_, q) in zip(PE.rec, PE.iterrows())}
PNULL = dict(zip(PE.rec, PE.p_null)); FOLD = dict(zip(PE.rec, PE.fold))
BASE = {r: (c[0] if c else (None, 0.0)) for r, c in CAND.items()}
def gp_rs(r):
    hs = hidEV[int(FOLD[r])]
    for a, p in CAND[r]:
        rivals = [b for b in CAND if b != r and TRAIN[idx[b]] and b not in hs and np.isfinite(DM[idx[b]]) and BASE[b][0] == a and BASE[b][1] > p]
        if not rivals: return a, p
    return None, PNULL[r]
RS = {r: gp_rs(r) for r in CAND}
recs = list(CAND); cols = sorted({a for c in CAND.values() for a, _ in c}); ci = {a: j for j, a in enumerate(cols)}
BIG = 1e6; C = np.full((len(recs), len(cols) + len(recs)), BIG)
for i, r in enumerate(recs):
    for a, p in CAND[r]:
        if a != r: C[i, ci[a]] = -np.log(max(p, 1e-9))
    C[i, len(cols) + i] = -np.log(max(PNULL[r], 1e-9))
ri, cj = linear_sum_assignment(C)
ALL = {}
for i, j in zip(ri, cj):
    r = recs[i]
    if j < len(cols) and C[i, j] < BIG: ALL[r] = (cols[j], dict(CAND[r]).get(cols[j], 0.0))
    else: ALL[r] = (None, PNULL[r])
rows = []
for r in recs:
    if r not in CERT: continue
    for nm, D in (("BASE", BASE), ("GP-RS", RS), ("GP-ALL", ALL)):
        a, p = D[r]; rows.append(dict(rec=r, arm=nm, pick=a, p=p, linked=a is not None, ok=(a == CERT[r]) if a is not None else np.nan))
T = pd.DataFrame(rows); pd.set_option("display.width", 200)
print("cert targets:", T.rec.nunique())
print(T.groupby("arm").agg(linked=("linked", "mean"), acc_when_linked=("ok", "mean"), n_correct=("ok", "sum")).round(3).to_string())
for nm, g in T.groupby("arm"):
    print(f"  {nm}: p>=.8 n {int((g.p >= .8).sum())} acc {g[g.p >= .8].ok.mean():.3f} | p .5-.8 n {int(((g.p >= .5) & (g.p < .8)).sum())} acc {g[(g.p >= .5) & (g.p < .8)].ok.mean():.3f} | p<.5 n {int((g.linked & (g.p < .5)).sum())} acc {g[g.linked & (g.p < .5)].ok.mean():.3f}")
ch = T.pivot_table(index="rec", columns="arm", values="pick", aggfunc="first")
for nm in ("GP-RS", "GP-ALL"):
    d = ch[ch["BASE"] != ch[nm]].index; sub = T[T.rec.isin(d)].pivot_table(index="rec", columns="arm", values="ok", aggfunc="first")
    print(f"\n{nm} changed {len(d)} cert targets: BASE correct {int(sub['BASE'].fillna(0).sum())} -> {nm} correct {int(sub[nm].fillna(0).sum())} (unlinked {int(sub[nm].isna().sum())})")
for r in ("F47_126", "F13_160"): print(r, {nm: D[r] for nm, D in (("BASE", BASE), ("GP-RS", RS), ("GP-ALL", ALL))})
pd.DataFrame([dict(rec=r, base=BASE[r][0], rs=RS[r][0], rs_p=RS[r][1], all=ALL[r][0]) for r in recs]).to_csv(os.path.join(HERE, "gp1_picks_v1.csv"), index=False)
