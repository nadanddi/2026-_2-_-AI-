# -*- coding: utf-8 -*-
"""FL3 (2026-10-15 집 클로드; 5점 모델 27번용).  현장형 링커(FL1·FL2 특징: oj, v0~v7, excl, xf) 연결표 생성(판정 없음), 실행 전 커밋.
가중치 = truth 짝 전체(train_groups7)로 조건부 로짓 적합(FL1 과 같음).
 TRAIN 표(학습 기록 400): 대상 같은 농장 번호 ±1 정답 숨김(가중치 학습 때와 같은 미리 고정 규칙, 평가 구조와 무관), 후보에서 평가 기록 제외(MASK: 평가 입력 안 씀)
   → local/fl3_train_links.csv (rec, prev, p)  — V3L·EV3L 학습 창, 사슬 표
 EVAL 표(평가 60): 기록마다 독립(정답 = 학습 기록 전부, 평가 후보는 같은 농장 앞 일차만 — feat 의 visible 규칙), 0시 결정
   → local/fl3_eval_links.csv (rec, prev, p, prev_is_labelled, xf, anchor_st23, anchor_ec23, dT_air)
Run:  cd 집/클로드/research && PYTHONPATH="" python -u fl3_field_linker_tables_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import *  # noqa: F401,F403
import numpy as np, pandas as pd
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf"]; IFL = [FEATS7.index(c) for c in FL]
TRG = train_groups7(); w = fit_clogit([(X_[:, IFL], y_) for b, a, X_, y_ in TRG], l2=0.5, iters=300); np.save(os.path.join(env.LOCAL, "fl3_w.npy"), w)
print("FL weights:", dict(zip(FL, np.round(w, 3))), flush=True)
def pick(b, lab, removed):
    df, _ = features7(b, 0, lab, removed=removed)
    if len(df) == 0: return None, 0.0
    s = df[FL].values.astype(float) @ w; p = np.exp(s - s.max()); p /= p.sum(); k = int(np.argmax(p)); return int(df.a.values[k]), float(p[k])
tr = []
for b in range(N):
    if not TRAIN[b] or FARM[b] not in ("F13", "F47"): continue
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    a0, p0 = pick(b, lab, set(TEST_IDX))
    if a0 is not None: tr.append(dict(rec=rid(b), prev=rid(a0), p=round(p0, 4)))
T = pd.DataFrame(tr); T.to_csv(os.path.join(env.LOCAL, "fl3_train_links.csv"), index=False)
TLc = pd.read_pickle(os.path.join(S3, "truth.pkl")); CERT = {rid(int(bb)): rid(int(aa)) for aa, bb, c in zip(TLc.a1, TLc.b, TLc.cert) if c}
c = T[T.rec.isin(CERT.keys())]; print(f"TRAIN links {len(T)} | p>=.5 {int((T.p >= .5).sum())} | acc vs cert {np.mean([CERT[r] == p for r, p in zip(c.rec, c.prev)]):.3f} (n {len(c)})", flush=True)
ev = []
for f in ("F13", "F47"):
    for b in sorted(np.where((FARM == f) & ~TRAIN)[0], key=lambda i: NUM[i]):
        a0, p0 = pick(b, TRAIN.copy(), set())
        if a0 is None: continue
        ev.append(dict(rec=rid(b), prev=rid(a0), p=round(p0, 4), prev_is_labelled=bool(TRAIN[a0]), xf=FARM[a0] != f, anchor_st23=round(ST[a0, 23], 2) if TRAIN[a0] else np.nan,
                       anchor_ec23=round(EC[a0, 23], 3) if TRAIN[a0] else np.nan, dT_air=round(IN_[b, 0, 0] - IN_[a0, 23, 0], 2)))
E = pd.DataFrame(ev); E.to_csv(os.path.join(env.LOCAL, "fl3_eval_links.csv"), index=False)
an = E[(E.p >= .8) & E.prev_is_labelled & (E.dT_air.abs() <= 1.5)]
print(f"EVAL links {len(E)} | p>=.5 {int((E.p >= .5).sum())} | anchors(p>=.8, labelled, |dT|<=1.5) {len(an)} | cross-farm {int(an.xf.sum())}")
print("W2/B5 records:", E[E.rec.isin(["F13_235", "F13_237", "F47_233"])][["rec", "prev", "p", "anchor_ec23"]].to_dict("records"))
