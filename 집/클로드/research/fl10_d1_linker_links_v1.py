# -*- coding: utf-8 -*-
"""FL10 (2026-10-16 집 클로드; 사용자 '9등 넘게 철저하게' — 기록 번호 = 공식 '온실별 상대 일차'(문제설명서, 7.045) -> 현장형 링커에 '같은 농장 상대 일차 바로 전날(번호 차 1)' 특징 d1 만 다시 넣음.
번호 차 2(d2w·d2c)·d3·ldn·179 경계·달력 위치(isprev·dsdev)는 계속 제외).  가중치 = FL 특징 + d1 로 정답 짝 전체(train_groups7) 조건부 로짓 새로 적합(fl10_w.npy).
갈래 OUT·IN 은 FL8 과 같은 마스크(d1 제외).  연결표 생성(판정 없음), 실행 전 커밋.  -> local/fl10_two_view_links.csv (FL8 과 같은 열).
원 FL8 설명: (2026-10-16 집 클로드; TAG1 비평(7.029): 일치 링커가 번호·달력 특징 + 좁은 숨김 -> 5점·누설 문제.  현장형 두 갈래 증거로 다시 정의).
연결표 생성(판정 없음), 실행 전 커밋.  FL5 와 같은 폴드 거울(배치 EXT10·EXT12·DIAG10 폴드 k 의 가림 기록 전체 + 같은 농장 번호 ±1 정답 숨김, 평가 기록 입력 안 씀),
같은 후보(features7)와 같은 가중치(fl3_w.npy)에서 세 가지 선택:
  FL  = 전체 점수(oj + v0..v7 + excl + xf)  — FL5 와 같음(재현 확인)
  OUT = 바깥 날씨 이어짐만(oj: 어제 23시 외기 4종 외삽 vs 오늘 0시) + excl + xf
  IN  = 실내 센서·운영 이어짐만(v0..v7: 실내 온·습·CO2 + 구동기 5종의 어제 23시 vs 오늘 0시) + excl + xf
번호 차·달력 위치 특징(d1·d3·ldn·d2w·d2c·isprev·dsdev) 은 FL 처럼 쓰지 않음.
-> local/fl8_two_view_links.csv (layout, fold, rec, prev, p, prev_out, p_out, prev_in, p_in).  서술: FL5 와 일치율, 갈래 일치 분포, 정답 표(cert) 진짜 비율.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u fl8_two_view_mirror_links_v1.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
sys.path.insert(0, os.path.join(HERE, "linker_v7")); sys.path.insert(0, HERE)
import env  # noqa: F401
from feat7 import *  # noqa: F401,F403
import numpy as np, pandas as pd
FL = ["oj"] + [f"v{j}" for j in range(8)] + ["excl", "xf", "d1"]; IFL = [FEATS7.index(c) for c in FL]
w = fit_clogit([(X_[:, IFL], y_) for b_, a_, X_, y_ in train_groups7()], l2=0.5, iters=300); np.save(os.path.join(env.LOCAL, "fl10_w.npy"), w); print("FL10 weights", dict(zip(FL, np.round(w, 3))), flush=True)
MO = np.array([1] + [0] * 8 + [1, 1, 0], float); MI = np.array([0] + [1] * 8 + [1, 1, 0], float)
def soft(s): p = np.exp(s - s.max()); return p / p.sum()
def picks(b, lab, removed):
    df, _ = features7(b, 0, lab, removed=removed)
    if len(df) == 0: return None
    X_ = df[FL].values.astype(float); A = df.a.values; out = {}
    for nm, m in (("", np.ones(len(FL))), ("_out", MO), ("_in", MI)):
        p = soft(X_ @ (w * m)); k = int(np.argmax(p)); out["prev" + nm] = rid(int(A[k])); out["p" + nm] = round(float(p[k]), 4)
    return out
RID = {rid(b): b for b in range(N)}
CK = os.path.join(env.LOCAL, "tt1_ckpt"); G = pd.concat([pd.read_csv(os.path.join(CK, f), usecols=["validator", "fold", "row_id"]) for f in sorted(os.listdir(CK))])
G["rec"] = G.row_id.str[:7]; G = G[G.validator.isin(["EXT10", "EXT12", "DIAG10"])].drop_duplicates(["validator", "fold", "rec"])
out = []
for (L, k), S in G.groupby(["validator", "fold"]):
    held = np.array([RID[r] for r in S.rec if r in RID])
    for b in held:
        lab = TRAIN.copy(); lab[held] = False; lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
        o = picks(b, lab, set(TEST_IDX))
        if o is not None: out.append(dict(layout=L, fold=int(k), rec=rid(b), **o))
    print(L, k, "done", len(held), flush=True)
D = pd.DataFrame(out); D.to_csv(os.path.join(env.LOCAL, "fl10_two_view_links.csv"), index=False)
M5 = pd.read_csv(os.path.join(env.LOCAL, "fl5_mirror_links.csv")); X5 = D.merge(M5, on=["layout", "fold", "rec"], suffixes=("", "_fl5"))
print("FL10 same prev as FL5:", round(float((X5.prev == X5.prev_fl5).mean()), 4), "| n", len(X5))
TT = pd.read_pickle(os.path.join(env.LOCAL, "linker_v3", "truth.pkl")); TT = TT[TT.cert]
TRUEP = {f"{fb}_{int(nb):03d}": f"{fa}_{int(na):03d}" for fb, nb, fa, na in zip(TT.fb, TT.nb, TT.fa, TT.na)}
D["n_agree"] = (D.prev_out == D.prev).astype(int) + (D.prev_in == D.prev).astype(int); D["true"] = [TRUEP.get(r) == p if r in TRUEP else np.nan for r, p in zip(D.rec, D.prev)]
D["pbin"] = pd.cut(D.p, [0, .5, .8, 1.01])
print(D.groupby(["pbin", "n_agree"], observed=True).agg(n=("rec", "size"), true_rate=("true", "mean"), known=("true", "count")).round(3).to_string())
