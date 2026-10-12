# -*- coding: utf-8 -*-
"""HP1 (2026-10-12 집 클로드; 사용자: '전날 기록이 있는 것끼리는 거의 완벽하게 매칭하려면').  서술.
행 h 에서는 대상 기록의 0..h 시 입력을 쓸 수 있음(규정).  v10(0시) 상위 3 후보 중에서 0..h 시 운영 지문 거리(같은 시각끼리, 8변수 표준화, 상한 4)로 다시 고르면
시각별로 얼마나 정확해지나.  대상 = 확실 연결 이을 수 있는 기록(거울 CV, 65).
보고: h = 0, 3, 6, 9, 12, 17, 23 별 (a) v10 상위 3 에 진짜가 있는 비율, (b) 지문 재선택 정확도, (c) 1·2등 거리 차 ≥ m 일 때 정확도·포괄률.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
_V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
_X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
_X = _X[_X.row_id.str[:3].isin(["F13", "F47"])]; _X["rec"] = _X.row_id.str[:7]; _X["h"] = _X.row_id.str[8:].astype(int)
FPA = np.stack([_X.pivot(index="rec", columns="h", values=v).reindex(index=[rid(i) for i in range(N)], columns=range(24)).values for v in _V], 2)
FPA = (FPA - np.nanmean(FPA, (0, 1))) / (np.nanstd(FPA, (0, 1)) + 1e-9)
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TP = {int(b): int(a) for a, b in zip(TLa.a1, TLa.b)}; CERT = {int(b): bool(c) for b, c in zip(TLa.b, TLa.cert)}
TRG = train_groups10(); HS = (0, 3, 6, 9, 12, 17, 23); rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w = fit_clogit([(X_, y_) for b_, a_, X_, y_ in TRG if not touch[b_] and not touch[a_]], l2=0.5, iters=300)
        lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if not TRAIN[b] or b not in TP or not CERT[b] or not lab[TP[b]]: continue
            D, _ = features10(b, 0, lab, removed=rem2)
            if len(D) == 0: continue
            a = D.a.values.astype(int); sc = D[FEATS10].values.astype(float) @ w; top = a[np.argsort(-sc)[:3]]; top = np.array([x for x in top if lab[x]])
            t = TP[b]; row = dict(rec=rid(b), in3=t in set(top), v10_top1=a[np.argmax(sc)] == t)
            for h in HS:
                if len(top) == 0: row[f"ok{h}"] = False; row[f"m{h}"] = 0; continue
                d = np.array([np.nanmean(np.minimum(np.abs(FPA[x, :h + 1] - FPA[b, :h + 1]), 4)) for x in top]); o = np.argsort(d)
                row[f"ok{h}"] = top[o[0]] == t; row[f"m{h}"] = (d[o[1]] - d[o[0]]) if len(top) > 1 else 9
            rows.append(row)
T = pd.DataFrame(rows)
print(f"cert linkable {len(T)} | truth in v10 top-3 (labelled) {T.in3.mean():.3f} | v10 top-1 correct {T.v10_top1.mean():.3f}")
for h in HS:
    line = f"   h {h:2d}: fingerprint re-pick correct {T[f'ok{h}'].mean():.3f}"
    for mth in (.05, .1, .2):
        S = T[T[f"m{h}"] >= mth]; line += f" | margin>={mth}: n {len(S)} prec {S[f'ok{h}'].mean() if len(S) else float('nan'):.3f} rec {S[f'ok{h}'].sum()/len(T):.2f}"
    print(line)
