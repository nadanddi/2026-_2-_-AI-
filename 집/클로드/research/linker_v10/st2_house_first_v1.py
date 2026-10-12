# -*- coding: utf-8 -*-
"""ST2 (2026-10-12 집 클로드; 사용자: '동 먼저 맞추고 날짜를 맞추면').  서술(학습 없는 2단계 규칙, ST1 의 순서 반대).
후보 = v10 후보 집합(거울 CV, 0시).  1단계 동: 실내·구동기 자정 연속 Σ min(dv_j, 6) + 0시 운영 지문 거리 fp(8변수 표준화, ×5 로 눈금 맞춤) 작은 상위 K.
2단계 날짜: 그 K 개 중 외기 자정 연속 oj 최소.  K = 1(동만), 3, 5, 10.
보고(확실 연결 이을 수 있는 65): 1단계 상위 K 안에 진짜가 있는 비율, 최종 1순위 정확도; ST1(날짜 먼저 .538)·v10·META 와 비교.
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
KS = (1, 3, 5, 10); res = {k: [] for k in KS}; inK = {k: [] for k in KS}; p2 = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if not TRAIN[b] or b not in TP or not CERT[b] or not lab[TP[b]]: continue
            D, _ = features10(b, 0, lab, removed=rem2)
            if len(D) == 0: continue
            a = D.a.values.astype(int); t = TP[b]
            fp = np.nan_to_num(np.nanmean(np.minimum(np.abs(FPA[a, 0] - FPA[b, 0][None]), 4), axis=1), nan=1.0)
            hs = np.minimum(D[[f"v{j}" for j in range(8)]].values, 6).sum(1) + 5 * fp
            order = np.argsort(hs); p2.append(NUM[b] >= 179)
            for k in KS:
                sl = order[:k]; pick = a[sl[np.argmin(D.oj.values[sl])]]
                res[k].append(pick == t); inK[k].append(t in set(a[sl]))
P2 = np.array(p2)
print(f"cert linkable: {len(P2)} (pass-2 {int(P2.sum())})")
for k in KS:
    r = np.array(res[k]); i = np.array(inK[k])
    print(f"   K={k:2d}: truth in house shortlist {i.mean():.3f} | final top-1 correct {r.mean():.3f} | pass-2 {r[P2].mean():.3f}")
print("reference: date-first (ST1) top-1 .538 | v10 top-1 (cert) see LR2: >=.8 38 anchors prec 1.000 rec .58 | META >=.7 rec .68")
