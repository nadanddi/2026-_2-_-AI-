# -*- coding: utf-8 -*-
"""ID1 (2026-10-12 집 클로드; 사용자: 'ID 가리고 입력 데이터만으로 전날 찾기 가능한가').  서술(거울 CV, v10 후보·가시성 규칙 그대로).
모델 IN: 조건부 로짓, 특징 = 외기 자정 연속 oj + 실내·구동기 자정 연속 v0..v7 + 0시 운영 지문 fp (번호·ID·번호 기반 달력 특징 isprev/dsdev/dsna/d1/d2*/d3/ldn/excl/xf/lat 제외).
모델 FULL: v10 특징 전부 + fp (비교용).  HP2 규칙(외기 정확 일치 날짜 + 지문)은 원래 ID·번호를 쓰지 않음 — HP2 층 A 정확도 함께.
보고: 1순위 정확도 — 확실 연결 이을 수 있는 65, 전체 이을 수 있는 171(달력 차 1 인 연결만 = 깨끗한 정답), 2차.
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
FIN = ["oj"] + [f"v{j}" for j in range(8)] + ["fp"]; FFULL = FEATS10 + ["fp"]
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TP = {int(b): int(a) for a, b in zip(TLa.a1, TLa.b)}; CERT = {int(b): bool(c) for b, c in zip(TLa.b, TLa.cert)}
def feats(b, lab, removed):
    D, _ = features10(b, 0, lab, removed=removed)
    if len(D): D["fp"] = np.nan_to_num(np.nanmean(np.minimum(np.abs(FPA[D.a.values.astype(int), 0] - FPA[b, 0][None]), 4), axis=1), nan=1.0)
    return D
TRG = []
for b, a in zip(TL7.b, TL7.a):
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    df = feats(b, lab, TEST_IDX)
    if len(df) == 0: continue
    y = np.where(df.a.values == a)[0]
    if len(y): TRG.append((b, a, df, y[0]))
rows = []
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        tr = [(d, y) for b_, a_, d, y in TRG if not touch[b_] and not touch[a_]]
        w_in = fit_clogit([(d[FIN].values.astype(float), y) for d, y in tr], l2=0.5, iters=300)
        w_fu = fit_clogit([(d[FFULL].values.astype(float), y) for d, y in tr], l2=0.5, iters=300)
        lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if not TRAIN[b] or b not in TP or not lab[TP[b]]: continue
            D = feats(b, lab, rem2)
            if len(D) == 0: continue
            a = D.a.values.astype(int); t = TP[b]
            pin = a[np.argmax(D[FIN].values.astype(float) @ w_in)]; pfu = a[np.argmax(D[FFULL].values.astype(float) @ w_fu)]
            rows.append(dict(rec=rid(b), cert=CERT[b], clean=(DAY[b] - DAY[t] == 1), p2=NUM[b] >= 179, ok_in=pin == t, ok_full=pfu == t))
T = pd.DataFrame(rows)
for nm, S in (("cert linkable", T[T.cert]), ("clean linkable (calendar gap 1)", T[T.clean]), ("clean pass-2", T[T.clean & T.p2])):
    print(f"{nm:32s} n {len(S):3d} | input-only top-1 {S.ok_in.mean():.3f} | full (v10+fp) top-1 {S.ok_full.mean():.3f}")
print("HP2 rule (no ID features; outdoor exact-match date + fingerprint): cert linkable tier A 44/44 correct, all cert .908 (6.722)")
