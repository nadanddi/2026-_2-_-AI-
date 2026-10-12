# -*- coding: utf-8 -*-
"""V11DC (2026-10-12 집 클로드; 사용자: '날짜 불일치는 최대한 비슷한 걸로') — v11 거울 CV 선택에 날짜 제약: 대상 날짜를 알면 '정확히 전날(DM11/외기 일치)'인 정답 후보 중 v11 점수 최고,
없으면 선택 안 함(앵커 없음); 대상 날짜 모르면 정답 후보 중 최고. 확률 p 는 허용 후보 안에서 다시 정규화. 출력 v11dc_mirror_picks.csv (TD3 입력).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
exec(open("id1_input_only_linker_v1.py", encoding="utf-8").read().split("TRG = []")[0])
SUCC = {}
for a_, b_, c_ in zip(TLa.a1, TLa.b, TLa.cert):
    if c_: SUCC.setdefault(int(a_), []).append(int(b_))
F11 = ["oj"] + [f"v{j}" for j in range(8)] + ["fp", "excl", "cd1", "cdev", "cna"]
def date_of(i, lab):
    if lab[i] and np.isfinite(DAY[i]): return DAY[i]
    vi = np.where(lab)[0]; eq = np.all(np.abs(X[vi, 0] - X[i, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
    return pd.Series(ds).mode().iloc[0] if len(ds) else np.nan
def feats11(b, lab, removed):
    D = feats(b, lab, removed)
    if len(D) == 0: return D
    db = date_of(b, lab & (np.arange(N) != b)); a = D.a.values.astype(int); da = np.array([date_of(x, lab) for x in a])
    na = ~np.isfinite(da) | (not np.isfinite(db))
    D["cd1"] = np.where(na, 0.0, (da == db - 1).astype(float)); D["cdev"] = np.where(na, 0.0, np.minimum(np.abs(da - (db - 1)), 6) / 6); D["cna"] = na.astype(float)
    return D
TRG = []
for b, a in zip(TL7.b, TL7.a):
    lab = TRAIN.copy(); lab[(FARM == FARM[b]) & (np.abs(NUM - NUM[b]) <= 1)] = False
    df = feats11(b, lab, TEST_IDX)
    if len(df) == 0: continue
    y = np.where(df.a.values == a)[0]
    if len(y): TRG.append((b, a, df, y[0]))

H_ = os.path.join(env.ROOT, "집", "클로드", "research")

picks = {}
for f in ["F13", "F47"]:
    g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
    for s in range(lo, hi + 1, 10):
        blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
        rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
        m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
        touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
        w11 = fit_clogit([(d[F11].values.astype(float), y) for b_, a_, d, y in TRG if not touch[b_] and not touch[a_]], l2=0.5, iters=300)
        lab = TRAIN.copy(); lab[touch] = False
        for b in blk:
            if not TRAIN[b]: continue
            D = feats11(b, lab, rem2)
            if len(D) == 0: continue
            a = D.a.values.astype(int); sc = D[F11].values.astype(float) @ w11; p = np.exp(sc - sc.max()); p /= p.sum()
            db = date_of(b, lab & (np.arange(N) != b))
            ok = np.array([lab[x] for x in a])
            if np.isfinite(db):
                da = np.array([date_of(x, lab) for x in a]); ok &= (da == db - 1)
            if not ok.any(): continue
            k = int(np.argmax(np.where(ok, p, -1)))
            picks[rid(b)] = (rid(a[k]), float(p[k] / p[ok].sum()), float(IN_[b, 0, 0] - IN_[a[k], 23, 0]), bool(np.isfinite(db)))
pd.DataFrame([(r, *v) for r, v in picks.items()], columns=["rec", "prev", "p", "dT", "date_known"]).to_csv(os.path.join(env.LOCAL, "linker_v10", "v11dc_mirror_picks.csv"), index=False)
print("date-constrained picks", len(picks))
