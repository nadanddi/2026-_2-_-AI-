# V11b (2026-10-12 집 클로드; 사용자: '안전하게 더 높일 방법') 서술: v11 이 늘 틀리는 5기록이 확신도로 가려지나 (10기하·2차).
# -*- coding: utf-8 -*-
"""V11 (2026-10-12 집 클로드; 사용자: '같은 ID 에 가중을 주면 안 되겠다').  서술(ID3 와 같은 10기하·2차·깨끗한 정답).
짝찾기 v11 = v10 후보·가시성 규칙(규정) 그대로, 특징에서 ID·번호 기반(isprev, dsdev, dsna, d1, d2w, d2c, d3, ldn, xf, lat) 전부 제거,
  대신 모든 후보 공통 달력 특징: cd1 = [후보 날짜 == 대상 날짜 − 1], cdev = min(|후보 날짜 − (대상 날짜 − 1)|, 6)/6, cna = [대상 또는 후보 날짜 모름].
  날짜: 대상 = 0시 외기 4칸이 정확히 같은 '보이는 정답 기록'들의 DM11 날짜(최빈); 후보 = 보이는 정답 기록이면 DM11, 아니면 같은 외기 일치 방식.
  나머지 특징: oj, v0..v7, fp(0시 지문), excl(다음날이 이미 정답으로 확정된 동).
비교: v10+fp(ID 특징 포함), C1(HP2/v10), C1'(HP2/v11).  같은 ID·다른 ID 진실별.
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

BAD = {"F13_214", "F13_244", "F47_191", "F47_239", "F47_231"}
rows = []
for o in range(10):
    for f in ["F13", "F47"]:
        g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
        lo, hi = NUM[FARM == f].min(), NUM[FARM == f].max()
        for s in range(lo - 10 + o, hi + 1, 10):
            blk = sorted(np.where((FARM == f) & (NUM >= s) & (NUM < s + 10))[0], key=lambda i: NUM[i])
            tg = [b for b in blk if TRAIN[b] and NUM[b] >= 179 and b in TP and DAY[b] - DAY[TP[b]] == 1]
            if not tg: continue
            rem = {i for i in np.where((FARM == f) & ((NUM == s - 1) | (NUM == s + 10)))[0]}
            m = s + sh; oth = set(np.where((FARM == g) & (NUM >= m - 1) & (NUM <= m + 10))[0].tolist()); rem2 = rem | oth
            touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[list(oth)] = True
            w11 = fit_clogit([(d[F11].values.astype(float), y) for b_, a_, d, y in TRG if not touch[b_] and not touch[a_]], l2=0.5, iters=300)
            lab = TRAIN.copy(); lab[touch] = False
            for b in tg:
                t = TP[b]
                if not lab[t]: continue
                D = feats11(b, lab, rem2)
                if len(D) == 0: continue
                a = D.a.values.astype(int); sc = D[F11].values.astype(float) @ w11; p = np.exp(sc - sc.max()); p /= p.sum(); o_ = np.argsort(-p)
                rows.append(dict(o=o, rec=rid(b), ok=a[o_[0]] == t, p1=p[o_[0]], marg=p[o_[0]] - (p[o_[1]] if len(p) > 1 else 0), cna=D.cna.values[o_[0]], bad=rid(b) in BAD, pick=rid(a[o_[0]]), truth=rid(t)))
T = pd.DataFrame(rows)
print("confidence of v11 top pick: wrong vs right")
print(T.groupby("ok")[["p1", "marg", "cna"]].describe().loc[:, (slice(None), ["mean", "25%", "50%", "75%"])].round(3).to_string())
for th in (.5, .6, .7, .8, .9):
    S = T[T.p1 >= th]; print(f"   p1 >= {th}: kept {len(S)}/{len(T)} judgements, accuracy {S.ok.mean():.3f}, records kept {S.rec.nunique()}/{T.rec.nunique()}, wrong kept {int((~S.ok).sum())}")
print(T[T.bad].groupby("rec").agg(p1=("p1", "mean"), marg=("marg", "mean"), cna=("cna", "mean"), pick=("pick", "first"), truth=("truth", "first")).round(3).to_string())
