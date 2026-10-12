# HP2 (2026-10-12 집 클로드; 사용자: '전날 기록 있는 것끼리 거의 완벽 매칭') 서술: 혼합 규칙
#  날짜: DF(0시 외기 정확 일치 → DM11 날짜 d) 가 있으면 d−1, 없으면 v10 1순위의 날짜.  후보 = 그 날짜의 보이는 정답 기록(같은 동 다음날이 이미 확실한 기록은 제외).
#  동: 0..h 시 운영 지문 거리 최소.  확실 연결 이을 수 있는 65, 시각 h 별 정확도; 날짜 출처(DF / v10 대체)별.
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
exec(open("hp1b_miss_types_v1.py", encoding="utf-8").read().split("TRG = train_groups10(); rows = []")[0])
SUCC = {}
for a_, b_, c_ in zip(TLa.a1, TLa.b, TLa.cert):
    if c_: SUCC.setdefault(int(a_), []).append(int(b_))
TRG = train_groups10(); rows = []; HS = (0, 3, 6, 9, 12, 17, 23)
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
            D, _ = features10(b, 0, lab, removed=rem2); a = D.a.values.astype(int); sc = D[FEATS10].values.astype(float) @ w; top1 = a[np.argmax(sc)]
            vi = np.where(lab)[0]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
            if len(ds): dprev = pd.Series(ds).mode().iloc[0] - 1; src = "DF date"
            else: dprev = DAY[top1]; src = "v10 date"
            cand = [x for x in vi if DAY[x] == dprev and not any(lab[s_] and DAY[s_] == dprev + 1 for s_ in SUCC.get(x, []))]
            t = TP[b]; row = dict(rec=rid(b), src=src, date_ok=dprev == DAY[t], in_cand=t in set(cand), v10_ok=top1 == t)
            for h in HS:
                if not cand: row[f"ok{h}"] = False; continue
                d = {x: np.nanmean(np.minimum(np.abs(FPA[x, :h + 1] - FPA[b, :h + 1]), 4)) for x in cand}
                row[f"ok{h}"] = min(d, key=d.get) == t
            rows.append(row)
T = pd.DataFrame(rows)
print(f"cert linkable {len(T)} | v10 top-1 {T.v10_ok.mean():.3f}")
for src, G in list(T.groupby("src")) + [("ALL", T)]:
    print(f"  [{src}] n {len(G)} | date correct {G.date_ok.mean():.3f} | truth in candidates {G.in_cand.mean():.3f} | " + " ".join(f"h{h} {G[f'ok{h}'].mean():.3f}" for h in HS))
