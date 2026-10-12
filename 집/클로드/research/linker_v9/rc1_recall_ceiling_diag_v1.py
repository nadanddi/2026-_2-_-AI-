# -*- coding: utf-8 -*-
"""RC1 (2026-10-12 집 클로드; 사용자: '포괄률을 엄청 높이려면').  서술(진단).
거울 CV(v7 기하)에서 '이을 수 있는' 기록(진짜 전날이 보이는 정답 기록) 각각에 대해:
  (1) v7 방식 조건부 로짓(덩어리마다 재학습, FEATS7) 후보 목록 안에 진짜가 있나, 있으면 몇 위인가(확률 순).
  (2) 진짜가 후보에 없으면 왜: 같은 날짜로 묶여 제외(대상과 외기 0시 일치) / 다른 농장인데 외기 자정 비용 cap(oj ≥ 6) / 그 밖.
  (3) DF: 날짜 찾음 여부, 찾은 날짜 = 진짜 날짜+1 여부, 후보에 진짜가 있나(다음날 확정 동 제외로 빠졌나).
보고: 진짜 순위 분포(1·2·3·4~5·6+·없음), 상위 k 후보 포괄률(k = 1, 2, 3, 5), DF 실패 유형.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v9")
from feat9 import *
idx = {rid(i): i for i in range(N)}
TPi = {int(b): int(a) for a, b in zip(pd.read_pickle(S3 + r"\truth.pkl").a1, pd.read_pickle(S3 + r"\truth.pkl").b)}
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day)); DAY = np.array([DAYr.get(rid(i), np.nan) for i in range(N)])
TLc = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLc[TLc.cert]; SUCC = {}
for a_, b_ in zip(TLc.a1, TLc.b): SUCC.setdefault(int(a_), []).append(int(b_))
TRG = train_groups7(); rows = []
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
            if not TRAIN[b] or b not in TPi: continue
            t = TPi[b]
            if not lab[t]: continue          # not linkable in mirror world
            df, info = features7(b, 0, lab, removed=rem2)
            rank = None; reason = ""
            if len(df):
                sc = df[FEATS7].values.astype(float) @ w; order = df.a.values[np.argsort(-sc)]
                if t in order: rank = int(np.where(order == t)[0][0]) + 1
            if rank is None:
                same_date = np.all(np.abs(X[t, 0] - X[b, 0]) < 1e-6) or (np.abs(X[t] - X[b]) < 1e-6).mean() >= .9
                pa = X[t, 23] + .5 * (X[t, 23] - X[t, 22]); oj = np.sum(np.array([1, 1, .3, .5]) * np.abs(pa - X[b, 0]) / sdO)
                reason = "same-date excluded" if same_date else ("other-farm oj>=6" if FARM[t] != f and oj >= 6 else ("removed/hidden" if t in rem2 else "other"))
            vi = np.where(lab)[0]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
            if len(ds) == 0: dfr = "DF: no date"
            else:
                d = pd.Series(ds).mode().iloc[0]
                if DAY[t] != d - 1: dfr = f"DF: date off (truth d{DAY[t]} vs d-1 {d - 1})"
                else:
                    cand = [a for a in vi if DAY[a] == d - 1 and not any(lab[s_] and DAY[s_] == d for s_ in SUCC.get(a, []))]
                    dfr = "DF: truth in cands" if t in cand else "DF: truth excluded (succ known)"
            rows.append(dict(rec=rid(b), truth=rid(t), rank=rank, ncand=len(df), reason=reason, df=dfr, xf=FARM[t] != f, p2=NUM[b] >= 179))
R = pd.DataFrame(rows)
print(f"linkable records: {len(R)}")
bins = pd.cut(R["rank"].fillna(999), [0, 1, 2, 3, 5, 10, 998, 1000], labels=["1", "2", "3", "4-5", "6-10", "11+", "not in list"])
print("truth rank in v7-style list:"); print(bins.value_counts().reindex(["1", "2", "3", "4-5", "6-10", "11+", "not in list"]).to_string())
for k in (1, 2, 3, 5, 10): print(f"   top-{k} contains truth: {(R['rank'].fillna(999) <= k).mean():.2f}")
print("\nnot-in-list reasons:"); print(R[R["rank"].isna()].reason.value_counts().to_string())
print("\nDF status:"); print(R.df.str.replace(r"\(truth.*", "(date off)", regex=True).value_counts().to_string())
print("\nunion: in v7 top-3 OR DF candidates: %.2f" % (((R["rank"].fillna(999) <= 3) | (R.df == "DF: truth in cands")).mean()))
R.to_csv(os.path.join(env.LOCAL, "linker_v9", "rc1_linkable.csv"), index=False)
