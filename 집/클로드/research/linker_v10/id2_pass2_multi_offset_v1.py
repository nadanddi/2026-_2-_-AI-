# -*- coding: utf-8 -*-
"""ID2 (2026-10-12 집 클로드; 사용자: '2차 기록을 좀 늘려봐').  서술.
학습 2차 기록은 56 개뿐 → 거울 CV 덩어리 시작 위치를 o = 0..9 로 밀어 10 가지 숨김 기하를 만들고, 각 기하에서 2차 학습 기록만 대상으로 판정.
덩어리 = 같은 ID 번호 [s, s+10), 양옆 s−1·s+10 제거, 다른 ID 거울 덩어리(번호 ±2) 숨김; 숨김에 닿는 학습 연결 빼고 덩어리마다 재학습(v7m 과 같은 규칙).
정답 = 진단용 연결 중 달력(DM11) 차 정확히 1일인 것(깨끗한 정답).  대상 = 진짜 전날이 그 기하에서 보이는 정답 기록인 2차 기록.
모델: IN(입력만: oj + v0..v7 + 0시 지문), FULL(v10 특징 + 지문), HP2(외기 정확 일치 날짜 + 지문; 날짜 없으면 FULL 1순위 날짜).
보고: 판정 건수·기록 수, 판정 단위 정확도, 기록 단위 평균 정확도(기록마다 맞은 비율 평균), HP2 층별.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
exec(open("id1_input_only_linker_v1.py", encoding="utf-8").read().split("rows = []\nfor f in")[0])
SUCC = {}
for a_, b_, c_ in zip(TLa.a1, TLa.b, TLa.cert):
    if c_: SUCC.setdefault(int(a_), []).append(int(b_))
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
            tr = [(d, y) for b_, a_, d, y in TRG if not touch[b_] and not touch[a_]]
            w_in = fit_clogit([(d[FIN].values.astype(float), y) for d, y in tr], l2=0.5, iters=300)
            w_fu = fit_clogit([(d[FFULL].values.astype(float), y) for d, y in tr], l2=0.5, iters=300)
            lab = TRAIN.copy(); lab[touch] = False
            for b in tg:
                t = TP[b]
                if not lab[t]: continue
                D = feats(b, lab, rem2)
                if len(D) == 0: continue
                a = D.a.values.astype(int)
                pin = a[np.argmax(D[FIN].values.astype(float) @ w_in)]; pfu = a[np.argmax(D[FFULL].values.astype(float) @ w_fu)]
                vi = np.where(lab)[0]; eq = np.all(np.abs(X[vi, 0] - X[b, 0]) < 1e-6, axis=1); ds = DAY[vi[eq]]; ds = ds[np.isfinite(ds)]
                if len(ds): dprev = pd.Series(ds).mode().iloc[0] - 1; tier = "A"
                else: dprev = DAY[pfu]; tier = "B"
                cand = [x for x in vi if DAY[x] == dprev and not any(lab[s_] and DAY[s_] == dprev + 1 for s_ in SUCC.get(x, []))]
                php = min(cand, key=lambda x: np.nanmean(np.minimum(np.abs(FPA[x, 0] - FPA[b, 0]), 4))) if cand else -1
                rows.append(dict(o=o, rec=rid(b), tier=tier, ok_in=pin == t, ok_full=pfu == t, ok_hp2=php == t))
T = pd.DataFrame(rows)
print(f"judgements {len(T)} over {T.rec.nunique()} pass-2 records (clean truth), offsets 10")
R = T.groupby("rec")[["ok_in", "ok_full", "ok_hp2"]].mean()
for c, nm in (("ok_in", "input-only"), ("ok_full", "v10+fp (ID/number)"), ("ok_hp2", "HP2 (date by outdoor match + fingerprint)")):
    print(f"   {nm:44s} judgement-level {T[c].mean():.3f} | record-level mean {R[c].mean():.3f} | records always right {(R[c] == 1).sum()} / never right {(R[c] == 0).sum()}")
for tier, G in T.groupby("tier"):
    print(f"   HP2 tier {tier}: judgements {len(G)}, records {G.rec.nunique()}, HP2 correct {G.ok_hp2.mean():.3f}, v10+fp {G.ok_full.mean():.3f}, input-only {G.ok_in.mean():.3f}")
T.to_csv(os.path.join(env.LOCAL, "linker_v10", "id2_rows.csv"), index=False)
