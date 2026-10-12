# -*- coding: utf-8 -*-
"""ID3 (2026-10-12 집 클로드; 사용자: '더 끌어올릴 수 없나').  서술.  ID2 와 같은 10기하·2차·깨끗한 정답에 결합 규칙 추가:
  C1: 층 A → HP2 / 층 B → v10.
  C2: 층 A 에서 HP2 == v10 이면 그것, 다르면 HP2 날짜 후보 안에서 0..h 지문(h = 9, 23)으로 재선택(행 h 에서 쓸 수 있는 정보 — 해당 시각 이후 행의 앵커 기준).
  C3: 층 B 날짜를 '같은 ID 앞 번호 기록(b−2, 없으면 b−1)의 날짜 + 1' 로 추정(그 기록이 학습이면 DM11, 평가 역할이면 그 기록의 0시 외기 정확 일치 날짜) → 그 날짜 −1 후보 중 지문.
  기록별 오답 목록(진단 정답 의심 확인용).
(원 ID2 설명)
ID2 (2026-10-12 집 클로드; 사용자: '2차 기록을 좀 늘려봐').  서술.
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
idx_ = {rid(i): i for i in range(N)}
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
                fpk = lambda x, h: np.nanmean(np.minimum(np.abs(FPA[x, :h + 1] - FPA[b, :h + 1]), 4))
                php = min(cand, key=lambda x: fpk(x, 0)) if cand else -1
                c2 = {}
                for h in (9, 23):
                    if tier == "A" and php != pfu and cand: c2[h] = min(cand, key=lambda x: fpk(x, h))
                    else: c2[h] = php if tier == "A" else pfu
                # C3: date from same-ID earlier record (b-2, else b-1)
                c3 = pfu if tier == "B" else php; dsrc = ""
                if tier == "B":
                    for k_ in (2, 1):
                        q = idx_.get(f"{FARM[b]}_{NUM[b] - k_:03d}")
                        if q is None or q in rem2: continue
                        if lab[q] and np.isfinite(DAY[q]): dq = DAY[q]
                        else:
                            eqq = np.all(np.abs(X[vi, 0] - X[q, 0]) < 1e-6, axis=1); dqs = DAY[vi[eqq]]; dqs = dqs[np.isfinite(dqs)]
                            dq = pd.Series(dqs).mode().iloc[0] if len(dqs) else np.nan
                        if np.isfinite(dq):
                            dd = dq + (1 if k_ == 2 else 0) - 1
                            cc = [x for x in vi if DAY[x] == dd and not any(lab[s_] and DAY[s_] == dd + 1 for s_ in SUCC.get(x, []))]
                            if cc: c3 = min(cc, key=lambda x: fpk(x, 0)); dsrc = f"b-{k_}"
                            break
                rows.append(dict(o=o, rec=rid(b), tier=tier, truth=rid(t), v10=rid(pfu), hp2=rid(php) if php >= 0 else "", ok_in=pin == t, ok_full=pfu == t, ok_hp2=php == t,
                                 ok_c1=(php == t) if tier == "A" else (pfu == t), ok_c2_h9=c2[9] == t, ok_c2_h23=c2[23] == t, ok_c3=c3 == t, c3src=dsrc))
T = pd.DataFrame(rows)
print(f"judgements {len(T)} over {T.rec.nunique()} pass-2 records")
R = T.groupby("rec").mean(numeric_only=True)
for c, nm in (("ok_in", "input-only"), ("ok_full", "v10+fp"), ("ok_hp2", "HP2"), ("ok_c1", "C1 A:HP2 / B:v10"), ("ok_c2_h9", "C2 (A disagree -> fp 0..9h)"), ("ok_c2_h23", "C2 (A disagree -> fp 0..23h)"), ("ok_c3", "C3 (B date from same-ID earlier record)")):
    print(f"   {nm:42s} judgement {T[c].mean():.3f} | record mean {R[c].mean():.3f} | never right {(R[c] == 0).sum()}")
for tier, G in T.groupby("tier"):
    print(f"   tier {tier}: n {len(G)} | HP2 {G.ok_hp2.mean():.3f} v10 {G.ok_full.mean():.3f} C2h9 {G.ok_c2_h9.mean():.3f} C2h23 {G.ok_c2_h23.mean():.3f} C3 {G.ok_c3.mean():.3f}")
print("   C3 date source counts (tier B):", T[T.tier == "B"].c3src.value_counts().to_dict())
bad = R[(R.ok_full < .5) & (R.ok_hp2 < .5)].index
print(); print("records mostly wrong for both v10 and HP2:"); print(T[T.rec.isin(bad)].drop_duplicates("rec")[["rec", "truth", "v10", "hp2", "tier"]].to_string(index=False))
T.to_csv(os.path.join(env.LOCAL, "linker_v10", "id3_rows.csv"), index=False)
