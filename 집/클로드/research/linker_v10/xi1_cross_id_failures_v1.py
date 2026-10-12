# -*- coding: utf-8 -*-
"""XI1 (2026-10-12 집 클로드; 사용자: '다른 ID 로 넘어간다는데 뭐가 문제인가').  서술(기본 거울 CV 기하 o=0, 숨김 없이 비교도).
ID3 에서 v10·HP2 가 대체로 틀린 4기록(F13_242, F13_244, F47_231, F47_191)마다: 진짜 전날 vs 고른 전날의
  달력 날짜, 같은 ID 여부, 0시 지문 거리(fp0)·0..9시(fp9), 실내·구동기 자정 연속 합(dv), 외기 자정 연속(oj), v10 점수와 그 분해(xf·번호 특징 몫).
그리고 학습 진실 연결 전체에서 '다른 ID 전날' 비율과 v10 의 다른 ID 진실 적중률(같은 ID 진실과 비교).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
exec(open("id1_input_only_linker_v1.py", encoding="utf-8").read().split("rows = []\nfor f in")[0])
idx = {rid(i): i for i in range(N)}
w_all = np.load(os.path.join(env.LOCAL, "linker_v10", "w_v10.npy")); W = dict(zip(FEATS10, w_all))
print("v10 weights (ID/number related):", {k: round(float(W[k]), 2) for k in ("xf", "lat", "d1", "d2w", "d2c", "d3", "ldn", "isprev", "dsdev", "dsna")})
for r in ("F13_242", "F13_244", "F47_231", "F47_191"):
    b = idx[r]; t = TP[b]
    lab = TRAIN.copy(); lab[b] = False
    D = feats(b, lab, set())
    if len(D) == 0: print(r, "no candidates"); continue
    D["score"] = D[FEATS10].values.astype(float) @ w_all
    D["id_part"] = sum(D[k] * W[k] for k in ("xf", "lat", "d1", "d2w", "d2c", "d3", "ldn", "isprev", "dsdev", "dsna"))
    D["dv"] = np.minimum(D[[f"v{j}" for j in range(8)]].values, 6).sum(1)
    D["fp9"] = [np.nanmean(np.minimum(np.abs(FPA[a, :10] - FPA[b, :10]), 4)) for a in D.a.astype(int)]
    D["rec"] = [rid(a) for a in D.a.astype(int)]; D["day"] = [DAY[a] for a in D.a.astype(int)]
    top = D.sort_values("score", ascending=False).head(4)
    show = pd.concat([top, D[D.a == t]]).drop_duplicates("rec")
    print(f"\n{r} (day {DAY[b]}) truth {rid(t)} (day {DAY[t]}, {'other ID' if FARM[t] != FARM[b] else 'same ID'})")
    print(show[["rec", "day", "xf", "fp", "fp9", "dv", "oj", "id_part", "score"]].round(2).to_string(index=False))
TLa2 = pd.read_pickle(S3 + r"\truth.pkl"); cl = [(int(b), int(a)) for a, b, c in zip(TLa2.a1, TLa2.b, TLa2.cert) if TRAIN[int(b)] and DAY[int(b)] - DAY[int(a)] == 1]
xs = [FARM[a] != FARM[b] for b, a in cl]
print(f"\nclean training links: {len(cl)} | predecessor in OTHER ID: {sum(xs)} ({np.mean(xs):.2f}) | pass-2 targets: other-ID {sum(x for (b, a), x in zip(cl, xs) if NUM[b] >= 179)} of {sum(1 for b, a in cl if NUM[b] >= 179)}")
R3 = pd.read_csv(os.path.join(env.LOCAL, "linker_v10", "id3_rows.csv")); R3["xid"] = R3.truth.str[:3] != R3.rec.str[:3]
print("ID3 judgements by truth same/other ID:", R3.groupby("xid")[["ok_full", "ok_hp2", "ok_c1"]].mean().round(3).to_dict("index"), "| counts", R3.xid.value_counts().to_dict())
