# -*- coding: utf-8 -*-
"""EL2 (2026-10-12 집 클로드; F47_233 처럼 '달력 소거로 정답 있는 전날이 하나로 정해지면 앵커' 를 일반 규칙으로 EV 검증).  사전 고정 — 실행 전 커밋.
EV 폴드 k: 숨김 = val + del.  볼 수 있는 기록 = 숨기지 않은 train + 같은 ID 앞번호 val(같은 폴드, 평가와 같은 조건).
대상: EA2 앵커가 없는 val 기록.  VF4 소거(DM11 달력 하루 전 = D1, 정답 자정 연속 비용 ≤ 2 상호 최선으로 '차지됨' 제거)에서
  남은 후보가 정확히 1개이고 그 기록이 정답 있는 train(숨기지 않음)이며 D1 '완전'(D1 의 보이는 기록 수 = 달력의 그 날짜 전체 기록 수)일 때만.
보정: EA2 공식 그대로 q_h = SG2_h + (A − SG2 의 0..h 누적평균), A = 고른 전날 23시 정답 EC.
판정: 대상 기록 RMSE 3시드 모두 개선 AND 덩어리 부트스트랩 P(worse) < .025 AND 최대 기록 몫 ≤ .5.  1차/2차 따로 보고, 선택 정확도(진실 연결 있는 경우) 보고.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day)); dsize = CAL.groupby("day").size().to_dict()
day_of = np.array([DAY.get(rid(i), np.nan) for i in range(N)])
pe = lambda A, i: A[i, 23] + 0.5 * (A[i, 23] - A[i, 22]); pb = lambda A, j: A[j, 0] - 0.5 * (A[j, 1] - A[j, 0])
cost = lambda j, k: abs(pe(EC, j) - pb(EC, k)) / .01 + abs(pe(ST, j) - pb(ST, k)) / .3
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
anc = R.groupby("rec").anchored.first()
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLa[TLa.cert]; truth = {int(b): int(a) for a, b in zip(TLc.a1, TLc.b)}
picks = []
for k in sorted(Fv.fold.unique()):
    hid = {idx[r] for r in Fv[Fv.fold == k].rec if r in idx}
    valk = {idx[r] for r in V[V.fold == k].index if r in idx}
    for r_ in V[V.fold == k].index:
        if bool(anc.get(r_, True)): continue
        t = idx[r_]
        vis = np.array([(TRAIN[i] and i not in hid) or (i in valk and FARM[i] == FARM[t] and NUM[i] < NUM[t]) for i in range(N)]); vis[t] = False
        lab = np.array([TRAIN[i] and i not in hid for i in range(N)])
        S0 = [j for j in np.where(vis)[0] if np.all(np.abs(X[j, 0] - X[t, 0]) < 1e-6) and np.isfinite(day_of[j])]
        if not S0: continue
        S0l = [j for j in S0 if lab[j]]; d0 = day_of[S0l[0]] if S0l else day_of[S0[0]]
        D1 = [j for j in np.where(vis)[0] if day_of[j] == d0 - 1]
        if not D1 or len(D1) != dsize.get(d0 - 1, -1): continue
        LD = [j for j in D1 if lab[j]]; taken = set()
        for q in S0l:
            if not LD: break
            cj = [cost(j, q) for j in LD]; j = LD[int(np.argmin(cj))]
            if min(cj) <= 2 and min(S0l, key=lambda z: cost(j, z)) == q: taken.add(j)
        left = [j for j in D1 if j not in taken]
        if len(left) != 1 or not lab[left[0]]: continue
        c = left[0]; picks.append(dict(t=r_, fold=k, pick=rid(c), A=float(EC[c, 23]), ok=(int(truth[t] == c) if t in truth else -1)))
Pk = pd.DataFrame(picks); print("targets:", len(Pk), "| pass-2:", int((Pk.t.str[4:7].astype(int) >= 179).sum()), "| accuracy where truth known: %.3f (n %d)" % (Pk[Pk.ok >= 0].ok.mean(), int((Pk.ok >= 0).sum())))
Y = R.groupby(["rec", "hour"]).y.first().unstack(); out, base = {}, {}
for sd, G in R.groupby("seed"):
    P1 = G.pivot_table(index="rec", columns="hour", values="SG2"); E2 = G.pivot_table(index="rec", columns="hour", values="EA2"); new = E2.copy()
    cm = P1.cumsum(1) / np.arange(1, 25)
    for r in Pk.itertuples(): new.loc[r.t] = P1.loc[r.t].values + (r.A - cm.loc[r.t].values)
    out[sd], base[sd] = new, E2
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261239)
for part, cond in (("all", lambda s: True), ("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
    recs = sorted(r for r in Pk.t if cond(r))
    if not recs: print(part, "none"); continue
    cells = [rm((out[s].loc[recs] - Y.loc[recs]).values) / rm((base[s].loc[recs] - Y.loc[recs]).values) - 1 for s in out]
    per = pd.Series(np.mean([(((out[s].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for s in out], 0), index=recs)
    cl = per.groupby(V.chunk.reindex(recs).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
    share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
    print(f"{part}: records {len(recs)} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P {pw:.3f} | top share {share:.2f}")
    print("   per record:", ", ".join("%s %+.3f" % (a, b) for a, b in per.sort_values().items()))
    if part == "all": print("VERDICT EL2:", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
