# -*- coding: utf-8 -*-
"""NA1 (2026-10-12 집 클로드; '뒤쪽 train 정답에 붙이기').  사전 고정 — 실행 전 커밋.
규정: 대상 t 의 행 h 는 t 의 0..h 입력 + train 전체만 사용(뒤 평가 입력·다른 ID 평가 입력 사용 안 함).
절차(EV 폴드 k, 숨김 = val + del):
  d = t 의 DM11 날짜(대상 0시 외기 = train 날짜와 같음으로 정해짐).  D2 = 숨기지 않은 train 기록 중 날짜 d+1.  비면 대상 아님.
  '차지됨' = 숨기지 않은 train 기록 중 날짜 d 인 기록 k 에서 정답 자정 연속(비용 ≤ 2, 상호 최선)으로 이어지는 D2 기록 → 제외.
  남은 후보 cand 가 비면 대상 아님.  시각 h 마다 fp_dist(t, c, h)(대상 0..h 와 후보 같은 시각 운영 지문) 최소 후보를 고르고,
  후보 1개이거나 2등과 차 ≥ .3 일 때만 채택(HA1 과 같은 문턱).
  보정: pred_h = EA2_t,h + β_h (A − EA2_t,h), A = 고른 후보의 0시 정답 EC, β_h 는 다른 폴드 1차(번호<179) 채택 행의 최소제곱(시각 블록 0~5/6~11/12~17/18~23 별), [0,1] 자르기.
판정: EV 대상 기록(채택 시각 ≥ 1) 전체 RMSE 가 3시드 모두 기준보다 좋음 AND 덩어리 부트스트랩 P(worse) < .025 AND 최대 기록 몫 ≤ .5. 1차·2차 따로 보고.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
day_of = np.array([DAY.get(rid(i), np.nan) for i in range(N)])
pe = lambda A, i: A[i, 23] + 0.5 * (A[i, 23] - A[i, 22]); pb = lambda A, j: A[j, 0] - 0.5 * (A[j, 1] - A[j, 0])
cost = lambda j, k: abs(pe(EC, j) - pb(EC, k)) / .01 + abs(pe(ST, j) - pb(ST, k)) / .3
Fv = pd.read_csv(os.path.join(H_, "ev_validator", "ev_folds_v1.csv")); V = Fv[Fv.role == "val"].set_index("rec")
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLa[TLa.cert]; succ = {}
for a, b in zip(TLc.a1, TLc.b): succ.setdefault(int(a), set()).add(int(b))
picks = []
for k in sorted(Fv.fold.unique()):
    hid = {idx[r] for r in Fv[Fv.fold == k].rec if r in idx}
    vis = np.array([TRAIN[i] and i not in hid for i in range(N)])
    for r_ in V[V.fold == k].index:
        t = idx[r_]; d = day_of[t]
        if not np.isfinite(d): continue
        D2 = [j for j in np.where(vis)[0] if day_of[j] == d + 1]
        if not D2: continue
        S0 = [j for j in np.where(vis)[0] if day_of[j] == d and j != t]
        taken = set()
        for q in S0:
            cj = [cost(q, j) for j in D2]; j = D2[int(np.argmin(cj))]
            if min(cj) <= 2 and min(S0, key=lambda z: cost(z, j)) == q: taken.add(j)
        cand = [j for j in D2 if j not in taken]
        if not cand: continue
        for h in range(24):
            sc = np.nan_to_num(np.array([F6.fp_dist(t, c, h) for c in cand]), nan=99); o = np.argsort(sc)
            gap = sc[o[1]] - sc[o[0]] if len(cand) > 1 else 9
            if gap >= .3:
                c = cand[o[0]]; picks.append(dict(t=r_, fold=k, h=h, pick=rid(c), A=float(EC[c, 0]), ok=(int(c in succ.get(t, set())) if t in succ else -1), ncand=len(cand)))
    print("fold", k, "picks", len(picks), flush=True)
Pk = pd.DataFrame(picks); Pk.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "na1_picks_v1.csv"), index=False)
K = Pk[Pk.ok >= 0]
print("targets with >=1 accepted hour:", Pk.t.nunique(), "| pass-2:", Pk[Pk.t.str[4:7].astype(int) >= 179].t.nunique(),
      "| pick accuracy where truth known: rows %.3f (n %d), records %.3f" % (K.ok.mean() if len(K) else np.nan, len(K), K.groupby("t").ok.mean().mean() if len(K) else np.nan))
blk = lambda h: h // 6
Y = R.groupby(["rec", "hour"]).y.first().unstack(); out = {}; base = {}
for sd, G in R.groupby("seed"):
    P = G.pivot_table(index="rec", columns="hour", values="EA2"); base[sd] = P; new = P.copy()
    for k in Pk.fold.unique():
        oth = Pk[(Pk.fold != k) & (Pk.t.str[4:7].astype(int) < 179)]; beta = {}
        for b in range(4):
            o = oth[oth.h // 6 == b]
            x = np.array([r.A - P.at[r.t, r.h] for r in o.itertuples()]); y = np.array([Y.at[r.t, r.h] - P.at[r.t, r.h] for r in o.itertuples()])
            beta[b] = float(np.clip((x * y).sum() / (x * x).sum(), 0, 1)) if len(x) > 5 else 0.0
        for r in Pk[Pk.fold == k].itertuples(): new.at[r.t, r.h] = P.at[r.t, r.h] + beta[r.h // 6] * (r.A - P.at[r.t, r.h])
    out[sd] = new; print("seed", sd, "example betas (last fold):", {b: round(v, 2) for b, v in beta.items()})
rm = lambda e: float(np.sqrt(np.mean(np.square(e)))); rng = np.random.default_rng(20261236)
for part, cond in (("all", lambda s: True), ("pass-1", lambda s: int(s[4:7]) < 179), ("pass-2", lambda s: int(s[4:7]) >= 179)):
    recs = sorted(r for r in Pk.t.unique() if cond(r))
    if not recs: print(part, "none"); continue
    cells = [rm((out[s].loc[recs] - Y.loc[recs]).values) / rm((base[s].loc[recs] - Y.loc[recs]).values) - 1 for s in out]
    per = pd.Series(np.mean([(((out[s].loc[recs] - Y.loc[recs]) ** 2).sum(1) - ((base[s].loc[recs] - Y.loc[recs]) ** 2).sum(1)).values for s in out], 0), index=recs)
    cl = per.groupby(V.chunk.reindex(recs).values).sum(); ii = rng.integers(0, len(cl), (20000, len(cl))); pw = float((cl.values[ii].sum(1) >= 0).mean())
    share = float(per.min() / per.sum()) if per.sum() < 0 else float("nan")
    print(f"{part}: records {len(recs)} | cells {' '.join('%+.2f%%' % (100*x) for x in cells)} | improved {int((per<0).sum())}/{len(per)} | cluster P {pw:.3f} | top share {share:.2f}")
    print("   largest:", ", ".join("%s %+.3f" % (a, b) for a, b in list(per.sort_values().items())[:3] + list(per.sort_values().items())[-3:]))
    if part == "all": print("VERDICT NA1:", "PASS" if all(x < 0 for x in cells) and pw < .025 and share <= .5 else "FAIL")
