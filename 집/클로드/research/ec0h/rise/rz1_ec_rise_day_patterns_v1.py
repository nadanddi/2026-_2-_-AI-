# -*- coding: utf-8 -*-
"""RZ1 (2026-10-12 집 클로드; 사용자: "EC가 상승하기 시작하거나 상승하는 날의 패턴은 없어? 일정 기간마다 반복된다거나").
서술 조사(학습 라벨만, 예측 특징 아님).  기존 6.4/6.210/6.466 과의 차이: 기록 순서 대신 (1) 외기 동일 묶음 = 날짜,
(2) 정답 확실 연결(truth cert, 같은 농장) 사슬 = 같은 동, (3) 연결로 날짜 묶음에 상대 날짜 번호를 매김.
사전에 적은 판독(가설 → 보여야 할 흔적):
  H1 일정 주기(양액 탱크 교체 등): 같은 동 상승 간격이 특정 값에 몰림, 날짜번호 mod P 에서 상승이 한 위상에 몰림(순열 P<.01, P=2..21 훑음 → 본페로니 20).
  H2 농장 전체 처방 변경: 같은 날짜의 다른 동들도 함께 오름 → 같은 날짜 동끼리 하루 변화 상관 > 무작위 짝.
  H3 톱니(서서히 떨어지다 한 번에 오름): 상승 전 2~3일 EC 하락 추세, 상승 전 수준이 낮음.
  H4 날씨 원인: 상승 전날/당일 일사·실내온도·환기 차이(같은 시각 비상승일 대비 z).
상승일 정의(훑음): 일평균 EC25 전날 대비 +.10/+.15/+.20/+.30; 상승 '시작'= 직전 2일 상승 없음.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
from collections import defaultdict, Counter
rng = np.random.default_rng(20261212)
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
TRi = np.where(TRAIN)[0]
# ---- date components (training records, >=90% equal outdoor cells) ----
Xf = X.reshape(N, -1); par = {i: i for i in TRi}
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for k, i in enumerate(TRi):
    eq = (np.abs(Xf[TRi[k + 1:]] - Xf[i]) < 1e-6).mean(1)
    for j in TRi[k + 1:][eq >= .9]: par[fd(j)] = fd(i)
comp = {i: fd(i) for i in TRi}
# ---- certain same-farm links among training records ----
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLa = TLa[TLa.cert & (TLa.fa == TLa.fb)]
lk = [(int(a), int(b)) for a, b in zip(TLa.a1, TLa.b) if TRAIN[int(a)] and TRAIN[int(b)]]
print("certain same-farm training links:", len(lk))
# ---- relative day numbers on the component graph (BFS, +1 per link), check consistency ----
adj = defaultdict(list)
for a, b in lk:
    ca, cb = comp[a], comp[b]
    if ca != cb: adj[ca].append((cb, 1)); adj[cb].append((ca, -1))
day = {}; bad = 0; groups = 0
for c0 in sorted(set(comp.values())):
    if c0 in day: continue
    groups += 1; day[c0] = (groups, 0); Q = [c0]
    while Q:
        c = Q.pop()
        for d, s in adj[c]:
            v = (groups, day[c][1] + s)
            if d not in day: day[d] = v; Q.append(d)
            elif day[d] != v: bad += 1
print("calendar pieces:", groups, "| inconsistent link checks:", bad)
sizes = Counter(g for g, _ in day.values()); big = [g for g, n in sizes.most_common() if n >= 10]
print("pieces with >=10 dates:", [(g, sizes[g]) for g in big][:8])
# ---- daily EC25 per training record ----
ECc = EC / (1 + .02 * (ST - 25))
DM = pd.DataFrame(dict(rec=[rid(i) for i in TRi], i=TRi, farm=FARM[TRi], num=NUM[TRi], p2=NUM[TRi] >= 179,
                       ec=np.nanmean(EC[TRi], 1), ec25=np.nanmean(ECc[TRi], 1), ec0=EC[TRi, 0], ec23=EC[TRi, 23],
                       piece=[day[comp[i]][0] for i in TRi], dnum=[day[comp[i]][1] for i in TRi])).set_index("i")
# daily inputs
DM["rad"] = np.nansum(X[TRi, :, 2], 1) if X.shape[2] > 2 else np.nan
DM["tin"] = np.nanmean(IN_[TRi, :, 0], 1)
# ---- house chains (follow certain links) ----
prev = {b: a for a, b in lk}; nxt = defaultdict(list)
for a, b in lk: nxt[a].append(b)
DM["prev"] = [prev.get(i, -1) for i in DM.index]
D1 = DM[DM.prev >= 0].copy()
D1["d_ec25"] = D1.ec25.values - DM.loc[D1.prev, "ec25"].values
D1["jump0"] = D1.ec0.values - DM.loc[D1.prev, "ec23"].values          # midnight-boundary part
D1["intra"] = D1.ec23 - D1.ec0                                           # within-day part
D1["ddays"] = D1.dnum.values - DM.loc[D1.prev, "dnum"].values
print("linked day pairs:", len(D1), "| date gap of link:", D1.ddays.value_counts().to_dict())
print("\n[0] day-to-day EC25 change: median |d| %.3f, sd %.3f; share of total change at midnight boundary vs inside day:" % (D1.d_ec25.abs().median(), D1.d_ec25.std()))
print("    |midnight jump| median %.3f, |within-day 0->23| median %.3f" % (D1.jump0.abs().median(), D1.intra.abs().median()))
# chains as ordered lists
starts = [i for i in DM.index if prev.get(i, -1) < 0]
chains = []
for s in starts:
    ch = [s]
    while len(nxt[ch[-1]]) == 1: ch.append(nxt[ch[-1]][0])
    if len(ch) >= 4: chains.append(ch)
print("chains (>=4 days):", len(chains), "| lengths:", sorted(len(c) for c in chains)[::-1][:20])
for TH in (.10, .15, .20, .30):
    print(f"\n===== rise threshold +{TH:.2f} (day-mean EC25 vs linked previous day) =====")
    rise = D1.d_ec25 >= TH; fall = D1.d_ec25 <= -TH
    print("rise days %d / %d (%.0f%%), fall days %d | pass-2 rise %d/%d" % (rise.sum(), len(D1), 100 * rise.mean(), fall.sum(), (rise & D1.p2).sum(), D1.p2.sum()))
    R_ = D1[rise]
    print("  where the rise happens: midnight jump share of rise %.0f%% (median jump %.2f, within-day %.2f)" % (
        100 * (R_.jump0.clip(lower=0).sum() / (R_.jump0.clip(lower=0) + R_.intra.clip(lower=0)).sum()), R_.jump0.median(), R_.intra.median()))
    # intervals between rises inside chains
    gaps = []; runs = []
    for ch in chains:
        flags = [bool(rise.get(i, False)) for i in ch]
        pos = [k for k, f in enumerate(flags) if f]
        gaps += list(np.diff(pos))
        r = 0
        for f in flags:
            if f: r += 1
            elif r: runs.append(r); r = 0
    g = pd.Series(gaps)
    print("  intervals between rises in same house (days):", g.value_counts().sort_index().to_dict() if len(g) else "none")
    # null: shuffle rise flags within each chain
    if len(g) >= 5:
        null = []
        for _ in range(2000):
            gg = []
            for ch in chains:
                fl = np.array([bool(rise.get(i, False)) for i in ch]); fl = rng.permutation(fl); p = np.where(fl)[0]; gg += list(np.diff(p))
            null.append(np.bincount(np.clip(gg, 0, 15), minlength=16))
        null = np.array(null); obs = np.bincount(np.clip(g.values, 0, 15), minlength=16)
        print("  interval counts obs vs shuffled mean (1..10):", " ".join(f"{k}:{obs[k]}/{null[:, k].mean():.1f}" for k in range(1, 11)))
        for k in range(1, 11):
            pk = (null[:, k] >= obs[k]).mean()
            if pk < .01: print(f"    interval {k} over-represented, P {pk:.4f}")
    print("  run lengths of consecutive rise days:", Counter(runs))
    # H1 calendar phase: rise days in big calendar pieces, phase test for period P
    for f in ("F13", "F47"):
        Z = D1[(D1.farm == f) & D1.piece.isin(big)]
        if rise[Z.index].sum() < 6: continue
        best = []
        for P in range(2, 22):
            ph = (Z.dnum % P).values; fl = rise[Z.index].values
            def stat(fl_):
                t = pd.crosstab(ph, fl_)
                if t.shape[1] < 2: return 0
                rate = t[True] / t.sum(1); return float(rate.max() - rate.mean())
            s0 = stat(fl); nl = [stat(rng.permutation(fl)) for _ in range(500)]
            best.append((P, s0, float(np.mean(np.array(nl) >= s0))))
        bp = sorted(best, key=lambda x: x[2])[:4]
        print(f"  {f} calendar-phase test (period P: excess rate, P-value; Bonferroni .0025):", ", ".join(f"P{P} {s:.2f} p{p:.3f}" for P, s, p in bp))
    # H2 farm-wide synchrony: same date (component) different houses
    same = []
    for c, G in D1.groupby([D1.farm, [comp[i] for i in D1.index]]):
        if len(G) >= 2:
            v = G.d_ec25.values
            for x in range(len(v)):
                for y in range(x + 1, len(v)): same.append((v[x], v[y]))
    if same:
        s = np.array(same); r_same = np.corrcoef(s[:, 0], s[:, 1])[0, 1]
        both = np.mean((s[:, 0] >= TH) & (s[:, 1] >= TH)); either = np.mean((s[:, 0] >= TH) | (s[:, 1] >= TH))
        rr = []
        for _ in range(2000):
            a, b = rng.choice(D1.d_ec25.values, (2, len(s))); rr.append(np.corrcoef(a, b)[0, 1])
        print("  same-date different-house pairs %d: corr of day change %.2f (random null sd %.2f) | both rise %.2f of pairs where either rises" % (len(s), r_same, np.std(rr), both / max(either, 1e-9)))
    # H3 sawtooth / precursors: trend before rise, level before rise
    pre = []
    for ch in chains:
        for k in range(3, len(ch)):
            i = ch[k]
            if i in D1.index:
                pre.append(dict(rise=bool(rise.get(i, False)), lvl=DM.at[ch[k - 1], "ec25"], tr2=DM.at[ch[k - 1], "ec25"] - DM.at[ch[k - 3], "ec25"],
                                drad=DM.at[ch[k - 1], "rad"], tin_prev=DM.at[ch[k - 1], "tin"], rad_today=DM.at[i, "rad"], prev_rise=bool(rise.get(ch[k - 1], False))))
    P_ = pd.DataFrame(pre)
    if P_.rise.sum() >= 5:
        for v in ("lvl", "tr2", "drad", "tin_prev", "rad_today"):
            a, b = P_[P_.rise][v], P_[~P_.rise][v]
            print(f"  before rise vs other days: {v:9s} {a.median():8.3f} vs {b.median():8.3f}  (Mann-Whitney AUC {((a.values[:, None] > b.values[None, :]).mean() + .5 * (a.values[:, None] == b.values[None, :]).mean()):.2f})")
        print("  P(rise | previous day rose) %.2f vs P(rise | previous no) %.2f" % (P_[P_.prev_rise].rise.mean(), P_[~P_.prev_rise].rise.mean()))
# ---- shape after a big rise: average EC25 path k days after (in chains) ----
print("\n[after a +.20 rise] mean EC25 change relative to the rise day, days +1..+5:")
paths = []
for ch in chains:
    for k, i in enumerate(ch):
        if i in D1.index and D1.at[i, "d_ec25"] >= .2 and k + 5 < len(ch):
            paths.append([DM.at[ch[k + j], "ec25"] - DM.at[i, "ec25"] for j in range(1, 6)])
if paths: print("  n %d:" % len(paths), np.round(np.nanmean(paths, 0), 3), "| median", np.round(np.nanmedian(paths, 0), 3))
# ---- per-chain series print for visual inspection (longest 8 chains) ----
print("\n[longest chains: day-mean EC25 sequence; * = +.15 rise]")
for ch in sorted(chains, key=len, reverse=True)[:8]:
    print(" ", rid(ch[0]), "->", rid(ch[-1]), "len", len(ch), ":", " ".join(("%.2f%s" % (DM.at[i, "ec25"], "*" if i in D1.index and D1.at[i, "d_ec25"] >= .15 else "")) for i in ch))
