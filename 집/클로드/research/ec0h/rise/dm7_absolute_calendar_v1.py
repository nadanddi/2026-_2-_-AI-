# -*- coding: utf-8 -*-
"""DM7 (2026-10-12 집 클로드; 사용자: "날짜가 섞여 있는 거 제대로 정리해봐").  서술 — 기록 460개(학습+평가)에 '실제 날짜 순번'을 매김.
절차:
 1) 날짜 묶음 = 외기 24h×4 가 90% 이상 같은 기록끼리(129개).
 2) 1차 날짜 순서: 묶음 안 F13 1차 번호 중앙값(없으면 F47 1차 번호를 F13 척도로 옮긴 값) 순으로 정렬 → 임시 순번.
 3) 검산: 학습 정답 확실 연결 a→b(다음날)마다 순번 차를 셈 — 1 이 아니면 그 사이에 1차에 없는 날짜가 끼었다는 뜻.
 4) 1차 순번이 없는 묶음(순수 2차 날짜)은 확실 연결(+1/−1)과 외기 자정 연속(묶음 a 23시 → b 0시, 4변수 z 합)으로 끼울 자리를 찾음.
 5) 최종 순번(1일부터) 을 매기고, 기록별 표(dm7_calendar_v1.csv)와 그림(dm7_calendar_v1.png) 저장.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
from collections import defaultdict
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
Xf = X.reshape(N, -1); par = list(range(N))
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for i in range(N):
    eq = (np.abs(Xf[i + 1:] - Xf[i]) < 1e-6).mean(1)
    for j in np.where(eq >= .9)[0] + i + 1: par[fd(j)] = fd(i)
comp = np.array([fd(i) for i in range(N)]); G = defaultdict(list)
for i in range(N): G[comp[i]].append(i)
vis = np.ones(N, bool); gpos, mp = calendar(vis, {i: comp[i] for i in range(N)})
pos = {}
for c, v in G.items():
    a = [NUM[i] for i in v if FARM[i] == "F13" and NUM[i] < 179 and rid(i) != "F13_000"]
    b = [mp(i) for i in v if FARM[i] == "F47" and NUM[i] < 179 and rid(i) not in ("F47_178", "F47_074")]
    pos[c] = float(np.median(a)) if a else (float(np.median(b)) if b else np.nan)
ranked = sorted([c for c in G if np.isfinite(pos[c])], key=lambda c: pos[c]); rank = {c: k for k, c in enumerate(ranked)}
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]
L = [(comp[int(a)], comp[int(b)], int(a), int(b)) for a, b in zip(TLc.a1, TLc.b) if comp[int(a)] != comp[int(b)]]
dif = [rank[cb] - rank[ca] for ca, cb, a, b in L if ca in rank and cb in rank]
print("groups", len(G), "| with pass-1 position", len(ranked), "| certain links between ranked groups", len(dif), "| rank difference counts", pd.Series(dif).value_counts().sort_index().to_dict())
gaps = sorted({(rank[ca], rank[cb]) for ca, cb, a, b in L if ca in rank and cb in rank and rank[cb] - rank[ca] > 1})
print("links that skip ranks (a date not in pass 1 lies between):", gaps[:20])
# outdoor midnight continuity between groups
rep = {c: v[0] for c, v in G.items()}
def mc(ca, cb): return float(np.nansum(np.abs(X[rep[ca], 23] - X[rep[cb], 0]) / sdO))
unr = [c for c in G if c not in rank]
print("\npure pass-2 date groups (no pass-1 member):", len(unr))
for c in unr: print("   ", ", ".join(sorted(rid(i) + ("" if TRAIN[i] else "*") for i in G[c])))
# chain the unranked groups: certain links among them / to ranked
adjc = defaultdict(set)
for ca, cb, a, b in L: adjc[ca].add(("next", cb)); adjc[cb].add(("prev", ca))
print("\ncertain links touching pure groups:")
for c in unr:
    nx = [("R%d" % rank[d] if d in rank else "U:" + rid(G[d][0])) for t, d in adjc[c] if t == "next"]
    pv = [("R%d" % rank[d] if d in rank else "U:" + rid(G[d][0])) for t, d in adjc[c] if t == "prev"]
    print(f"   {rid(G[c][0]):8s} prev {pv} next {nx}")
# order pure groups by outdoor continuity: greedy chain + best insertion between consecutive ranked groups
ordered_u = []
rem = set(unr)
# start = pure group whose best predecessor (min continuity) is a ranked group
best_in = {c: min(((mc(d, c), d) for d in ranked), key=lambda x: x[0]) for c in unr}
start = min(unr, key=lambda c: best_in[c][0] if not any(mc(d, c) < best_in[c][0] for d in unr if d != c) else 9e9)
cur = start; ordered_u.append(cur); rem.discard(cur)
while rem:
    nxt = min(rem, key=lambda d: mc(cur, d)); ordered_u.append(nxt); rem.discard(nxt); cur = nxt
print("\npure groups chained by outdoor 23h->0h continuity:", " -> ".join(rid(G[c][0]) + f"({mc(a, c):.2f})" if k else rid(G[c][0]) for k, (a, c) in enumerate(zip([None] + ordered_u[:-1], ordered_u))))
# where does the chain fit? continuity from each ranked group to chain start, and chain end to each ranked group
s, e = ordered_u[0], ordered_u[-1]
cin = sorted(((mc(d, s), rank[d]) for d in ranked))[:5]; cout = sorted(((mc(e, d), rank[d]) for d in ranked))[:5]
print("best ranked predecessors of chain start (cont, rank):", [(round(a, 2), b) for a, b in cin])
print("best ranked successors of chain end (cont, rank):", [(round(a, 2), b) for a, b in cout])
typical = [mc(ranked[k], ranked[k + 1]) for k in range(len(ranked) - 1)]
print("continuity between consecutive ranked groups: median %.2f, 90%% %.2f, max %.2f" % (np.median(typical), np.quantile(typical, .9), np.max(typical)))
bigjumps = sorted(((t, k) for k, t in enumerate(typical)), reverse=True)[:6]
print("largest continuity breaks between consecutive ranked groups (cont, after rank):", [(round(t, 2), k) for t, k in bigjumps])
# final day index: insert pure chain after the ranked group with best predecessor continuity
ins = cin[0][1]
order = ranked[:ins + 1] + ordered_u + ranked[ins + 1:]
day = {c: k + 1 for k, c in enumerate(order)}
T = pd.DataFrame([dict(rec=rid(i), farm=FARM[i], num=int(NUM[i]), split="train" if TRAIN[i] else "eval", pass_=2 if NUM[i] >= 179 else 1,
                       day=day[comp[i]], group_size=len(G[comp[i]]), pure2=comp[i] in set(unr)) for i in range(N)]).sort_values(["day", "farm", "num"])
T.to_csv("dm7_calendar_v1.csv", index=False)
print("\nfinal: days", len(order), "| pure pass-2 days inserted after day", ins + 1)
for f in ("F13", "F47"):
    for p in (1, 2):
        S_ = T[(T.farm == f) & (T.pass_ == p)].sort_values("num")
        back = int((np.diff(S_.day.values) < 0).sum())
        print(f"  {f} pass-{p}: records {len(S_)} | day range {S_.day.min()}-{S_.day.max()} | backward steps along numbering {back} | records per day {S_.groupby('day').size().value_counts().to_dict()}")
# plot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for k, f in enumerate(("F13", "F47")):
    S_ = T[T.farm == f]
    for p, col in ((1, "#9aa5b1"), (2, "#2f6fb0")):
        Z = S_[(S_.pass_ == p) & (S_.split == "train")]; ax[k].scatter(Z.num, Z.day, s=10, c=col, label=f"{p}차 학습")
    Z = S_[S_.split == "eval"]; ax[k].scatter(Z.num, Z.day, s=16, c="#d9480f", marker="s", label="평가")
    Z = S_[S_.rec.isin(["F47_178", "F47_074"])]; ax[k].scatter(Z.num, Z.day, s=90, facecolors="none", edgecolors="k", label="날짜 어긋난 기록")
    ax[k].set_title(f"{f}: 기록 번호 vs 실제 날짜 순번"); ax[k].set_xlabel("기록 번호"); ax[k].grid(alpha=.3)
ax[0].set_ylabel("실제 날짜 순번(외기 묶음 기준)"); ax[0].legend(fontsize=8)
plt.rcParams["font.family"] = "Malgun Gothic"
for a in ax:
    for t in [a.title, a.xaxis.label, a.yaxis.label]: t.set_fontfamily("Malgun Gothic")
leg = ax[0].get_legend()
for t in leg.get_texts(): t.set_fontfamily("Malgun Gothic")
plt.tight_layout(); plt.savefig("dm7_calendar_v1.png", dpi=110)
print("saved dm7_calendar_v1.csv / .png")
