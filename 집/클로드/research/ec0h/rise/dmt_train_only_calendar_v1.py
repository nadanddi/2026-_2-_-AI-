# -*- coding: utf-8 -*-
"""DMT (2026-10-12 집 클로드; 규정 점검 6.752 후속) = DM11 을 학습 기록만으로 다시 만든 판: 날짜 묶음은 학습 기록끼리의 하루 외기 일치, 순수 2차 창 규칙(평가 번호 순서) 제거,
끼워 넣기는 학습 확실 연결만. 출력 dmt_calendar_v1.csv(학습 기록만) + DM11 과 날짜 비교. 평가 기록 날짜는 별도로 0시 외기가 같은 학습 기록에서 붙여야 함.
(원 DM11 설명)
DM11 (최종 달력): 1차 F13 순서 묶음(R0..R115)을 확실 연결로 확인된 '끼어든 날짜'와 합쳐 실제 날짜 순번을 매김.
끼워 넣기 규칙(사전): (1) 확실 연결 prev 가 R_k 이고 next 가 R_k+1 인 묶음 → R_k 와 R_k+1 사이(1일 끼움).
(2) 순수 2차 창(F13_216~238 묶음 11개): F13_126 묶음(R75) 다음날부터 2차 번호 순서대로, R76 앞에 끼움.
(3) prev 만 있는 묶음(F13_244: R89 다음) → R89 와 R90 사이. (4) 연결이 없는 F13_208* → 2차 번호상 F13_207(달력 93)과 F13_209(109) 사이, 위치 미확정 표시.
(5) F47_109·F47_111(R61→109→111→R64): R61 과 R64 사이 원래 R62·R63 과 겹침 → 그대로 두고 '충돌' 표시.
검산: 확실 연결이 모두 +1일이 되는지 센다."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
from collections import defaultdict
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
Xf = X.reshape(N, -1); par = list(range(N))
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for i in range(N):
    if not TRAIN[i]: continue
    eq = (np.abs(Xf[i + 1:] - Xf[i]) < 1e-6).mean(1)
    for j in np.where(eq >= .9)[0] + i + 1:
        if TRAIN[j]: par[fd(j)] = fd(i)
comp = np.array([fd(i) for i in range(N)]); G = defaultdict(list)
for i in range(N):
    if TRAIN[i]: G[comp[i]].append(i)
idx = {rid(i): i for i in range(N)}; C = lambda r: comp[idx[r]]
pos = {c: (np.median([NUM[i] for i in G[c] if FARM[i] == "F13" and NUM[i] < 179]) if any(FARM[i] == "F13" and NUM[i] < 179 for i in G[c]) else np.nan) for c in G}
ranked = sorted([c for c in G if np.isfinite(pos[c])], key=lambda c: pos[c])
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]
nxt = defaultdict(set); prv = defaultdict(set)
for a, b in zip(TLc.a1, TLc.b):
    ca, cb = comp[int(a)], comp[int(b)]
    if ca != cb: nxt[ca].add(cb); prv[cb].add(ca)
R = {c: k for k, c in enumerate(ranked)}
ins = defaultdict(list)   # rank k -> groups inserted after R_k
pure_window = []
for c in G:
    if c in R or c in pure_window: continue
    p = [R[d] for d in prv[c] if d in R]; n = [R[d] for d in nxt[c] if d in R]
    if p and n and n[0] == p[0] + 1: ins[p[0]].append(c)
    elif p and not n: ins[p[0]].append(c)
order = []
for c in ranked:
    order.append(c); order += ins[R[c]]
placed = set(order); unplaced = [c for c in G if c not in placed]
day = {c: k + 1 for k, c in enumerate(order)}
chk = [day[comp[int(b)]] - day[comp[int(a)]] for a, b in zip(TLc.a1, TLc.b) if comp[int(a)] in day and comp[int(b)] in day and comp[int(a)] != comp[int(b)]]
print("days placed", len(order), "| unplaced groups", [sorted(rid(i) for i in G[c]) for c in unplaced])
print("check: certain links day difference counts", pd.Series(chk).value_counts().sort_index().to_dict())
bad = [(rid(int(a)), rid(int(b)), day[comp[int(b)]] - day[comp[int(a)]]) for a, b in zip(TLc.a1, TLc.b) if comp[int(a)] in day and comp[int(b)] in day and comp[int(a)] != comp[int(b)] and day[comp[int(b)]] - day[comp[int(a)]] != 1]
print("links not +1:", bad)
rows = []
for c in order:
    for i in sorted(G[c], key=lambda i: (FARM[i], NUM[i])):
        rows.append(dict(day=day[c], rec=rid(i), farm=FARM[i], num=int(NUM[i]), split="train" if TRAIN[i] else "eval", pass_=2 if NUM[i] >= 179 else 1,
                         kind="1차 날짜" if c in R else ("순수 2차 창" if c in pure_window else "1차에 끼어든 날짜")))
for c in unplaced:
    for i in G[c]: rows.append(dict(day=np.nan, rec=rid(i), farm=FARM[i], num=int(NUM[i]), split="train" if TRAIN[i] else "eval", pass_=2 if NUM[i] >= 179 else 1, kind="위치 미확정"))
T = pd.DataFrame(rows); T.to_csv("dmt_calendar_v1.csv", index=False)
D11 = pd.read_csv("dm11_calendar_v1.csv").set_index("rec").day; TT = T.set_index("rec"); cmp_ = pd.DataFrame(dict(dmt=TT.day, dm11=D11.reindex(TT.index)))
cmp_["diff"] = cmp_.dmt - cmp_.dm11; print("DMT vs DM11 (training records):", cmp_["diff"].value_counts(dropna=False).head(12).to_dict())
print("records whose DMT day differs or is unknown:"); print(cmp_[(cmp_["diff"] != 0)].sort_values("dm11").to_string())
raise SystemExit
print("\ninserted (non pass-1) days:")
for c in order:
    if c not in R: print(f"  day {day[c]:3d}: {', '.join(sorted(rid(i) + ('' if TRAIN[i] else '*') for i in G[c]))}")
# number gaps in pass-1 F13 where the calendar jumps
F = T[(T.farm == "F13") & (T.pass_ == 1)].sort_values("num"); j = np.diff(F.day.values)
print("\nF13 pass-1 consecutive numbers whose dates jump by >1 day:", [(int(F.num.values[k]), int(F.num.values[k + 1]), int(j[k])) for k in range(len(j)) if j[k] > 1])
F = T[(T.farm == "F47") & (T.pass_ == 1)].sort_values("num"); j = np.diff(F.day.values)
print("F47 pass-1 consecutive numbers whose dates go backward or jump >2:", [(int(F.num.values[k]), int(F.num.values[k + 1]), int(j[k])) for k in range(len(j)) if j[k] < 0 or j[k] > 2])
for f in ("F13", "F47"):
    P = T[(T.farm == f) & (T.pass_ == 2)].sort_values("num"); jj = np.diff(P.day.dropna().values)
    print(f"{f} pass-2: day range {P.day.min():.0f}-{P.day.max():.0f}, backward steps {int((jj < 0).sum())}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for k, f in enumerate(("F13", "F47")):
    S_ = T[T.farm == f]
    for (p, sp), (col, mk, lb) in {(1, "train"): ("#9aa5b1", "o", "1차 학습"), (2, "train"): ("#2f6fb0", "o", "2차 학습"), (2, "eval"): ("#d9480f", "s", "평가")}.items():
        Z = S_[(S_.pass_ == p) & (S_.split == sp)]; ax[k].scatter(Z.num, Z.day, s=12, c=col, marker=mk, label=lb)
    Z = S_[S_.rec.isin(["F47_178", "F47_074"])]; ax[k].scatter(Z.num, Z.day, s=110, facecolors="none", edgecolors="k", label="날짜 어긋난 1차 기록")
    ax[k].axhspan(T[T.kind == "순수 2차 창"].day.min() - .5, T[T.kind == "순수 2차 창"].day.max() + .5, color="#fff3bf", zorder=0, label="순수 2차 창(1차에 없는 날짜)")
    ax[k].set_title(f"{f}: 기록 번호와 실제 날짜 순번"); ax[k].set_xlabel("기록 번호"); ax[k].grid(alpha=.3)
ax[0].set_ylabel("실제 날짜 순번 (확실 연결로 맞춘 달력)"); ax[0].legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("dm11_calendar_v1.png", dpi=110); print("saved dm11_calendar_v1.csv / .png")
