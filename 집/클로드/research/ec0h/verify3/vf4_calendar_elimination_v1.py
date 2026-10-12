# -*- coding: utf-8 -*-
"""VF4 (2026-10-12 집 클로드; 사용자: "다른 팀들도 0.07 수준으로 올라오는 거 보니 뭐가 있긴 한 듯" → 소거를 달력으로 확장).  사전 고정.
vf3c 소거와 같되 전날 날짜 D1 을 '달력(DM11)의 하루 전'으로 찾음:
  대상 t 의 날짜 = t 의 0시 외기가 정확히 같은 '볼 수 있는' 학습 기록(또는 같은 ID 앞번호 평가 기록)의 DM11 날짜(학습 기록 우선).
  D1 = DM11 날짜가 (대상 날짜 − 1)인 볼 수 있는 기록(학습 + 같은 ID 앞번호 평가); 다른 ID 평가 기록은 보지 않음(규정).
  taken / left 판정은 vf3c 와 동일(정답 자정 연속 비용 ≤ 2, 상호 최선).  left 가 정확히 1개면 그 기록을 전날로 예측.
  판정 신뢰: D1 '완전'(같은 날짜 기록 수 = DM11 그 날짜의 전체 기록 수, 즉 안 보이는 다른 ID 평가 기록이 D1 에 없음) 여부를 표시.
보정(LOO): 확실 연결 진실이 있는 학습 기록 b 를 정답 없음으로 두고 같은 절차(볼 수 있는 기록 = 학습 + b 와 같은 ID 앞번호 평가) — 달력 자체는
  학습 정답 연결로 만든 것이라 LOO 가 약간 낙관적(한계 기록).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
OUT = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(OUT, "..", "rise", "dm11_calendar_v1.csv")); DAY = {r: d for r, d in zip(CAL.rec, CAL.day)}
day_of = np.array([DAY.get(rid(i), np.nan) for i in range(N)]); dsize = CAL.groupby("day").size().to_dict()
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLa[TLa.cert]; truth = {int(b): int(a) for a, b in zip(TLc.a1, TLc.b)}
pe = lambda A, i: A[i, 23] + 0.5 * (A[i, 23] - A[i, 22]); pb = lambda A, j: A[j, 0] - 0.5 * (A[j, 1] - A[j, 0])
cost = lambda j, k: abs(pe(EC, j) - pb(EC, k)) / .01 + abs(pe(ST, j) - pb(ST, k)) / .3
def run(t, vis, lab):
    vi = np.where(vis)[0]
    S0 = [j for j in vi if np.all(np.abs(X[j, 0] - X[t, 0]) < 1e-6) and np.isfinite(day_of[j])]
    S0l = [j for j in S0 if lab[j]]
    if not S0: return None
    d0 = day_of[S0l[0]] if S0l else day_of[S0[0]]
    D1 = [j for j in vi if day_of[j] == d0 - 1]
    if not D1: return dict(d0=d0, nD1=0, complete=False, pick=None, ntaken=0, nleft=0, D1="")
    L0 = [k for k in S0 if lab[k]]; LD = [j for j in D1 if lab[j]]
    taken = set()
    for k in L0:
        if not LD: break
        cj = [cost(j, k) for j in LD]; j = LD[int(np.argmin(cj))]
        if min(cj) <= 2 and min(L0, key=lambda q: cost(j, q)) == k: taken.add(j)
    left = [j for j in D1 if j not in taken]
    return dict(d0=d0, nD1=len(D1), complete=len(D1) == dsize.get(d0 - 1, -1), ntaken=len(taken), nleft=len(left),
                pick=left[0] if len(left) == 1 else None, D1=" ".join(rid(j) + ("*" if lab[j] else "") + ("x" if j in taken else "") for j in D1))
rows = []
for b, a in truth.items():
    if not TRAIN[b]: continue
    lab = TRAIN.copy(); lab[b] = False
    vis = TRAIN | ((FARM == FARM[b]) & (NUM < NUM[b])); vis[b] = False
    r = run(b, vis, lab)
    if r is None or r["pick"] is None: continue
    rows.append(dict(rec=rid(b), truth=rid(a), pick=rid(r["pick"]), ok=int(r["pick"] == a), complete=r["complete"], nD1=r["nD1"], ntaken=r["ntaken"]))
D = pd.DataFrame(rows)
print("LOO unique-leftover cases:", len(D), "accuracy %.3f" % D.ok.mean())
print(D.groupby(["complete", "nD1"]).ok.agg(["size", "mean"]).round(3).to_string())
print("wrong:", D[D.ok == 0][["rec", "truth", "pick", "complete", "nD1"]].to_string(index=False))
E = pd.read_csv(os.path.join(OUT, "..", "..", "local", "linker_v7", "eval60_predecessor_v7.csv")).set_index("row")
C = pd.read_csv(os.path.join(OUT, "..", "..", "local", "linker_v7", "eval60_confident_1step_anchor_v7.csv")).set_index("row")
S = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); S["rec"] = S.row_id.str[:7]; S["h"] = S.row_id.str[8:].astype(int)
P0 = S[S.h == 0].set_index("rec").sub_ec
print("\nEVAL (visible = training + same-ID earlier eval):")
out = []
for r_ in E.index:
    t = idx[r_]; lab = TRAIN.copy(); vis = lab | ((FARM == FARM[t]) & (NUM < NUM[t])); vis[t] = False
    r = run(t, vis, lab)
    if r is None: out.append(dict(rec=r_, status="날짜 못 정함")); continue
    pk = rid(r["pick"]) if r["pick"] is not None else None
    a23 = float(EC[r["pick"], 23]) if (r["pick"] is not None and TRAIN[r["pick"]]) else np.nan
    out.append(dict(rec=r_, day=r["d0"], status=("소거 성공" if pk else f"남은 후보 {r['nleft']}"), complete=r["complete"], D1=r["D1"], pick=pk, pick_ec23=a23,
                    v7=E.at[r_, "prev_h0"], v7_p=E.at[r_, "p_h0"], anchored_now=r_ in C.index, pred0=P0.get(r_, np.nan)))
O = pd.DataFrame(out); O.to_csv(os.path.join(OUT, "vf4_eval_v1.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(O.drop(columns=["D1"]).round(3).to_string(index=False))
print("\nsummary:", O.status.value_counts().to_dict())
G = O[O.status == "소거 성공"]
print("eliminated picks: agree with v7 %d / %d | currently anchored %d | labelled picks %d | |pick EC23 - pred0| > .1: %s" % (
    int((G.pick == G.v7).sum()), len(G), int(G.anchored_now.sum()), int(G.pick_ec23.notna().sum()),
    ", ".join(f"{r.rec} ({r.pick_ec23:.2f} vs {r.pred0:.2f})" for r in G.itertuples() if np.isfinite(r.pick_ec23) and abs(r.pick_ec23 - r.pred0) > .1)))
