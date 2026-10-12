# -*- coding: utf-8 -*-
"""VF5 (2026-10-12 집 클로드; 사용자: "본질적인 문제를 해결해야 해") — 양방향 소거.  사전 고정 — 실행 전 커밋.
목적: 정답 있는 전날이 없는 평가 기록에 대해, '다음날 쪽' 학습 기록(학습 자료는 방향 제한 없음, 6.630)으로 같은 동을 정할 수 있는가.
대상 날짜 d = 대상 0시 외기가 정확히 같은 학습 기록의 DM11 날짜.
전날 쪽(P): VF4 와 같음(볼 수 있는 기록 = 학습 + 같은 ID 앞번호 평가).
다음날 쪽(N): d+1 날짜의 기록이 '모두 학습 기록'인 경우만(평가 기록이 섞이면 판정 안 함 — 다른 ID 평가의 존재 사용 금지).
  d 날짜의 정답 있는 학습 기록 k 에서 이어지는(정답 자정 연속 비용 ≤ 2, 상호 최선) d+1 학습 기록을 '이미 차지됨'으로 지우고,
  남는 d+1 학습 기록이 정확히 1개면 그것을 대상의 다음날로 예측.  (대상 자신의 입력은 날짜 판정에만 사용.)
보정(LOO): 진실 연결(확실)이 있는 학습 기록 b 를 정답 없음으로 두고 같은 절차 — 다음날 쪽 진실 = b 를 앞으로 하는 확실 연결의 뒤 기록.
보고: 평가 60기록별 P/N 결과, N 만으로 새로 풀리는 기록, 다음날 0시 EC(= 대상 23시 근처) vs 현재 예측 23시.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
OUT = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(OUT, "..", "rise", "dm11_calendar_v1.csv")); DAY = {r: d for r, d in zip(CAL.rec, CAL.day)}
day_of = np.array([DAY.get(rid(i), np.nan) for i in range(N)])
TLa = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TLa[TLa.cert]
pred_of = {int(b): int(a) for a, b in zip(TLc.a1, TLc.b)}; succ_of = {}
for b, a in pred_of.items(): succ_of.setdefault(a, []).append(b)
pe = lambda A, i: A[i, 23] + 0.5 * (A[i, 23] - A[i, 22]); pb = lambda A, j: A[j, 0] - 0.5 * (A[j, 1] - A[j, 0])
cost = lambda j, k: abs(pe(EC, j) - pb(EC, k)) / .01 + abs(pe(ST, j) - pb(ST, k)) / .3
def tdate(t, lab):
    S0 = [j for j in range(N) if j != t and lab[j] and np.all(np.abs(X[j, 0] - X[t, 0]) < 1e-6) and np.isfinite(day_of[j])]
    return (day_of[S0[0]], S0) if S0 else (np.nan, [])
def side_next(t, lab):
    d, S0 = tdate(t, lab)
    if not np.isfinite(d): return None
    D2 = [j for j in range(N) if day_of[j] == d + 1 and j != t]
    if not D2: return dict(status="다음날 없음")
    if not all(lab[j] for j in D2): return dict(status="다음날에 평가 기록 섞임")
    taken = set()
    for k in S0:
        cj = [cost(k, j) for j in D2]; j = D2[int(np.argmin(cj))]
        if min(cj) <= 2 and min(S0, key=lambda q: cost(q, j)) == k: taken.add(j)
    left = [j for j in D2 if j not in taken]
    return dict(status="성공" if len(left) == 1 else f"남은 {len(left)}", pick=left[0] if len(left) == 1 else None, n=len(D2), taken=len(taken))
rows = []
for b in np.where(TRAIN)[0]:
    if b not in succ_of: continue
    lab = TRAIN.copy(); lab[b] = False
    r = side_next(b, lab)
    if r and r.get("pick") is not None: rows.append(dict(rec=rid(b), ok=int(r["pick"] in succ_of[b]), n=r["n"], taken=r["taken"]))
D = pd.DataFrame(rows)
print("LOO next-side unique picks:", len(D), "accuracy %.3f" % (D.ok.mean() if len(D) else np.nan)); print(D.groupby("n").ok.agg(["size", "mean"]).to_string() if len(D) else "")
V4 = pd.read_csv(os.path.join(OUT, "vf4_eval_v1.csv")).set_index("rec")
S = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); S["rec"] = S.row_id.str[:7]; S["h"] = S.row_id.str[8:].astype(int)
P23 = S[S.h == 23].set_index("rec").sub_ec; P0 = S[S.h == 0].set_index("rec").sub_ec
out = []
for r_ in V4.index:
    t = idx[r_]; lab = TRAIN.copy()
    rN = side_next(t, lab)
    pk = rid(rN["pick"]) if (rN and rN.get("pick") is not None) else None
    out.append(dict(rec=r_, prev_side=V4.at[r_, "status"], prev_pick=V4.at[r_, "pick"], next_side=(rN["status"] if rN else "날짜 못 정함"), next_pick=pk,
                    next_ec0=float(EC[idx[pk], 0]) if pk else np.nan, pred23=P23.get(r_, np.nan), pred0=P0.get(r_, np.nan)))
O = pd.DataFrame(out); O.to_csv(os.path.join(OUT, "vf5_eval_v1.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(O.round(3).to_string(index=False))
print("\nnext-side status:", O.next_side.value_counts().to_dict())
new = O[(O.next_pick.notna()) & (O.prev_pick.isna())]
print("newly resolved by next side only:", len(new), "|", ", ".join(f"{r.rec}->{r.next_pick} (next 0h EC {r.next_ec0:.2f} vs pred 23h {r.pred23:.2f})" for r in new.itertuples()))
both = O[(O.next_pick.notna()) & (O.prev_pick.notna())]
print("resolved both sides:", len(both))
