# -*- coding: utf-8 -*-
"""WA2 (2026-10-12 집 클로드; 손해 앵커 해부, 서술 — 정답 사용 진단).  AF1 의 손해 앵커(ΔSSE > 0, EV·DIAG10)마다:
mid_jump = |고른 전날 23시 EC − 대상 실제 0시 EC| (작으면 같은 동, 크면 다른 동/자정 변화), 손해의 시각 분해(0-5 / 6-11 / 12-17 / 18-23),
대상 하루 변화 |y23 − y0|, 날짜 일치, 진실(확실/비확실 a1).  분류: W 다른 동(mid_jump > .1) / C 같은 동 + 그날 변화(mid_jump ≤ .1).
그리고 이득 앵커와 비교해 W 를 0시 신호(dT, dST, fp0, date_ok)로 가를 수 있는지."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
here = os.path.dirname(os.path.abspath(__file__)); H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
D = pd.read_csv(os.path.join(here, "af1_v1.csv"))
TL = pd.read_pickle(S3 + r"\truth.pkl"); a1 = {rid(int(b)): (rid(int(a)), bool(c)) for a, b, c in zip(TL.a1, TL.b, TL.cert)}
R = pd.read_csv(os.path.join(H_, "ev_validator", "ec", "out_v1", "v2_ev_ec_rows.csv")); R = R[(R.arm == "BASE") & (R.seed.astype(str) != "avg")]
E2 = R.groupby(["rec", "hour"]).EA2.mean().unstack(); S2 = R.groupby(["rec", "hour"]).SG2.mean().unstack(); Y = R.groupby(["rec", "hour"]).y.first().unstack()
rows = []
for r_ in D.itertuples():
    t, j = idx[r_.rec], idx[r_.prev]; mj = abs(float(EC[j, 23]) - float(EC[t, 0]))
    blocks = {}
    if r_.world == "EV" and r_.rec in E2.index:
        d = (E2.loc[r_.rec] - Y.loc[r_.rec]) ** 2 - (S2.loc[r_.rec] - Y.loc[r_.rec]) ** 2
        blocks = {f"h{a}-{b}": float(d.loc[a:b].sum()) for a, b in ((0, 5), (6, 11), (12, 17), (18, 23))}
    tp = a1.get(r_.rec, (None, False))
    rows.append(dict(world=r_.world, rec=r_.rec, prev=r_.prev, dsse=r_.dsse, mid_jump=mj, day_change=abs(float(EC[t, 23]) - float(EC[t, 0])), cls=("W other house" if mj > .1 else "C same house, day changed"),
                     date_ok=r_.date_ok, dT=r_.dT, dST=r_.dST, fp0=r_.fp0, a1=tp[0], a1_cert=tp[1], **blocks))
T = pd.DataFrame(rows); pd.set_option("display.width", 250)
H = T[T.dsse > 0].sort_values("dsse", ascending=False)
print("HARMFUL anchors:"); print(H.round(3).to_string(index=False))
print("\nharm by class (sum dSSE):", H.groupby(["world", "cls"]).dsse.sum().round(2).to_dict())
G = T[T.dsse <= 0]
print("\nmid_jump > .1 among GAINFUL anchors:", int((G.mid_jump > .1).sum()), "/", len(G), "| among HARMFUL:", int((H.mid_jump > .1).sum()), "/", len(H))
W = T.mid_jump > .1
def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, int); ok = np.isfinite(x); x, y = x[ok], y[ok]; a, b = x[y == 1], x[y == 0]
    return float((a[:, None] > b[None, :]).mean() + .5 * (a[:, None] == b[None, :]).mean()) if len(a) and len(b) else np.nan
print("AUC for 'other house' (mid_jump > .1) across all anchors (both worlds):", " | ".join(f"{c} {auc(T[c], W):.2f}" for c in ("dT", "dST", "fp0")), "| date_ok(lower)", f"{auc(-T.date_ok, W):.2f}", f"| n other-house {int(W.sum())} of {len(T)}")
