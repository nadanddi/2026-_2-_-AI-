# -*- coding: utf-8 -*-
"""NX1 (2026-10-13 집 클로드; 사용자: '전날 없는 고EC 수준 새 각도') 서술(상한): '다음날 같은 동 학습 기록의 0시 EC' 를 거꾸로 앵커로 쓰면?
규정: 학습 데이터는 뒤 날짜도 사용 가능(10-12), 짝 고르기는 대상 0..h 입력 + 학습 입력만으로 해야 함(여기선 식별 완벽 가정 = 상한).
대상: DIAG10 360 기록. 전날 상태는 CNT2 와 같은 분류(v7 구성 앵커 여부). 다음날 = 진단 연결에서 대상이 a1 인 b(DM11 달력 차 1, 학습 기록, DIAG10 폴드 숨김 아님, 거울 G0 보임).
식: pred_h = clip(sg_h + β·(B − sg 0..h 누적평균)), B = 다음날 0시 EC, β = 1 과 .9.  보고: 분류별 다음날 보이는 기록 수, 오차 변화(전체·2차), 고EC."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v10")
from feat10 import *
import env
from feat4 import EC
CAL = pd.read_csv(r"C:\work\farmai\집\클로드\research\ec0h\rise\dm11_calendar_v1.csv"); DAYr = dict(zip(CAL.rec, CAL.day))
TLa = pd.read_pickle(os.path.join(S3, "truth.pkl")); TP = {rid(int(b)): rid(int(a)) for a, b in zip(TLa.a1, TLa.b)}
NXT = {}
for b_, a_ in TP.items():
    if DAYr.get(b_, np.nan) - DAYr.get(a_, np.nan) == 1: NXT.setdefault(a_, []).append(b_)
idx = {rid(i): i for i in range(N)}
def mirror_lab(r):
    b = idx[r]; f = FARM[b]; g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2; lo = NUM[FARM == f].min()
    s = lo + ((NUM[b] - lo) // 10) * 10; m = s + sh
    touch = ((FARM == f) & (NUM >= s - 1) & (NUM <= s + 10)); touch[(FARM == g) & (NUM >= m - 1) & (NUM <= m + 10)] = True
    return TRAIN & ~touch
L7 = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
V7 = {r: p for r, p, q, l, d in zip(L7.rec, L7.prev, L7.p, L7.prev_is_labelled, L7.dT_air) if q >= .8 and l and abs(d) <= 1.5}
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
G = O.groupby(["rec", "hour", "fold", "day"], as_index=False).agg(y=("sub_ec", "first"), sg=("sg", "mean")).sort_values(["rec", "hour"]).reset_index(drop=True)
G["cm"] = G.groupby("rec").sg.transform(lambda s: s.expanding().mean()); hidden = G.groupby("fold").rec.apply(set).to_dict()
lo_, hi_ = np.nanmin(EC[TRAIN]), np.nanmax(EC[TRAIN])
rows = []
for r, g in G.groupby("rec"):
    hs = hidden[int(g.fold.iloc[0])]; a = V7.get(r); fwd = a is not None and a not in hs
    nx = [b for b in NXT.get(r, []) if TRAIN[idx[b]] and b not in hs and mirror_lab(r)[idx[b]] and np.isfinite(EC[idx[b], 0])]
    base = np.clip(g.sg.values + (EC[idx[a], 23] - g.cm.values), lo_, hi_) if fwd else g.sg.values
    row = dict(rec=r, p2=int(r[4:7]) >= 179, fwd=fwd, has_next=len(nx) > 0, ymean=g.y.mean(), sse0=float(np.sum((base - g.y.values) ** 2)))
    for beta in (1.0, .9):
        if nx and not fwd:
            B = EC[idx[nx[0]], 0]; pr = np.clip(g.sg.values + beta * (B - g.cm.values), lo_, hi_)
        else: pr = base
        row[f"sse_b{beta}"] = float(np.sum((pr - g.y.values) ** 2))
    rows.append(row)
T = pd.DataFrame(rows); pd.set_option("display.width", 220)
for part, S in (("ALL", T), ("PASS-2", T[T.p2])):
    nf = S[~S.fwd]
    print(f"\n== {part}: records {len(S)} | no forward anchor {len(nf)} | of which next-day visible {nf.has_next.sum()} (high-EC {int((nf.has_next & (nf.ymean >= 1)).sum())})")
    for beta in (1.0, .9):
        print(f"   beta {beta}: RMSE {np.sqrt(S.sse0.sum() / (24 * len(S))):.4f} -> {np.sqrt(S[f'sse_b{beta}'].sum() / (24 * len(S))):.4f} | SSE change on next-anchored {(S[f'sse_b{beta}'] - S.sse0)[~S.fwd & S.has_next].sum():+.2f}")
print("\npass-2 records without forward anchor:"); X_ = T[T.p2 & ~T.fwd].copy(); X_["gain"] = X_.sse0 - X_["sse_b1.0"]
print(X_.sort_values("sse0", ascending=False)[["rec", "has_next", "ymean", "sse0", "gain"]].round(3).to_string(index=False))
