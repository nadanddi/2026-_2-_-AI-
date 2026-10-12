# -*- coding: utf-8 -*-
"""WA1 (2026-10-12 집 클로드; 사용자: '잘못 붙은 앵커의 이유 분석 + 개선 방안').  서술.
대상: 짝찾기 v7 거울 CV 선택(cv_all400_choice_v7m) 중 앵커 조건(p ≥ .8, 전날 정답 있음, 같은 ID) + 진실 확실 연결이 있는 기록.
분류(틀린 선택만): D 날짜 불일치(DM11 전날 ≠ 대상 −1) / H 진짜 전날이 그 거울 세계에서 숨겨짐 / S 날짜 맞는데 다른 동 / O 그 밖.
신호(0시): date_ok, dT(자정 실내 차, v7m 표), dST(전날 23시 배지온 정답 vs 온도 EV 0시 예측), fp0(운영 지문), dEC(전날 23시 EC vs 모델 0시 sg).
보고: 원인별 개수, 신호별 맞음/틀림 분포·AUC, 그리고 규칙 후보(사전 나열)별로 '버려지는 맞는 앵커 수 / 걸러지는 틀린 앵커 수'."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
H_ = os.path.join(env.ROOT, u"집", u"클로드", "research")
CAL = pd.read_csv(os.path.join(H_, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
L = pd.read_csv(os.path.join(env.LOCAL, "linker_v7", "cv_all400_choice_v7m.csv")).set_index("rec")
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]; truth = {rid(int(b)): rid(int(a)) for a, b in zip(TLc.a1, TLc.b)}
T = pd.read_pickle(os.path.join(H_, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); T["rec"] = T.row_id.str[:7]
TP0 = T[T.hour == 0].groupby("rec")[["v22_a", "v22_b"]].mean().mean(axis=1)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; q0 = O[O.hour == 0].groupby("rec").sg.mean()
def hidden_in_mirror(r, a):
    f = r[:3]; n = int(r[4:7]); g = "F47" if f == "F13" else "F13"; sh = -2 if f == "F13" else 2
    lo = min(int(x[4:7]) for x in L.index if x[:3] == f); s = lo + ((n - lo) // 10) * 10; m = s + sh
    fa, na = a[:3], int(a[4:7])
    return (fa == f and (s - 1 <= na <= s + 10)) or (fa == g and m - 1 <= na <= m + 10)
rows = []
for r in L.index:
    pr = L.loc[r]
    if pr.xf or pr.p < .8 or not pr.prev_is_labelled or r not in truth: continue
    a, p = truth[r], pr.prev; t, j = idx[r], idx[p]
    ok = p == a; date_ok = DAY.get(p, np.nan) == DAY.get(r, np.nan) - 1
    cause = "OK" if ok else ("D date mismatch" if not date_ok else ("H true prev hidden" if hidden_in_mirror(r, a) else ("S same date other house" if DAY.get(p) == DAY.get(a) else "O other")))
    rows.append(dict(rec=r, pick=p, truth=a, ok=int(ok), cause=cause, p=float(pr.p), date_ok=int(date_ok), dT=abs(float(pr.dT_air)) if np.isfinite(pr.dT_air) else np.nan,
                     dST=abs(float(ST[j, 23]) - TP0.get(r, np.nan)), fp0=float(np.nan_to_num(F6.fp_dist(t, j, 0), nan=6)), dEC=abs(float(EC[j, 23]) - q0.get(r, np.nan)),
                     ecjump=abs(float(EC[j, 23]) - float(EC[t, 0]))))
D = pd.DataFrame(rows); pd.set_option("display.width", 220)
print("anchor-eligible picks with certain truth:", len(D), "| correct", int(D.ok.sum()), f"({D.ok.mean():.2f})")
print("causes of wrong picks:", D[D.ok == 0].cause.value_counts().to_dict())
print("EC jump at midnight (|pick EC23 - true EC0|): correct median %.3f | wrong median %.3f" % (D[D.ok == 1].ecjump.median(), D[D.ok == 0].ecjump.median()))
def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, int); ok = np.isfinite(x); x, y = x[ok], y[ok]; a, b = x[y == 1], x[y == 0]
    return float((a[:, None] > b[None, :]).mean() + .5 * (a[:, None] == b[None, :]).mean()) if len(a) and len(b) else np.nan
print("AUC for WRONG (higher -> wrong):", " | ".join(f"{c} {auc(D[c], 1 - D.ok):.2f}" for c in ("dT", "dST", "fp0", "dEC")), "| date_ok(lower->wrong)", f"{auc(-D.date_ok, 1 - D.ok):.2f}", "| p(lower->wrong)", f"{auc(-D.p, 1 - D.ok):.2f}")
print("\nwrong picks:"); print(D[D.ok == 0][["rec", "pick", "truth", "cause", "p", "date_ok", "dT", "dST", "fp0", "dEC", "ecjump"]].round(2).to_string(index=False))
print("\nrule candidates (drop anchor if ...): dropped correct / dropped wrong / kept wrong")
rules = {"date mismatch": D.date_ok == 0, "dT>1.5": D.dT > 1.5, "dST>1.0": D.dST > 1.0, "fp0>.7": D.fp0 > .7,
         "date mismatch OR dT>1.5": (D.date_ok == 0) | (D.dT > 1.5), "date mismatch OR dST>1.0": (D.date_ok == 0) | (D.dST > 1.0),
         "date mismatch OR (dT>1.5 AND dST>.8)": (D.date_ok == 0) | ((D.dT > 1.5) & (D.dST > .8)), "dT>1.5 AND dST>.8": (D.dT > 1.5) & (D.dST > .8)}
for nm, m in rules.items():
    print(f"  {nm:38s}: {int((m & (D.ok == 1)).sum()):3d} / {int((m & (D.ok == 0)).sum()):2d} / {int((~m & (D.ok == 0)).sum()):2d}")
D.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wa1_v1.csv"), index=False)
